"""
Wraps the LLM provider and forces structured output via native tool/function
calling, rather than an `instructor`-style wrapper, to keep the dependency
footprint small.

Two providers are supported behind the same interface (generate_sql,
generate_alternative_sql, back_translate, semantic_similarity):

  - GeminiLLMClient    (default) -- Google AI Studio, has a free tier
  - AnthropicLLMClient (optional) -- set T2SQL_LLM_PROVIDER=anthropic

Pick the provider with `get_llm_client()`, which reads app.config.LLM_PROVIDER.
"""
from __future__ import annotations

import os

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
)
from app.prompt_builder import build_system_prompt
from app.schema_extractor import TableInfo
from app.schemas import ClarificationNeeded, GeneratedSQL


# ===========================================================================
# Gemini implementation (default -- free tier via Google AI Studio)
# ===========================================================================
class GeminiLLMClient:
    """
    Uses Gemini's forced function-calling mode (tool_config mode="ANY") to get
    the same "the model must return one of these structured shapes" guarantee
    the Anthropic client gets from tool_choice.
    """

    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL):
        from google import genai  # imported lazily so the package is optional
        from google.genai import types

        key = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Get a free key at https://aistudio.google.com/apikey "
                "and set GEMINI_API_KEY in your environment or .env file."
            )
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model

    # ---- tool schema helpers -----------------------------------------------
    @staticmethod
    def _generate_sql_tool():
        return {
            "name": "generate_sql",
            "description": "Return the SQL query that answers the user's question, along with metadata.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "sql": {"type": "STRING", "description": "A single read-only SQL statement (SELECT/WITH only)."},
                    "explanation": {"type": "STRING", "description": "Plain-English explanation of what the query does."},
                    "confidence": {"type": "NUMBER", "description": "Self-reported confidence 0-1 that this SQL correctly answers the question."},
                    "tables_used": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "columns_used": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["sql", "explanation", "confidence", "tables_used", "columns_used"],
            },
        }

    @staticmethod
    def _request_clarification_tool():
        return {
            "name": "request_clarification",
            "description": "Use this instead of generate_sql when the question has multiple reasonable "
                            "SQL interpretations (e.g. 'revenue' = gross vs net) and guessing would risk "
                            "answering the wrong question.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "reason": {"type": "STRING"},
                    "options": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "interpretation": {"type": "STRING"},
                                "example_sql": {"type": "STRING"},
                            },
                            "required": ["interpretation", "example_sql"],
                        },
                    },
                },
                "required": ["reason", "options"],
            },
        }

    @staticmethod
    def _alternative_sql_tool():
        return {
            "name": "generate_alternative_sql",
            "description": "Return a SECOND, independently-constructed SQL query that answers the same "
                            "question using a different valid approach, so results can be cross-checked.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "sql": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "approach_difference": {"type": "STRING"},
                },
                "required": ["sql", "explanation", "approach_difference"],
            },
        }

    @staticmethod
    def _describe_intent_tool():
        return {
            "name": "describe_query_intent",
            "description": "State, as a single natural-language question, exactly what business question this SQL query answers.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"restated_question": {"type": "STRING"}},
                "required": ["restated_question"],
            },
        }

    @staticmethod
    def _score_alignment_tool():
        return {
            "name": "score_alignment",
            "description": "Score how semantically aligned two questions are.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "score": {"type": "NUMBER", "description": "0 (unrelated) to 1 (same question)"},
                    "rationale": {"type": "STRING"},
                },
                "required": ["score", "rationale"],
            },
        }

    def _call(self, system_prompt: str | None, user_prompt: str, tool_defs: list[dict],
               allowed_names: list[str] | None = None):
        types = self._types
        tool = types.Tool(function_declarations=tool_defs)
        config_kwargs = dict(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=allowed_names or [t["name"] for t in tool_defs],
                )
            ),
            max_output_tokens=LLM_MAX_TOKENS,
        )
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return self._first_function_call(resp)

    @staticmethod
    def _first_function_call(resp):
        candidate = resp.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "function_call", None) is not None:
                fc = part.function_call
                return fc.name, dict(fc.args)
        raise ValueError(f"No function_call part in Gemini response: {resp}")

    # ---- Phase 1.4: ambiguity handling --------------------------------------
    def generate_sql(self, question: str, tables: list[TableInfo]) -> GeneratedSQL | ClarificationNeeded:
        system_prompt = build_system_prompt(tables)
        name, args = self._call(
            system_prompt,
            question,
            [self._generate_sql_tool(), self._request_clarification_tool()],
        )
        if name == "request_clarification":
            return ClarificationNeeded(**args)
        return GeneratedSQL(**args)

    # ---- Phase 3.3: independent second query for cross-checking -------------
    def generate_alternative_sql(self, question: str, tables: list[TableInfo], first_sql: str) -> dict | None:
        system_prompt = build_system_prompt(tables)
        prompt = (
            f"Original question: {question}\n\n"
            f"A first query was already written:\n{first_sql}\n\n"
            "Write a SECOND, genuinely different SQL query for the SAME question -- "
            "use a different join strategy, subquery instead of join (or vice versa), "
            "or a different aggregation path. This is for cross-checking, not the same query reformatted."
        )
        _, args = self._call(system_prompt, prompt, [self._alternative_sql_tool()])
        return args

    # ---- Phase 3.1: SQL-to-question back-translation -------------------------
    def back_translate(self, sql: str) -> str:
        _, args = self._call(
            None,
            f"What business question does this SQL query answer?\n\n{sql}",
            [self._describe_intent_tool()],
        )
        return args["restated_question"]

    def semantic_similarity(self, question_a: str, question_b: str) -> float:
        prompt = (
            f"Question A (original, asked by user): {question_a}\n"
            f"Question B (back-translated from generated SQL): {question_b}\n"
            "How well does B capture the same intent as A?"
        )
        _, args = self._call(None, prompt, [self._score_alignment_tool()])
        return float(args["score"])


# ===========================================================================
# Anthropic implementation (optional -- no free tier, set T2SQL_LLM_PROVIDER=anthropic)
# ===========================================================================
class AnthropicLLMClient:
    GENERATE_SQL_TOOL = {
        "name": "generate_sql",
        "description": "Return the SQL query that answers the user's question, along with metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single read-only SQL statement (SELECT/WITH only)."},
                "explanation": {"type": "string", "description": "Plain-English explanation of what the query does."},
                "confidence": {"type": "number", "description": "Self-reported confidence 0-1 that this SQL correctly answers the question."},
                "tables_used": {"type": "array", "items": {"type": "string"}},
                "columns_used": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sql", "explanation", "confidence", "tables_used", "columns_used"],
        },
    }

    REQUEST_CLARIFICATION_TOOL = {
        "name": "request_clarification",
        "description": "Use this instead of generate_sql when the question has multiple reasonable "
                        "SQL interpretations (e.g. 'revenue' = gross vs net) and guessing would risk "
                        "answering the wrong question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "interpretation": {"type": "string"},
                            "example_sql": {"type": "string"},
                        },
                        "required": ["interpretation", "example_sql"],
                    },
                },
            },
            "required": ["reason", "options"],
        },
    }

    ALTERNATIVE_SQL_TOOL = {
        "name": "generate_alternative_sql",
        "description": "Return a SECOND, independently-constructed SQL query that answers the same "
                        "question using a different valid approach (e.g. different JOIN order, "
                        "subquery vs JOIN, different aggregation path) so results can be cross-checked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "explanation": {"type": "string"},
                "approach_difference": {"type": "string", "description": "How this differs from a naive approach."},
            },
            "required": ["sql", "explanation", "approach_difference"],
        },
    }

    BACK_TRANSLATE_TOOL = {
        "name": "describe_query_intent",
        "description": "State, as a single natural-language question, exactly what business question this SQL query answers.",
        "input_schema": {
            "type": "object",
            "properties": {"restated_question": {"type": "string"}},
            "required": ["restated_question"],
        },
    }

    def __init__(self, api_key: str | None = None, model: str = ANTHROPIC_MODEL):
        import anthropic  # imported lazily so the package is optional

        key = api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in your environment or .env file."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def generate_sql(self, question: str, tables: list[TableInfo]) -> GeneratedSQL | ClarificationNeeded:
        system_prompt = build_system_prompt(tables)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            system=system_prompt,
            tools=[self.GENERATE_SQL_TOOL, self.REQUEST_CLARIFICATION_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": question}],
        )
        block = self._first_tool_use(resp)
        if block.name == "request_clarification":
            return ClarificationNeeded(**block.input)
        return GeneratedSQL(**block.input)

    def generate_alternative_sql(self, question: str, tables: list[TableInfo], first_sql: str) -> dict | None:
        system_prompt = build_system_prompt(tables)
        prompt = (
            f"Original question: {question}\n\n"
            f"A first query was already written:\n{first_sql}\n\n"
            "Write a SECOND, genuinely different SQL query for the SAME question -- "
            "use a different join strategy, subquery instead of join (or vice versa), "
            "or a different aggregation path. This is for cross-checking, not the same query reformatted."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            system=system_prompt,
            tools=[self.ALTERNATIVE_SQL_TOOL],
            tool_choice={"type": "tool", "name": "generate_alternative_sql"},
            messages=[{"role": "user", "content": prompt}],
        )
        block = self._first_tool_use(resp)
        return block.input

    def back_translate(self, sql: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            tools=[self.BACK_TRANSLATE_TOOL],
            tool_choice={"type": "tool", "name": "describe_query_intent"},
            messages=[{"role": "user", "content": f"What business question does this SQL query answer?\n\n{sql}"}],
        )
        block = self._first_tool_use(resp)
        return block.input["restated_question"]

    def semantic_similarity(self, question_a: str, question_b: str) -> float:
        tool = {
            "name": "score_alignment",
            "description": "Score how semantically aligned two questions are.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "description": "0 (unrelated) to 1 (same question)"},
                    "rationale": {"type": "string"},
                },
                "required": ["score", "rationale"],
            },
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            tools=[tool],
            tool_choice={"type": "tool", "name": "score_alignment"},
            messages=[{
                "role": "user",
                "content": (
                    f"Question A (original, asked by user): {question_a}\n"
                    f"Question B (back-translated from generated SQL): {question_b}\n"
                    "How well does B capture the same intent as A?"
                ),
            }],
        )
        block = self._first_tool_use(resp)
        return float(block.input["score"])

    @staticmethod
    def _first_tool_use(resp):
        for block in resp.content:
            if block.type == "tool_use":
                return block
        raise ValueError(f"No tool_use block in model response: {resp.content}")


# ===========================================================================
# Factory
# ===========================================================================
def get_llm_client():
    provider = LLM_PROVIDER.lower()
    if provider == "anthropic":
        return AnthropicLLMClient()
    if provider == "gemini":
        return GeminiLLMClient()
    raise RuntimeError(f"Unknown T2SQL_LLM_PROVIDER '{provider}'. Use 'gemini' or 'anthropic'.")


# Backwards-compatible alias -- older code / eval scripts import LLMClient directly.
LLMClient = GeminiLLMClient
