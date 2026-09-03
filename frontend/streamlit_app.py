"""
Phase 4.2 -- Streamlit frontend.

Run with:  streamlit run frontend/streamlit_app.py
Expects the FastAPI backend running at API_BASE (default http://localhost:8000).
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("T2SQL_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Text-to-SQL", page_icon="🗃️", layout="wide")
st.title("🗃️ Text-to-SQL Interface")
st.caption("Natural language → guarded, validated SQL, with a confidence score you can trust.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Session history")
    if not st.session_state.history:
        st.caption("No queries yet this session.")
    for h in reversed(st.session_state.history[-15:]):
        with st.expander(h["question"][:60], expanded=False):
            st.code(h.get("sql") or "(no SQL generated)", language="sql")
            if h.get("confidence"):
                st.caption(f"Confidence: {h['confidence']:.0%}")

    st.divider()
    run_cross_check = st.checkbox("Run cross-check (2nd independent query)", value=False)
    st.caption("Disabling skips Phase 3.3 multi-query agreement, but is faster/cheaper.")
question = st.text_input(
    "Ask a question about the data",
    placeholder="e.g. What is the total net revenue by product category?",
)
col_a, col_b = st.columns([1, 5])
submit = col_a.button("Run", type="primary")

if submit and question.strip():
    with st.spinner("Generating SQL, running guardrails, executing, and cross-checking..."):
        try:
            resp = requests.post(
                f"{API_BASE}/v1/query",
                json={"question": question, "run_cross_check": run_cross_check},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            data = None

    if data:
        st.session_state.history.append({
            "question": question,
            "sql": data.get("sql"),
            "confidence": data["confidence"]["overall"] if data.get("confidence") else None,
            "query_id": data.get("query_id"),
        })

        status = data["status"]

        if status == "clarification_needed":
            st.warning(f"**Clarification needed:** {data['clarification']['reason']}")
            for opt in data["clarification"]["options"]:
                st.markdown(f"**{opt['interpretation']}**")
                st.code(opt["example_sql"], language="sql")

        elif status == "blocked":
            st.error("🚫 Query blocked by guardrails before execution.")
            st.code(data.get("sql", ""), language="sql")
            for v in data["guardrail"]["violations"]:
                st.markdown(f"- **{v['rule']}**: {v['detail']}")

        elif status == "error":
            st.error(f"Execution error: {data.get('error')}")
            st.code(data.get("sql", ""), language="sql")

        elif status == "ok":
            left, right = st.columns([3, 1])

            with left:
                st.markdown("**Generated SQL**")
                st.code(data["sql"], language="sql")
                st.markdown(f"*{data['explanation']}*")

                st.markdown("**Results**")
                result = data["result"]
                if result["rows"]:
                    df = pd.DataFrame(result["rows"], columns=result["columns"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Query returned no rows.")
                st.caption(
                    f"{result['row_count']} rows · {result['execution_time_ms']} ms"
                    + (" · truncated to row limit" if result["truncated"] else "")
                )

                if data.get("alternative_sql"):
                    with st.expander("🔀 Independent cross-check query (Phase 3.3)"):
                        st.code(data["alternative_sql"], language="sql")
                        if data.get("alternative_result"):
                            alt_df = pd.DataFrame(
                                data["alternative_result"]["rows"],
                                columns=data["alternative_result"]["columns"],
                            )
                            st.dataframe(alt_df, use_container_width=True)

                if data.get("sanity_flags"):
                    st.markdown("**⚠️ Flags**")
                    for f in data["sanity_flags"]:
                        st.markdown(f"- **{f['check']}**: {f['detail']}")

            with right:
                conf = data["confidence"]
                overall_pct = conf["overall"]
                st.metric("Confidence", f"{overall_pct:.0%}")
                st.progress(min(1.0, max(0.0, overall_pct)))
                st.markdown("**Breakdown**")
                for label, key in [
                    ("Syntax valid", "syntax_valid"),
                    ("Back-translation alignment", "back_translation_alignment"),
                    ("Result sanity", "result_sanity"),
                    ("Cross-check agreement", "multi_query_agreement"),
                    ("Schema coverage", "schema_coverage"),
                ]:
                    st.caption(f"{label}: {conf[key]:.0%}")

                st.divider()
                st.markdown("**Was this correct?**")
                fb1, fb2 = st.columns(2)
                if fb1.button("👍 Correct", key=f"up_{data.get('query_id')}"):
                    requests.post(f"{API_BASE}/v1/feedback",
                                  json={"query_id": data["query_id"], "correct": True})
                    st.success("Thanks -- logged for the eval flywheel.")
                if fb2.button("👎 Incorrect", key=f"down_{data.get('query_id')}"):
                    requests.post(f"{API_BASE}/v1/feedback",
                                  json={"query_id": data["query_id"], "correct": False})
                    st.info("Logged -- this becomes a candidate regression test case.")

st.divider()
with st.expander("📖 Database schema"):
    try:
        schema = requests.get(f"{API_BASE}/v1/schema", timeout=10).json()
        for table_name, info in schema.items():
            st.markdown(f"**{table_name}** ({info['row_count']} rows)")
            cols = [f"{c['name']} ({c['data_type']})" for c in info["columns"]]
            st.caption(", ".join(cols))
    except Exception as e:
        st.caption(f"Could not load schema: {e}")
