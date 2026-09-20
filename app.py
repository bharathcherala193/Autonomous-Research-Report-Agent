import streamlit as st

from agents import planner_node, researcher_node, writer_node, critic_node, MAX_REVISIONS
from agent import save_report
from memory import remember

st.set_page_config(page_title="Research Agent", page_icon="🤖")

st.title("🤖 Autonomous Research & Report Agent")
st.caption("Planner → Researcher → Writer → Critic, with a human approval step before saving.")

if "state" not in st.session_state:
    st.session_state.state = None
if "saved" not in st.session_state:
    st.session_state.saved = False

question = st.text_area(
    "Ask a research question",
    placeholder="Compare AWS EC2 and Google Cloud Compute Engine pricing for a small startup running a t3.medium-equivalent instance 24/7 for a month. Which is cheaper and by how much?",
    height=100,
)

run_clicked = st.button("Run Research", type="primary", disabled=not question.strip())

if run_clicked:
    st.session_state.saved = False
    state = {
        "question": question,
        "plan": "",
        "findings": "",
        "report": "",
        "critic_feedback": "",
        "approved": False,
        "revision_count": 0,
        "tool_call_count": 0,
    }

    with st.status("Planning...", expanded=False) as status:
        state.update(planner_node(state))
        status.update(label="Plan ready", state="complete")
    with st.expander("📝 Plan", expanded=False):
        st.markdown(state["plan"])

    with st.status("Researching (this can take a minute)...", expanded=False) as status:
        state.update(researcher_node(state))
        status.update(label="Research complete", state="complete")
    with st.expander("🔍 Research findings", expanded=False):
        st.markdown(state["findings"])

    for attempt in range(MAX_REVISIONS + 1):
        with st.status(f"Writing report (attempt {attempt + 1})...", expanded=False) as status:
            state.update(writer_node(state))
            status.update(label="Draft ready", state="complete")

        with st.status("Critic reviewing...", expanded=False) as status:
            state.update(critic_node(state))
            if state["approved"]:
                status.update(label="Critic approved ✅", state="complete")
            else:
                status.update(label="Critic requested changes", state="error")

        if state["approved"] or state.get("revision_count", 0) > MAX_REVISIONS:
            break

    st.session_state.state = state

if st.session_state.state:
    state = st.session_state.state

    st.subheader("✍️ Final Report")
    st.markdown(state["report"])

    if state["approved"]:
        st.success("Critic approved this report.")
    else:
        st.warning("Critic did not fully approve this report (max revisions reached). Review before saving.")

    if not state["saved"] if "saved" in state else True:
        pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Approve & Save Report", disabled=st.session_state.saved):
            result = save_report.invoke({"content": state["report"], "filename": "report.md"})
            lesson_prompt = f"In one sentence, what's a reusable lesson from solving: {state['question']}\nAnswer: {state['report']}"
            from agent import llm
            from langchain_core.messages import HumanMessage
            lesson = llm.invoke([HumanMessage(content=lesson_prompt)]).content
            remember(state["question"], lesson)
            st.session_state.saved = True
            st.success(result)
    with col2:
        if st.button("❌ Discard"):
            st.session_state.state = None
            st.session_state.saved = False
            st.rerun()

    if st.session_state.saved:
        st.info("Saved to outputs/report.md")