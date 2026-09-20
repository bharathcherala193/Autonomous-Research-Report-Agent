import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage

from agent import calculator, web_search, fetch_page, save_report, llm, extract_text
from memory import recall, remember
from guardrails import check_call, ask_human_approval

MAX_RESEARCH_STEPS = 10
MAX_REVISIONS = 2
STEPS_PER_TOPIC = 5

RESEARCH_TOOLS = [calculator, web_search, fetch_page]
RESEARCH_TOOLS_BY_NAME = {t.name: t for t in RESEARCH_TOOLS}
llm_researcher = llm.bind_tools(RESEARCH_TOOLS)


def safe_invoke(model, messages, max_retries=4, base_delay=8):
    """Invoke an LLM, automatically waiting and retrying on rate-limit errors
    instead of crashing the whole run. Re-raises any other kind of error."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in msg:
                wait = base_delay * (attempt + 1)
                print(f"⏳ Rate limited — waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    # Final attempt — if it still fails, let the error propagate normally.
    return model.invoke(messages)


class TeamState(TypedDict):
    question: str
    plan: str
    findings: str
    report: str
    critic_feedback: str
    approved: bool
    revision_count: int
    tool_call_count: int


def planner_node(state: TeamState) -> dict:
    past_lessons = recall(state["question"])
    prompt = f"""Break this question into a short, numbered step-by-step
research plan (3-5 steps).

Past lessons that might help:
{past_lessons}

Question: {state['question']}"""
    response = safe_invoke(llm, [HumanMessage(content=prompt)])
    plan_text = extract_text(response.content)
    print("\n PLAN:\n", plan_text)
    return {"plan": plan_text}


def _research_topic(topic_prompt: str, count: int) -> tuple[str, int]:
    messages = [HumanMessage(content=topic_prompt)]

    for _ in range(STEPS_PER_TOPIC):
        try:
            response = safe_invoke(llm_researcher, messages)
        except Exception as e:
            print(f" MODEL CALL FAILED (likely hallucinated tool): {e}")
            messages.append(HumanMessage(content="""Your last attempt failed because you tried to call a
tool that does not exist. You have EXACTLY these tools: calculator, web_search, fetch_page.
Do not call any other tool. Try again using only these tools, or if you already have enough
information, just answer in plain text without calling a tool."""))
            continue

        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return extract_text(response.content), count

        for call in response.tool_calls:
            count += 1

            if call["name"] not in RESEARCH_TOOLS_BY_NAME:
                reason = f"Unknown tool '{call['name']}'. Available tools are: {list(RESEARCH_TOOLS_BY_NAME.keys())}"
                print(f" UNKNOWN TOOL: {reason}")
                messages.append(ToolMessage(content=f"Error: {reason}", tool_call_id=call["id"]))
                continue

            allowed, reason = check_call(call["name"], call["args"], count)
            if not allowed:
                print(f" DENIED: {call['name']} — {reason}")
                messages.append(ToolMessage(content=f"Denied: {reason}", tool_call_id=call["id"]))
                continue

            tool_fn = RESEARCH_TOOLS_BY_NAME[call["name"]]
            output = tool_fn.invoke(call["args"])
            print(f" TOOL CALL: {call['name']}({call['args']}) -> {output}")
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        time.sleep(3)

    tool_outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
    summary_context = "\n\n".join(tool_outputs) if tool_outputs else "No research was gathered."

    fallback_prompt = f"""Based on this research data, state the single best answer
(price, fact, or figure) you can find, with its source, in one or two plain sentences.
Do not call any tools — just answer in plain text.

Original task: {topic_prompt}

Research data collected:
{summary_context}"""

    try:
        final = safe_invoke(llm, [HumanMessage(content=fallback_prompt)])
        return extract_text(final.content), count
    except Exception as e:
        print(f" FINAL FALLBACK ALSO FAILED: {e}")
        return f"Research incomplete due to a model error. Raw data collected:\n{summary_context[:500]}", count


def researcher_node(state: TeamState) -> dict:
    count = state.get("tool_call_count", 0)

    topic_prompt = f"""Look at this question. Identify the distinct SUBJECTS being
compared or asked about — for example, if the question compares two products,
services, or providers, each one is its own subject. If the question asks for a
single calculation with no comparison, there is only ONE subject: the calculation
itself.

Do NOT split by process steps (like "define specs", "gather data", "calculate
cost") — those are not subjects, they are steps every subject needs. Each subject
you list should be something you could hand to a researcher who then finds ALL
the facts needed about that one thing, pricing included.

Reply with ONLY a numbered list of 1-3 subjects, one per line, nothing else.

Question: {state['question']}"""

    topics_response = safe_invoke(llm, [HumanMessage(content=topic_prompt)])
    topics_text = extract_text(topics_response.content)
    topics = [line.split(".", 1)[-1].strip() for line in topics_text.strip().split("\n") if line.strip()]
    topics = topics[:3] if topics else [state["question"]]

    print("\n RESEARCH TOPICS:", topics)

    all_findings = []
    for topic in topics:
        topic_research_prompt = f"""You are a researcher. Your ONLY job right now is: {topic}

Find the specific price/fact needed AND perform any needed calculation for this
one subject only. Use web_search and fetch_page for current facts or prices.
Use calculator for any math. Do not repeat the same search query twice. Once you
have a clear, confirmed answer for this subject, stop and report it in a clear
sentence, including the specific number.

Overall question for context: {state['question']}"""

        findings, count = _research_topic(topic_research_prompt, count)
        print(f"\n FINDINGS [{topic}]:\n", findings)
        all_findings.append(f"{topic}:\n{findings}")
        time.sleep(3)

    combined = "\n\n".join(all_findings)
    return {"findings": combined, "tool_call_count": count}


def writer_node(state: TeamState) -> dict:
    feedback_note = f"\nCritic feedback to address: {state['critic_feedback']}" if state.get("critic_feedback") else ""
    prompt = f"""Write a clear, well-organized report answering this question,
using ONLY the facts below. Do not invent numbers or claims not present in the facts.

Question: {state['question']}

Facts gathered by the researcher:
{state['findings']}
{feedback_note}"""
    response = safe_invoke(llm, [HumanMessage(content=prompt)])
    report_text = extract_text(response.content)
    print("\n DRAFT REPORT:\n", report_text)
    time.sleep(3)
    return {"report": report_text}


def critic_node(state: TeamState) -> dict:
    prompt = f"""You are a strict fact-checker. Compare the REPORT to the FACTS.
If every claim in the report is supported by the facts, reply with exactly:
APPROVED

Otherwise reply with:
REVISE: <specific feedback on what is unsupported or wrong>

FACTS:
{state['findings']}

REPORT:
{state['report']}"""
    response = safe_invoke(llm, [HumanMessage(content=prompt)])
    verdict = extract_text(response.content).strip()
    print("\n🕵️ CRITIC VERDICT:\n", verdict)
    time.sleep(3)

    if verdict.upper().startswith("APPROVED"):
        return {"approved": True, "critic_feedback": ""}
    return {"approved": False, "critic_feedback": verdict, "revision_count": state.get("revision_count", 0) + 1}


def save_node(state: TeamState) -> dict:
    approved = ask_human_approval("Save the final approved report to outputs/report.md?")
    if approved:
        result = save_report.invoke({"content": state["report"], "filename": "report.md"})
        print(result)
    lesson_prompt = f"In one sentence, what's a reusable lesson from solving: {state['question']}\nAnswer: {state['report']}"
    lesson = extract_text(safe_invoke(llm, [HumanMessage(content=lesson_prompt)]).content)
    remember(state["question"], lesson)
    return {}


def after_critic(state: TeamState) -> str:
    if state.get("approved"):
        return "save"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        print("⚠️ Max revisions reached — saving best-effort report anyway.")
        return "save"
    return "writer"


graph = StateGraph(TeamState)
graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)
graph.add_node("save", save_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", after_critic, {"writer": "writer", "save": "save"})
graph.add_edge("save", END)

team_app = graph.compile()


if __name__ == "__main__":
    question = "Compare AWS EC2 and Google Cloud Compute Engine pricing for a small startup running a t3.medium-equivalent instance 24/7 for a month. Which is cheaper and by how much?"

    result = team_app.invoke({
        "question": question,
        "plan": "",
        "findings": "",
        "report": "",
        "critic_feedback": "",
        "approved": False,
        "revision_count": 0,
        "tool_call_count": 0,
    })

    print("\n FINAL REPORT:\n", result["report"])