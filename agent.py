import os
import ast
import operator
import requests
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from memory import remember, recall
from guardrails import check_call, ask_human_approval

load_dotenv()


def extract_text(content) -> str:
    """Some providers (e.g. Gemini) return a list of content blocks instead of
    a plain string. Normalize both to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class AgentState(TypedDict):
    question: str
    plan: str
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '0.10 * 24 * 30 * 3'.
    Supports + - * / ** and parentheses. Use it for any cost comparison."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


_tavily = TavilySearch(max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web and return short snippets with URLs.
    Use this to find pages worth reading in detail with fetch_page."""
    try:
        results = _tavily.invoke({"query": query})
        items = results.get("results", []) if isinstance(results, dict) else results
        lines = []
        for r in items:
            lines.append(f"- {r.get('title', 'no title')}: {r.get('url', '')}\n  {r.get('content', '')[:200]}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Error: {e}"


@tool
def fetch_page(url: str) -> str:
    """Fetch a webpage and return its main readable text (first ~1200 chars).
    Use this after web_search to read a specific page in detail."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:1200] if text else "No readable content found."
    except Exception as e:
        return f"Error: {e}"


@tool
def save_report(content: str, filename: str) -> str:
    """Save the final report to the outputs folder. Requires human approval —
    do not call this until you have your complete final answer ready."""
    os.makedirs("outputs", exist_ok=True)
    safe_name = os.path.basename(filename)
    path = os.path.join("outputs", safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {path}"

TOOLS = [calculator, web_search, fetch_page, save_report]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def planner_node(state: AgentState) -> dict:
    past_lessons = recall(state["question"])
    prompt = f"""You are a planning assistant. Break this question into a short,
numbered step-by-step plan (3-5 steps).

Past lessons that might help:
{past_lessons}

Question: {state['question']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    plan_text = extract_text(response.content)
    print("\n PLAN:\n", plan_text)
    return {"plan": plan_text}


def agent_node(state: AgentState) -> dict:
    system = SystemMessage(content=f"""You are a research agent.
Here is your plan:
{state['plan']}

You have EXACTLY these tools available: calculator, web_search, fetch_page, save_report.
Do not call any tool that is not in this exact list — there is no 'find' tool or any other tool.

Follow the plan step by step. Use web_search to find pages, fetch_page to read
a specific page in detail, calculator for any math, and save_report to save your
final written report to a file. When you have a final answer, respond normally
without calling a tool.""")

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    results = []
    count = state.get("tool_call_count", 0)

    for call in last_message.tool_calls:
        count += 1

        if call["name"] not in TOOLS_BY_NAME:
            reason = f"Unknown tool '{call['name']}'. Available tools are: {list(TOOLS_BY_NAME.keys())}"
            print(f" UNKNOWN TOOL: {reason}")
            results.append(ToolMessage(content=f"Error: {reason}", tool_call_id=call["id"]))
            continue

        allowed, reason = check_call(call["name"], call["args"], count)

        if not allowed:
            print(f" DENIED: {call['name']} — {reason}")
            results.append(ToolMessage(content=f"Denied: {reason}", tool_call_id=call["id"]))
            continue

        if call["name"] == "save_report":
            approved = ask_human_approval(f"Save a report to '{call['args'].get('filename')}'")
            if not approved:
                results.append(ToolMessage(content="Denied: human did not approve saving.", tool_call_id=call["id"]))
                continue

        tool_fn = TOOLS_BY_NAME[call["name"]]
        output = tool_fn.invoke(call["args"])
        print(f" TOOL CALL: {call['name']}({call['args']}) -> {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    return {"messages": results, "tool_call_count": count}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


if __name__ == "__main__":
    question = "Compare AWS EC2 and Google Cloud Compute Engine pricing for a small startup running a t3.medium-equivalent instance 24/7 for a month. Which is cheaper and by how much?"

    result = app.invoke({
        "question": question,
        "plan": "",
        "messages": [HumanMessage(content=question)],
        "tool_call_count": 0,
    })

    final_answer = extract_text(result["messages"][-1].content)
    print("\n FINAL ANSWER:\n", final_answer)

    lesson_prompt = f"In one sentence, what's a reusable lesson from solving: {question}\nAnswer: {final_answer}"
    lesson = extract_text(llm.invoke([HumanMessage(content=lesson_prompt)]).content)
    remember(question, lesson)