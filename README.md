# Autonomous Research & Report Agent

A multi-agent research system that takes a question, plans how to answer it,
researches it using real tools, writes a report, fact-checks its own report
against what it actually found, and asks a human before saving anything.

Built in five layers — Planning, Tool Use, Memory, Safety, and Multi-Agent
Orchestration — each one runnable on its own so the architecture is easy to
follow end to end.

---

## Architecture

                          User Question
                                │
                                ▼
                    ┌──────────────────────┐
                    │      Planner          │
                    │  breaks the question   │
                    │  into a 3-5 step plan  │
                    │  (checks Memory first) │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Topic Splitter       │
                    │  identifies 1-3        │
                    │  distinct SUBJECTS      │
                    │  to research (not       │
                    │  process steps)         │
                    └──────────┬────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          ┌───────────┐ ┌───────────┐ ┌───────────┐
          │ Subject 1  │ │ Subject 2  │ │ Subject 3  │
          │ Researcher │ │ Researcher │ │ Researcher │
          │  (isolated │ │  (isolated │ │  (isolated │
          │   budget)  │ │   budget)  │ │   budget)  │
          └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                │             │             │
                ▼             ▼             ▼
          ┌─────────────────────────────────────┐
          │           Tool Layer                  │
          │  web_search · fetch_page · calculator │
          │  every call passes through Guardrails │
          └──────────────────┬────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Combined Findings     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Writer           │
                    │  drafts a report using │
                    │  ONLY the findings —    │
                    │  no invented numbers    │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Critic           │◄──── REVISE + feedback
                    │  checks every claim in │       (loops back to
                    │  the report against the│        Writer, capped
                    │  findings               │        retries)
                    └──────────┬────────────┘
                               │ APPROVED
                               ▼
                    ┌──────────────────────┐
                    │   Human Approval Gate  │
                    │  "Save this report?"   │
                    │       (y/n)             │
                    └──────────┬────────────┘
                               │ yes
                 ┌─────────────┴─────────────┐
                 ▼                            ▼
        ┌─────────────────┐         ┌─────────────────┐
        │  outputs/report  │         │     Memory        │
        │      .md          │         │  one-sentence      │
        │  (saved to disk)  │         │  lesson saved for   │
        └─────────────────┘         │  next time          │
                                      └─────────────────┘

          Every tool call, denial, and human decision is
          logged to safety_log.json for a full audit trail.


          
---

## What's actually going on here

This is **not** a single LLM call with tools bolted on. It's a small team of
specialized agents, each with a narrow job, coordinated by a graph
(LangGraph) instead of one giant prompt trying to do everything at once.

| Stage | Who does it | What it actually does |
|---|---|---|
| **Plan** | Planner | Turns a vague question into a concrete numbered plan. Checks Memory first for lessons from past runs. |
| **Split** | Topic Splitter | Figures out whether the question has 1-3 distinct *subjects* (e.g. "AWS" and "GCP"), not process steps. This is what stops the Researcher from wasting its whole budget on one subject and starving the other. |
| **Research** | Researcher (×N) | Each subject gets its own fixed step budget and its own isolated instruction: research *only* this subject, don't touch the others. Ends with a forced clean summary even if the budget runs out mid-search, so a raw tool dump never leaks into the report as a "finding." |
| **Write** | Writer | Turns findings into a readable report. Explicitly told not to invent numbers that aren't in the findings. |
| **Check** | Critic | Re-reads the report next to the raw findings and either says `APPROVED` or `REVISE: <specific reason>`. If it revises, the Writer gets another shot (capped, so it can't loop forever). |
| **Approve** | Human | Nothing gets written to disk without an explicit y/n from a person. |
| **Remember** | Memory | Distills one reusable lesson from the run and saves it for the next Planner call. |

### Why split research by subject instead of just looping?

Early versions gave the Researcher one open-ended budget for the whole
question. In practice this consistently failed two ways:

- **Budget starvation** — it would re-verify the same subject 6+ times with
  slightly reworded searches and never get to the second subject.
- **Topic bleed** — mid-research it would drift and start researching the
  *other* subject inside what was supposed to be a focused task, corrupting
  both findings.

Giving each subject its own prompt, its own budget, and an explicit "do NOT
mention the other subject" instruction fixed both problems.

---

## The five layers

1. **Planning & Reasoning** — a ReAct-style loop (`agent.py`): the model
   thinks, optionally calls a tool, observes the result, and repeats until
   it has a final answer.
2. **Tool Use** — `calculator`, `web_search` (Tavily), `fetch_page`
   (fetches + strips a webpage to readable text), `save_report`.
3. **Memory** — `memory.py` keeps a simple JSON notebook. The Planner reads
   relevant past lessons before making a plan; after each run, one
   reusable lesson gets written back.
4. **Safety & Control** — `guardrails.py` enforces a hard cap on tool calls
   per run, blocks disallowed domains, rejects oversized calculator
   expressions, and requires human approval before any file is saved.
   Every decision is logged to `safety_log.json`.
5. **Multi-Agent** — `agents.py` wires Planner → Researcher → Writer →
   Critic → Save into a LangGraph graph, with the Critic able to send work
   back to the Writer.

---

## Tech stack

- **LangGraph** — the state machine that wires the nodes together
- **LangChain** — model + tool abstraction layer
- **Groq** (`openai/gpt-oss-120b`) — the LLM, via a free API tier
- **Tavily** — web search API
- **BeautifulSoup** — strips HTML down to readable text for `fetch_page`
- **Streamlit** — a simple UI (`app.py`) as an alternative to the CLI
- **python-dotenv** — loads API keys from `.env`
