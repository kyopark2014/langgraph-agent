# Multi Agent

[LangGraph: Multi-Agent Workflows](https://www.youtube.com/watch?v=hvAPnpSfSGo&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&index=10)에서 설명하고 있는 3가지 multi agent에 대해 정리합니다. 

## Basic Multi-agent Collaboration

[multi-agent-collaboration.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/multi-agent-collaboration.ipynb)에서는 여러 agent들이 서로 협력하는 방법을 설명하고 있습니다. 

[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/pdf/2308.08155)을 참조하였습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/518a970a-87d8-4637-a152-f3fab96e2984)

이때의 구조는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/01ddaae6-a656-41d6-afc5-f60d4d672c32)

구현 코드는 아래와 같습니다.

```python
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()
```

## Agent Supervisor

다른 여러개의 Agent를 orchestration하는 방법에 대해 설명합니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/746af98d-1cee-4659-9783-f17d0eb0c4b1)

```python
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()
```

## Hierarchical Agent Teams

[hierarchical_agent_teams.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/hierarchical_agent_teams.ipynb)에서는 Agent Superviser가 여러개 있을때의 supervisor node를 설명하고 있습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/985e5a27-4236-427f-8337-cdbfba8d8205)

### Research Team

이때의 구현된 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/b7b85170-fc34-4425-a6ac-8dd467a5b267)

이를 구현한 코드는 아래와 같습니다. 

```python
research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("WebScraper", research_node)
research_graph.add_node("supervisor", supervisor_agent)

research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("WebScraper", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
)


research_graph.set_entry_point("supervisor")
chain = research_graph.compile()
```


### Document Writing Team

```python
authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("supervisor", doc_writing_supervisor)

# Add the edges that always occur
authoring_graph.add_edge("DocWriter", "supervisor")
authoring_graph.add_edge("NoteTaker", "supervisor")
authoring_graph.add_edge("ChartGenerator", "supervisor")

# Add the edges where routing applies
authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)

authoring_graph.set_entry_point("supervisor")
chain = authoring_graph.compile()
```

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/ee7fd3be-812a-4922-8978-908d649eb9cc)

### Add Layers

```python
super_graph = StateGraph(State)
super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
super_graph.add_node(
    "PaperWritingTeam", get_last_message | authoring_chain | join_graph
)
super_graph.add_node("supervisor", supervisor_node)

super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("PaperWritingTeam", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)
super_graph.set_entry_point("supervisor")
super_graph = super_graph.compile()
```

이때의 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/47e936fa-8acf-415d-b8c4-95469a2626c1)
