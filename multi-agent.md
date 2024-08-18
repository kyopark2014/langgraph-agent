# Multi Agent

[LangGraph: Multi-Agent Workflows](https://www.youtube.com/watch?v=hvAPnpSfSGo&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&index=10)에서 설명하고 있는 3가지 multi agent에 대해 정리합니다. 

## Multi agent 방식으로 구현

아래에서는 research와 chart_generator agent들이 tool을 이용하여 순차적인 agent 동작을 수행하는것을 설명합니다. 상세한 코드는 [multi-agent-collaboration.ipynb](./multi-agent-collaboration.ipynb)와 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다.

1) research agent는 사용자의 질문으로 적절한 정보를 tavily search를 통해 얻어옵니다. 검색한 정보가 충분하지 않다면 결과를 얻을때까지 반복합니다.
2) 웹검색을 통해 얻언 정보를 기반으로 chart_generator agent가 chart에 필요한 정보를 생성합니다. 생성된 chart정보가 충분치 않을때에는 반복하여 chart_generator agent가 반복하여 동작을 수행합니다.
   
![image](https://github.com/user-attachments/assets/361e4034-6d6a-457f-b376-ce2e4b4f5c74)


tool을 정의합니다.

```python
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
```

Graph State를 정의합니다.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
```

agent_node()와 create_agent()을 정의합니다. 

```python
import functools
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

def create_agent(llm, tools, system_message: str):
    tool_names = ", ".join([tool.name for tool in tools])
    print("tool_names: ", tool_names)
                           
    system = (
        "You are a helpful AI assistant, collaborating with other assistants."
        "Use the provided tools to progress towards answering the question."
        "If you are unable to fully answer, that's OK, another assistant with different tools "
        "will help where you left off. Execute what you can to make progress."
        "If you or any of the other assistants have the final answer or deliverable,"
        "prefix your response with FINAL ANSWER so the team knows to stop."
        "You have access to the following tools: {tool_names}."
        "{system_message}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=tool_names)
    
    return prompt | llm.bind_tools(tools)
```

Research와 Chart를 위한 agent node를 정의합니다. 

```python
# Research agent and node
research_agent = create_agent(
    chat,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    chat,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")
```

tool node를 정의합니다. 

```python
tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)
```

conditional edge를 위해 route와 route3를 정의합니다. 

```python
def router(state) -> Literal["call_tool", "end", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"

def router3(state):
    sender = state["sender"]
        
    return sender
```

Workflow를 위한 Graph를 준비합니다.

```python
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "chart_generator", 
        "call_tool": "call_tool", 
        "end": END
    },
)

workflow.add_conditional_edges(
    "chart_generator",
    router,
    {
        "continue": "Researcher", 
        "call_tool": "call_tool", 
        "end": END
    },
)

workflow.add_conditional_edges(
    "call_tool",
    router3,
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)

workflow.add_edge(START, "Researcher")
graph = workflow.compile()
```

Graph를 그려봅니다.

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![image](https://github.com/user-attachments/assets/88121aed-c94b-479c-8fc5-9458c0bed36c)


아래와 같이 실행해봅니다.

```python
qurry = "대한민국의 지난 10년간의 GDP를 찾으세요. 다음에 이 데이터를 가지고 line graph를 그리세요. 준비가 다 되면 마칩니다."
config = {"recursion_limit": 150}

events = graph.stream({"messages": [HumanMessage(content=qurry)]}, config)

for s in events:
    print(s)
    print("----")
```

이때의 최종 결과는 아래와 같습니다.

```java
{'Researcher': {'messages': [AIMessage(content="The search results provide data on South Korea's GDP growth rate and GDP values over the past 10 years. Here are the key GDP data points from the last 10 years:\n\n2022: GDP $1,673.92 billion, growth rate 2.61%\n2021: GDP $1,818.43 billion, growth rate 4.30% \n2020: GDP $1,644.31 billion, growth rate -0.71%\n2019: GDP $1,651.42 billion, growth rate 2.24%\n2018: GDP $1,619.42 billion, growth rate 2.90%\n2017: GDP $1,530.75 billion, growth rate 3.16%\n2016: GDP $1,416.95 billion, growth rate 2.94% \n2015: GDP $1,382.83 billion, growth rate 2.79%\n2014: GDP $1,368.81 billion, growth rate 3.16%\n2013: GDP $1,305.61 billion, growth rate 2.90%\n\nTo visualize this data as a line graph:\n\nFINAL ANSWER:\n\nYear   GDP ($ billion)\n2013     1305.61\n2014     1368.81  \n2015     1382.83\n2016     1416.95\n2017     1530.75\n2018     1619.42\n2019     1651.42\n2020     1644.31\n2021     1818.43\n2022     1673.92\n\n[A line graph showing South Korea's GDP rising overall from around $1.3 trillion in 2013 to over $1.8 trillion in 2021, with a dip in 2020 likely due to the COVID-19 pandemic.]", additional_kwargs={'usage': {'prompt_tokens': 1381, 'completion_tokens': 417, 'total_tokens': 1798}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, response_metadata={'usage': {'prompt_tokens': 1381, 'completion_tokens': 417, 'total_tokens': 1798}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, name='Researcher', id='run-ec931a81-26a6-4af4-a6d9-1bf492ac6ec9-0', usage_metadata={'input_tokens': 1381, 'output_tokens': 417, 'total_tokens': 1798})], 'sender': 'Researcher'}}
```


## Multi-agent Collaboration

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
