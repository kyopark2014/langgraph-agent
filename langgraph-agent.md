# LangGraph Agent

## Graph state

- input: 사용자로부터 입력으로 전달된 주요 요청을 나타내는 입력 문자열
- chat_history: 이전 대화 메시지
- intermediate_steps: Agent가 시간이 지남에 따라 취하는 행동과 관찰 사항의 목록.
- agent_outcome: Agent의 응답. AgentAction인 경우에 tool을 호출하고, AgentFinish이면 AgentExecutor를 종료함

상세한 내용은 [agent_executor/base.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb)을 참조합니다.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
```

## Node 정의

Node는 함수(Function)이나 [Runnable](https://python.langchain.com/v0.1/docs/expression_language/interface/)입니다. Action을 실행하거나 tool을 실행합니다. 

- Conditional Edge: Tool을 호출하거나 작업을 종료
- Normal Edge: tool이 호출(invoke)된 후에 normal edge는 다음에 해야할 것을 결정하는 agent로 돌아감

```python
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor

tool_executor = ToolExecutor(tools)

def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

def execute_tools(data):
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"
```

## Graph 정의

```python
from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile()
```

Agent의 실행결과는 아래와 같이 stream으로 결과를 얻을 수 있습니다.

```python
inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

이때의 Node와 Edge는 아래와 같습니다. 

```python
from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except:
    pass
```

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/d43b1f81-7d5d-4bad-abe9-f7c91acb181a)
