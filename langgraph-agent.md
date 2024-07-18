# LangGraph Agent

### Tool Execution Agent의 구현

[Introduction to LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)은 Agent 종류별로 설명하고 있습니다. 또한, [agent-executor.md](./agent-executor.md)에서는 LangGraph를 이용하여 Tool을 실행하는 Agent Executor에 대해 설명하고 있습니다. 자세한 구현한 코드는 [agent-executor.ipynb](./agent-executor.ipynb)와 [lambda-chat](./lambda-chat-ws/lambda_function.py)를 참조합니다. 

Agent를 위한 Class인 AgentState와 tool을 비롯한 각 노드를 정의합니다.

```python
class ChatAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
tool_node = ToolNode(tools)

def should_continue(state: ChatAgentState) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state: ChatAgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                "다음의 Human과 Assistant의 친근한 이전 대화입니다."
                "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
        
    response = chain.invoke(state["messages"])
    return {"messages": [response]}   
```

각 Node state를 정의합니다. 

```python
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
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

Graph로 Agent를 정의하고 아래와 같이 실행합니다. 

```python
from langchain_core.messages import HumanMessage

inputs = [HumanMessage(content="강남역 맛집 알려줘")]

for event in app.stream({"messages": inputs}, stream_mode="values"):    
    event["messages"][-1].pretty_print()
```

생성된 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/9383094f-0507-4a64-96b3-278e3f6e8d3e)

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


### Checkpoint 활용

#### Breakpoints

[breakpoints.ipynb](./agent/breakpoints.ipynb)에서는 breakpoint의 개념과 사용예를 보여줍니다. 상세한 내용은 [breakpoints.md](./breakpoints.md)를 참조합니다. 

#### Checkpoint

[Checkpoint는 thread의 state](https://langchain-ai.github.io/langgraph/concepts/#checkpoints)를 의미합니다. [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/how-tos/)와 [Memory를 이용해 checkpoint](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot)를 참조하여 아래처럼 memory_task를 정의합니다. 

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory_task = SqliteSaver.from_conn_string(":memory:")
```

실제 Lambda 환경에서 구성할때에는 사용자(userId)별로 memory를 관리하여야 하므로, 아래와 같이 map_task를 정의한 후, userId 존재여부에 따라 기존 memory를 재사용할 있도록 해줍니다.

```python
map_task = dict()

if userId in map_task:  
    print('memory_task exist. reuse it!')        
    memory_task = map_task[userId]
else: 
    print('memory_task does not exist. create new one!')                
    memory_task = SqliteSaver.from_conn_string(":memory:")
    map_task[userId] = memory_task
```

[LangGraph](https://langchain-ai.github.io/langgraph/)와 같이 "action"이 호출될 때에 state machine이 멈추도록 "interrupt_before"을 설정할 수도 있습니다. 

```python
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```


### Human-in-the-loop

[Human-in-the-loop](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-4-human-in-the-loop)에서는 human의 approval을 수행할 수 있습니다. 

아래와 같이 사용자의 confirm을 받은 후에 agent_action을 수행하도록 할 수 있습니다.

```python
def execute_tools(state: AgentState):
    agent_action = state["agent_outcome"]
    response = input(prompt=f"[y/n] continue with: {agent_action}?")
    if response == "n":
        raise ValueError
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}
```


