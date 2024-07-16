# Breakpoints

[breakpoints.ipynb](./agent/breakpoints.ipynb)에서는 breakpoint의 개념과 사용예를 보여줍니다. 이 노트북의 원본은 [langchain-breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)입니다. 

## Simple Case

먼저 간단한 케이스에 대한 breakpoint 예제 입니다.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

workflow = StateGraph(State)
workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("step_3", step_3)
workflow.add_edge(START, "step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "step_3")
workflow.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add 
graph = workflow.compile(checkpointer=memory, interrupt_before=["step_3"])
```

이를 통해 구현된 workflow는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a52ab2ae-29a6-4a39-8b75-fb47fc166191)


이제 아래와 같이 입력을 지정하고 실행합니다. Step3에서 멈춘다음에 입력을 받으려고 대기 합니다. 

```python
initial_input = {"input": "hello world"}
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

user_approval = input("Do you want to go to Step 3? (yes/no): ")

if user_approval.lower() == 'yes':
    
    # If approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")
```

이때의 실행결과는 아래와 같습니다. Step 3을 실행하기 전에 멈춰선 후에 "yes"를 입력하면, breakpoint 이후로 실행을 계속합니다. 

```text
{'input': 'hello world'}
---Step 1---
---Step 2---
Do you want to go to Step 3? (yes/no):  yes
---Step 3---
```

## Agent Case

아래와 같은 Tool을 이용하는 경우에 Breakpoints를 이용할 수 있습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/af93fc2b-b37e-4897-9377-f24a15077474)

아래와 같이 tool과 node에 대한 함수를 정의합니다. 

```python
@tool
def search(query: str):
    """Call to surf the web."""
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]

tools = [search]
tool_node = ToolNode(tools)

model = chat.bind_tools(tools)

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}
```

이제, 아래와 같이 workflow를 정의합니다. 

```python
workflow = StateGraph(MessagesState)

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

memory = MemorySaver()

app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

아래와 같이 실행합니다.

```python
from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": "3"}}
inputs = [HumanMessage(content="search for the weather in sf now")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

action을 실행하기 전에 아래와 같이 멈춥니다. 

```text
================================ Human Message =================================

search for the weather in sf now
================================== Ai Message ==================================
Tool Calls:
  search (toolu_bdrk_01KJqMCKm1nd7ej6w3xayBSy)
 Call ID: toolu_bdrk_01KJqMCKm1nd7ej6w3xayBSy
  Args:
    query: san francisco weather
```    

이제 아래와 같이 Resume을 요청합니다.

```python
for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

이후 아래와 같이 나머지 동작을 수행합니다.

```text
================================= Tool Message =================================
Name: search

["It's sunny in San Francisco, but you better look out if you're a Gemini \ud83d\ude08."]
================================== Ai Message ==================================

The search results show the current weather conditions in San Francisco. It looks like it is sunny there right now. However, the results also include a humorous astrological warning for people with the Gemini zodiac sign, which doesn't seem directly relevant to the weather query.

To summarize the key information from the search:

The current weather in San Francisco is sunny.
```


