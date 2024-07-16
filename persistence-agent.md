# Persistence Agent

[persistence.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/persistence.ipynb)에서는 checkpoint를 이용해 state를 관리하는것을 보여줍니다. 이것은 agent의 메모리 역할을 합니다. 

```python
workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("action", "agent")
```

이때의 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4600a709-ec26-4684-88ed-2060b3b41813)

## Interacting with the Agent

메모리를 이용하여 context를 공유 합니다.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

app = workflow.compile(checkpointer=memory)
````
