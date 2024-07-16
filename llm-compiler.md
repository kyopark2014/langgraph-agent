# LLMCompiler

[LLMCompiler.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb)에서는 "An LLM Compiler for Parallel Function Calling"을 구현한 것을 설명하고 있습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c17e641b-93eb-451d-9020-be198ae184fc)

Task fetching unit

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4daeafb1-b804-441c-91d5-dad30558c261)


```python
from langgraph.graph import MessageGraph, END
from typing import Dict

graph_builder = MessageGraph()

graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")

def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"

graph_builder.add_conditional_edges(
    start_key="join",
    condition=should_continue,
)

graph_builder.set_entry_point("plan_and_schedule")

chain = graph_builder.compile()
```
