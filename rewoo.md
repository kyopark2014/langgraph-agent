# Reasoning without Observation

[rewoo.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb)에서는 multi-step planner를 진행할때 observation없이 사용하는 방법을 설명합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/ece962bf-d13a-459a-b547-23fc1dd018fc)

planner는 task 처리 형태는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3ff28ecd-67ff-4500-a8cb-8a7758de92be)

```python
class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

def get_plan(state: ReWOO): # plan

def tool_execution(state: ReWOO): # tool

def solve(state: ReWOO):  # solve

def _get_current_task(state: ReWOO):

def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"
```

이때의 Graph 구성은 아래와 같습니다. 

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

app = graph.compile()
```
