# Essay Writer

## Reflection을 이용한 Easy Writer의 구현

[essay-writer.ipynb](./agent/essay-writer.ipynb)에서는 Easy Writer을 실행해보고 동작을 확인할 수 있습니다. Easy Writer는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)에서 구현된 코드를 확인할 수 있습니다. [deep learning.ai의 Essay Writer](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/7/essay-writer)는 LangGrap의 workflow를 이용하여 주어진 주제에 적합한 Essay를 작성할 수 있도록 도와줍니다. [reflection-agent.md](./reflection-agent.md)와의 차이점은 reflection agend에서는 외부 검색없이 reflection을 이용해 LLM으로 생성된 essay를 업데이트 하는것에 비해 easy writer에서는 인터넷 검색에 필요한 keyword를 reflection으로 업데이트하고 있습니다. 성능은 검색된 데이터의 질과 양에 따라 달라지므로 성능의 비교보다는 workflow를 이해하는 용도로 활용합니다. 

Easy writer의 activity diagram은 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/ad19e5b8-6c02-4ce5-963d-1d9ed889c39e)



## Easy Writer

[Essay Writer](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/7/essay-writer)의 내용을 정리합니다.

전체 Graph의 구성도는 아래와 같습니다.

<img src="https://github.com/kyopark2014/llm-agent/assets/52392004/e99efd4a-10ff-41e9-9d3b-5f2dcd2d341a" width="300">


먼저 class와 프롬프트를 정의합니다.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
```

class와 함수를 구성합니다.

```python
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]

from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
```

아래와 같이 Graph를 구성합니다.

```python
builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)
```

그래프를 확인합니다.

```python
from IPython.display import Image

Image(graph.get_graph().draw_png())
```

![image](https://github.com/user-attachments/assets/f7cce78b-2339-454c-8286-f30739ecc392)


아래와 같이 실행합니다.

```pyhton
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
```

Interface는 아래와 같습니다.

```python
import warnings
warnings.filterwarnings("ignore")

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()
```
