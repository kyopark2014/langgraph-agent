# Essay Writer

## Reflection을 이용한 Easy Writer의 구현

[essay-writer.ipynb](./agent/essay-writer.ipynb)에서는 Easy Writer을 실행해보고 동작을 확인할 수 있습니다. Easy Writer는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)에서 구현된 코드를 확인할 수 있습니다. [deep learning.ai의 Essay Writer](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/7/essay-writer)는 LangGrap의 workflow를 이용하여 주어진 주제에 적합한 Essay를 작성할 수 있도록 도와줍니다. [reflection-agent.md](./reflection-agent.md)와의 차이점은 reflection agend에서는 외부 검색없이 reflection을 이용해 LLM으로 생성된 essay를 업데이트 하는것에 비해 easy writer에서는 인터넷 검색에 필요한 keyword를 reflection으로 업데이트하고 있습니다. 성능은 검색된 데이터의 질과 양에 따라 달라지므로 성능의 비교보다는 workflow를 이해하는 용도로 활용합니다. 

Easy writer의 activity diagram은 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/ad19e5b8-6c02-4ce5-963d-1d9ed889c39e)

### 상세 구현

LangGraph를 위한 State Class는 아래와 같습니다.

```python
class State(TypedDict):
    task: str
    plan: list[str]
    essay: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
```

에세이의 주제를 설명하는 Plan을 아래와 같이 준비합니다. Plan 클래스는 with_structured_output를 이용해 Plan을 추출할 때에 사용합니다.

```python
class Plan(BaseModel):
    """List of session topics and outline as a json format"""

    steps: List[str] = Field(
        description="different sessions to follow, should be in sorted order without numbers. Eash session has detailed description"
    )

def get_planner():
    system = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections. \
Make sure that each session has all the information needed."""
            
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{messages}"),
        ]
    )
        
    chat = get_chat()   
        
    planner = planner_prompt | chat
    return planner

def plan(state: State):
    print("###### plan ######")
    print('task: ', state["task"])
        
    task = [HumanMessage(content=state["task"])]

    planner = get_planner()
    response = planner.invoke({"messages": task})
    print('response.content: ', response.content)
        
    chat = get_chat()
    structured_llm = chat.with_structured_output(Plan, include_raw=True)
    info = structured_llm.invoke(response.content)
    print('info: ', info)
        
    if not info['parsed'] == None:
        parsed_info = info['parsed']
        # print('parsed_info: ', parsed_info)        
        print('steps: ', parsed_info.steps)
            
        return {
            "task": state["task"],
            "plan": parsed_info.steps
        }
    else:
        print('parsing_error: ', info['parsing_error'])
            
        return {"plan": []}  
```

이때 얻어지는 데이터의 형태는 아래와 같습니다.

```java
{'task': '즐겁게 사는 방법',
 'plan': ['행복의 정의와 중요성에 대한 간략한 설명과 주제 제시',
  '긍정적인 마음가짐을 유지하기 위한 방법들 - 부정적 생각을 긍정적으로 전환, 감사하는 습관, 낙관주의와 희망적 사고의 중요성',
  '균형 잡힌 생활을 영위하기 위한 방법들 - 일과 휴식의 균형, 운동과 건강한 식습관, 취미생활과 여가활동',
  '인간관계를 소중히 여기는 방법들 - 가족/친구/동료와의 관계 돈독히 하기, 타인 배려와 이해, 사회적 유대감 형성',
  '자기계발과 성장을 추구하는 방법들 - 새로운 것 배우기와 도전, 장기 목표 설정과 성취 노력, 영적/정신적 성장의 가치',
  '주요 내용 요약 및 즐거운 삶을 위한 통합적 접근의 필요성 강조']}
```

웹검색을 위한 keyword를 생성하기 위하여 Queries 클래스를 정의한 후에 아래와 같이 Query를 추출합니다. 

```python
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]

def research_plan(state: State):
    task = state['task']
    print('task: ', task)
    
    system = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""
        
    research_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{task}"),
        ]
    )
        
    chat = get_chat()   
        
    research = research_prompt | chat
    
    response = research.invoke({"task": task})
    print('response.content: ', response.content)
    
    chat = get_chat()
    structured_llm = chat.with_structured_output(Queries, include_raw=True)
    info = structured_llm.invoke(response.content)
    # print('info: ', info)
    
    if not info['parsed'] == None:
        queries = info['parsed']
        print('queries: ', queries.queries)
        
    content = state["content"] if state.get("content") is not None else []
    search = TavilySearchResults(k=2)
    for q in queries.queries:
        response = search.invoke(q)     
        # print('response: ', response)        
        for r in response:
            content.append(r['content'])
    return {        
        "task": state['task'],
        "plan": state['plan'],
        "content": content,
    }
```

이때 얻어지는 쿼리를 위한 keyword는 아래와 같습니다. 웹검색의 결과는 content에 저장되어 generation에서 활용합니다.

```text
queries:  ['행복한 삶의 방법', '긍정적 마인드 기르기', '일상에서 행복 찾기 팁']
```

generation()에서는 Plan과 웹검색으로 얻어진 정보인 content를 이용해 에세이를 생성합니다.

```python
def generation(state: State):    
    content = "\n\n".join(state['content'] or [])
    
    system = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

<content>
{content}
</content>
"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{task}\n\nHere is my plan:\n\n{plan}"),
        ]
    )
        
    chat = get_chat()
    chain = prompt | chat

    response = chain.invoke({
        "content": content,
        "task": state['task'],
        "plan": state['plan']
    })
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "essay": response, 
        "revision_number": revision_number + 1
    }
```

이때 생성된 에세이는 아래와 같습니다. 

```java
{'essay': AIMessage(content='행복한 삶을 살기 위한 길잡이\n\n모두가 행복한 삶을 꿈꾸지만 실제로 행복을 느끼는 것은 쉽지 않습니다. 행복은 단순히 물질적 풍요나 일시적인 기쁨을 넘어서는 삶의 만족과 충만함을 의미합니다. 진정한 행복을 위해서는 우리의 마음가짐과 생활 방식에 주목해야 합니다.\n\n첫째, 긍정적인 마음가짐을 기르는 것이 중요합니다. 부정적인 생각과 감정에 사로잡히기보다는 낙관적이고 희망적인 관점을 가져야 합니다. 작은 일상의 기쁨에 감사하는 습관을 들이고, 어려운 상황에서도 긍정적인 면을 찾아보세요. \n\n둘째, 균형 잡힌 생활 리듬을 유지하는 것이 행복의 열쇠입니다. 일과 휴식의 적절한 조화를 이루고, 운동과 건강한 식습관으로 몸과 마음의 건강을 돌보세요. 또한 취미생활이나 여가활동을 통해 재미와 활력을 얻을 수 있습니다.\n\n셋째, 인간관계를 소중히 여기는 자세가 필요합니다. 가족, 친구, 동료들과 정서적 유대를 돈독히 하고 서로를 이해하고 배려하는 마음을 가져야 합니다. 타인과의 유대감은 행복감을 높이는 원천이 됩니다.\n\n넷째, 자기계발과 성장의 기회를 놓치지 마세요. 새로운 것을 배우고 도전하며 장기적인 목표를 향해 나아가는 과정에서 성취감과 보람을 느낄 수 있습니다. 영적, 정신적 성장 또한 행복에 이르는 중요한 길잡이가 될 것입니다.\n\n마지막으로 행복은 이 모든 요소들이 조화를 이룰 때 비로소 가능해집니다. 긍정적 마음가짐, 균형 잡힌 생활, 인간관계, 자기계발을 통합적으로 추구할 때 진정한 행복이 우리 곁에 머물 것입니다. 오늘부터 작은 실천으로 행복한 삶의 여정을 시작해보세요.', additional_kwargs={'usage': {'prompt_tokens': 2806, 'completion_tokens': 826, 'total_tokens': 3632}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, response_metadata={'usage': {'prompt_tokens': 2806, 'completion_tokens': 826, 'total_tokens': 3632}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, id='run-8490a66f-b9bc-4192-b6c6-7f0e34ebeb91-0', usage_metadata={'input_tokens': 2806, 'output_tokens': 826, 'total_tokens': 3632}),
 'revision_number': 2}
```

생성된 에세이를 이용하여 평가(critique)를 생성합니다.

```python
def reflection(state: State):    
    """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
                "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
                "각 문단의 길이는 최소 200자 이상이 되도록 관련된 예제를 충분히 포함합니다.",
            ),
            ("human", "{essay}"),
        ]
    )
    
    chat = get_chat()
    reflect = reflection_prompt | chat
            
    res = reflect.invoke({"essay": state['essay'].content})    
    response = HumanMessage(content=res.content)    
    
    return {
        "critique": response,
        "revision_number": int(state['revision_number'])
    }
```

평가의 결과는 예는 아래와 같습니다.

```text
'전반적으로 행복한 삶을 살기 위한 좋은 지침들을 제시하고 있습니다. 다음은 몇 가지 장점과 개선 사항에 대한 피드백입니다.\n\n장점:\n\n1. 행복의 의미를 잘 정의하고, 물질적 풍요나 일시적 기쁨을 넘어서는 삶의 만족과 충만함이라고 설명한 점이 좋습니다.\n\n2. 행복을 위한 구체적인 요소들(긍정적 마음가짐, 균형 잡힌 생활, 인간관계, 자기계발)을 잘 제시하고 있습니다. \n\n3. 각 요소에 대해 구체적인 실천 방안을 제안하고 있어 실용적입니다. 예를 들어 "작은 일상의 기쁨에 감사하는 습관"이나 "운동과 건강한 식습관"과 같은 구체적인 방법들을 언급하고 있습니다.\n\n4. 마지막 부분에서 행복의 요소들이 조화를 이루어야 함을 강조하고 있어 균형 잡힌 관점을 제시하고 있습니다.\n\n개선 사항:\n\n1. 각 문단의 길이가 다소 짧은 편입니다. 각 요소에 대해 좀 더 구체적인 예시나 설명을 추가하면 이해도를 높일 수 있을 것입니다.\n\n2. 긍정적 마음가짐을 기르는 구체적인 방법(예: 명상, 긍정적 자기 대화 등)에 대한 설명이 부족합니다.\n\n3. 인간관계 부분에서 가족, 친구 외에도 지역사회나 사회적 유대감의 중요성에 대해 언급할 수 있습니다.\n\n4. 자기계발과 성장의 구체적인 방법(예: 새로운 기술 배우기, 자원봉사 등)에 대한 설명이 부족합니다.\n\n5. 행복을 방해하는 요인들(예: 스트레스, 부정적 습관 등)에 대한 언급과 이를 극복하는 방법에 대한 조언이 추가되면 좋을 것 같습니다.\n\n전반적으로 행복한 삶을 위한 좋은 지침을 제시하고 있지만, 각 요소에 대한 구체적인 예시와 설명을 보완하고 균형 잡힌 관점을 더 강조한다면 더욱 도움이 될 것입니다.'
```

평가를 이용하여 새로운 검색 keyword를 생성하고 Tavily를 이용한 웹검색을 통해 content를 업데이트 합니다. 

```python
def research_critique(state: State):
    system = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
    
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{critique}"),
        ]
    )
    
    chat = get_chat()           
    critique = critique_prompt | chat    
    response = critique.invoke({"critique": state['critique']})
    print('response.content: ', response.content)
    
    chat = get_chat()
    structured_llm = chat.with_structured_output(Queries, include_raw=True)
    info = structured_llm.invoke(response.content)
    # print('info: ', info)
    
    content = ""
    if not info['parsed'] == None:
        queries = info['parsed']
        print('queries: ', queries.queries)
        
        content = state["content"] if state.get("content") is not None else []
        search = TavilySearchResults(k=2)
        for q in queries.queries:
            response = search.invoke(q)     
            # print('response: ', response)        
            for r in response:
                content.append(r['content'])
    return {
        "content": content,
        "revision_number": int(state['revision_number'])
    }
```

should_continue()에서는 max_revision 반복하도록 합니다. 

```python
def should_continue(state, config):
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
    print("max_revisions: ", max_revisions)
        
    if state["revision_number"] > max_revisions:
        return "end"
    return "contine"
```

Workflow을 위한 Graph를 준비합니다.

```python
workflow = StateGraph(State)

workflow.add_node("planner", plan)
workflow.add_node("generation", generation)
workflow.add_node("reflection", reflection)
workflow.add_node("research_plan", research_plan)
workflow.add_node("research_critique", research_critique)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "generation", 
    should_continue, 
    {
        "end": END, 
        "contine": "reflection"}
)

workflow.add_edge("planner", "research_plan")
workflow.add_edge("research_plan", "generation")

workflow.add_edge("reflection", "research_critique")
workflow.add_edge("research_critique", "generation")

app = workflow.compile()
```

이때 얻어진 Graph는 아래와 같습니다.

```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

![image](https://github.com/user-attachments/assets/95a053e8-b2e3-4efd-97d7-cf93872e1779)

아래와 같이 실행합니다.

```python
inputs = {"task": "즐겁게 사는 방법"}
config = {
    "recursion_limit": 50,
    "max_revisions": 2,
}
for output in app.stream(inputs, config=config):
    for key, value in output.items():
        print(f"Finished: {key}")

print("Final: ", value["essay"])
```

실행한 결과는 아래와 같습니다. 

![easy-writer](https://github.com/user-attachments/assets/6fda99c8-a902-49c8-a82a-994569429932)

LangSmith로 확인한 동작은 아래와 같습니다. 전체 138초가 소요되었습니다.

![image](https://github.com/user-attachments/assets/0b96a360-ce52-496b-9fa3-e7505c650fb3)




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
