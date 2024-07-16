# LangGraph로 구현하는 Agent

LLM을 사용할 때 다양한 API로부터 얻은 결과를 사용하여 더 정확한 결과를 얻고 싶을 때에 Agent을 사용합니다. 어떤 상황에 어떤 API를 쓸지를 판단하기 위해서는 상황 인식(Context-Aware)에 기반한 Reasoning(추론: 상황에 대한 인식을 바탕으로 새로운 사실을 유도)이 필요합니다. 여기에서는 Agent를 이용하여 여러개의 API를 선택적으로 사용하는 한국어 Chatbot을 구현합니다. 이를 위한 Architecture는 아래와 같습니다. 

1) 사용자가 채팅창에서 질문을 입력하면 WebSocket 방식으로 Lambda(chat)에 전달됩니다.
2) Lambda(chat)은 Agent 동작을 수행하는데, Action - Observation - Thought - Final Answer의 동작을 수행합니다. 만약 Thought에서 Final Answer를 얻지 못하면 Action부터 다시 수행합니다.
3) Agent의 Action은 API를 이용해 필요한 정보를 얻어옵니다. 이때 사용하는 API에는 도서 추천, 날씨정보, 검색엔진이 있을 수 있습니다. 또한 시스템 시간을 가져오는 동작은 별도 API가 아닌 내부 함수를 이용해 구현할 수 있습니다.
4) 만약 RAG의 정보가 필요한 경우에는 Action의 하나로 RAG을 이용하여 필요한 정보를 조회합니다.
5) Observation/Thought/Final Answer를 위해 Agent는 prompt를 이용해 LLM에 요청을 보내고 응답을 받습니다.
6) Agent가 Final Answer을 구하면 사용자에게 전달합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c372c125-4e05-41f8-b691-784e4c2028af)

아래에서 구현한 Agent는 zero-shot agent로 사용자의 질문에 따라 tools 리스트로부터 적절한 tool을 선택하여 활용합니다. tool은 함수 또는 API로 구현됩니다. 선택된 tool로 원하는 작업을 완료하지 못하면 다른 tool을 추가로 활용합니다.

## LangGraph Agent

### Agent의 정의

[Agent란](https://terms.tta.or.kr/dictionary/dictionaryView.do?word_seq=171384-1%29) 주변 환경을 탐지하여 자율적으로 동작하는 장치 또는 프로그램을 의미합니다. 인공지능을 이용한 지능형 에이전트는 센서를 이용하여 주변 환경을 자각하여 Actuator를 이용하여 적절한 행동을 합니다. agent의 라틴어 어원인 [agere의 뜻](https://m.blog.naver.com/skyopenus/221783830658)은 to do 또는 to act의 의미를 가지고 있습니다. Agent를 이용하면 LLM 결과를 향상시킬 수 있습니다. 

LangGraph는 agent를 생성하고 여러개의 Agent가 있을때의 흐름을 관리하기 위한 LangChain의 Extention입니다. 이를 통해 cycle flow를 생성할 수 있으며, 메모리가 내장되어 Agent를 생성에 도움을 줍니다. 상세한 내용은 [LangGraph guide](https://langchain-ai.github.io/langgraph/how-tos/)을 참조합니다.

### LangChain Agent와 비교

- LangChain Agent는 Resoning/Action을 효과적으로 수행하고 매우 powerful 합니다.
- LLM의 성능이 매우 중요하므로 LLM 모델을 잘 선택하여야 합니다. 성능이 더 좋은 모델은 일반적으로 더 많은 연산시간을 필요로 합니다. (예 지연시간: Opus > Sonnet > Haiku)
- 각 Tool의 invocation을 위해서 매번 LLM을 호출하여야 합니다. Tool을 연속적으로 실행(Observation 없이)할 때에는 불필요한 시간이 될 수 있습니다. 
- 한번에 한개의 step을 수행하고 parallel call을 지원하지 않습니다.
- LangGraph를 이용한 Agent는 복잡한 process를 State Machine을 이용해 구현할 수 있으며, Multi-Agent 구조에 적합합니다.

### Components

- Memory: Shared state across the graph
- Tools: Nodes can call tools and modify state
- Planning: Edges can route control flow based on LLM decisions

참조: [Building and Testing Reliable Agents](https://www.youtube.com/watch?v=XiySC-d346E): chain/agent 비교하여 개념 설명 매우 좋음



### LangGraph Agent의 구현

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

### Agent Case Studies

1) Reflection: [reflection-agent.md](./reflection-agent.md)에서는 LangGraph를 이용해 Reflection을 반영하는 Agent를 생성하는 방법을 설명하고 있습니다.

2) Plan and Execution: [planning-agents.md](./planning-agents.md)에서는 plan-and-execution 형태의 agent를 생성하는 방법을 설명합니다.

3) Reflexion: [reflexion-agent.md](./reflexion-agent.md)에서는 Reflexion방식의 Agent에 대해 설명합니다.

4) Corrective RAG: [corrective-rag-agent.md](./corrective-rag-agent.md)에서는 Self reflection을 이용한 RAG 성능 강화에 대해 설명합니다.

5) Self-Corrective RAG: [self-corrective-rag.md](./self-corrective-rag.md)에서는 Self Corrective RAGf를 Agent로 구현하는것을 설명합니다.

6) Self RAG: [Self RAG](https://github.com/kyopark2014/llm-agent/blob/main/self-rag.md)에서는 RAG의 결과를 Grade하고 Hallucination을 방지하기 위한 task를 활용해 RAG의 성능을 높입니다.


## Use Cases

- [langgraph-agent.md](./langgraph-agent.md)에서는 LangGraph를 이용해 Agent를 생성하는 방법을 설명합니다. 

- [reflection-agent.md](./reflection-agent.md)에서는 reflection을 이용해 성능을 향상시키는 방법에 대해 설명합니다.

- [persistence-agent.md](./persistence-agent.md)에서는 checkpoint를 이용해 이전 state로 돌아가는 것을 보여줍니다.

- [olympiad-agent.md](./olympiad-agent.md)에서는 Reflection, Retrieval, Human-in-the-loop를 이용해 Olympiad 문제를 푸는것을 설명합니다.

- [code-agent.md](./code-agent.md)에서는 LangGraph를 이용해 code를 생성하는 예제입니다.

- [email-agent.md](./email-agent.md)에서는 LangGraph를 이용해 email을 생성하는 예제입니다.

- [support-bot-agent.md](./support-bot-agent.md)에서는 고객 지원하는 Bot을 Agent로 생성합니다.

- Language Agent Tree Search: [language-agent-tree-search.md](./language-agent-tree-search.md)에서는 Tree Search 방식의 Agent를 만드는것을 설명합니다.

- Reasoning without Observation: [rewoo.md](./rewoo.md)에서는 Reasoning without Observation 방식의 Agent에 대해 설명합니다.

- LLMCompiler: [llm-compiler.md](./llm-compiler.md)에서는 "An LLM Compiler for Parallel Function Calling"을 구현하는것에 대해 설명합니다. 

- Multi agent: [multi-agent.md](./multi-agent.md)에서는 여러개의 Agent를 이용하는 방법에 대해 설명합니다. 

- [stome-agent.md](./stome-agent.md)에서는 풍부한 기사를 생성(richer article generation) Storm Agent에 대해 설명합니다.

- [GPT Newspape](https://www.youtube.com/watch?v=E7nFHaSs3q8)에서는 신문요약에 대해 설명하고 있습니다. ([github](https://github.com/rotemweiss57/gpt-newspaper/tree/master) 링크)

- [Essay Writer](https://github.com/kyopark2014/llm-agent/blob/main/essay-writer.md)에서는 essay를 작성하는 Agent를 생성합니다.
  



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 Seoul 리전 (ap-northeast-2)을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과

실행한 결과는 아래와 같습니다.

- "안녕"이라고 입력하고, 동작하는것을 LangSmith로 확인합니다. 
  
![image](https://github.com/kyopark2014/llm-agent/assets/52392004/9e737a68-1e7b-4062-9dde-f94b7b03a2b4)

Tools에 여러개의 API를 등록해 놓았지만, LLM이 Tool을 사용할 필요가 없다고 생각하면 LLM이 답변을 수행합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/da33d115-62fc-454d-ac26-71d13358bc90)

이때의 로그는 아래와 같습니다.

```text
Thought: Tool을 사용해야 하나요? No
Final Answer: 안녕하세요! 무엇을 도와드릴까요?
```  

- "서울 날씨는?"를 입력하면 현재의 [날씨 정보를 조회](./apis.md#%EB%82%A0%EC%94%A8-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)하여 알려줍니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4b2f79cc-6782-4c44-b594-1c5f22472dc7)

- "오늘 날짜 알려줘"를 하면 [시스템 날짜를 확인](./apis.md#%EB%82%A0%EC%A7%9C%EC%99%80-%EC%8B%9C%EA%B0%84-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)하여 알려줍니다. 

<img width="850" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/a0190426-33d4-46d3-b9d2-5294f9222b8c">

- "서울 여행에 대한 책을 추천해줘"를 입력하면 [교보문고의 검색 API](./apis.md#%EB%8F%84%EC%84%9C-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)를 이용하여 관련책을 검색하여 추천합니다.

<img width="849" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/e62b4654-ba18-40e6-86ae-2152b241aa04">

- 오늘 날짜를 알수 있으므로 "올해 크리스마스까지 몇일이 남아 있어?"와 같은 질문에 정확히 답변할 수 있습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/8905c677-7a26-4a4e-9e14-ee8af8a481cf)


- "서울과 부산의 날씨를 알려줘"와 같이 서울과 부산의 결과를 각각 검색한 후에 아래와 같은 결과를 얻습니다. 

<img width="848" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/7b5c4993-1178-442d-9fb0-ddaff6b7ab09">

이때의 LangSmith의 로그를 확인하면 서울과 부산과 대한 검색후 결과를 생성하였습니다. (get_weather_info를 서울과 부산에 대해 각각 호출함)

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/38334666-c71d-4076-9be1-eb8fc16a34f5)


- "미국 여행을 하려고 해. 추천해줘 어떻게 여행하는게 좋아?"로 질문을 하면 아래와 같이 로스웬젤레스를 추천해주는데 날씨정보도 같이 전달하고 있습니다.

상세한 내부 동작은 아래와 같습니다. 

1) 질문에 필요한 정보를 찾습니다. 여기에서는 여행일정, 방문도시, 관심사에 선택했습니다.
2) 현재 가지고 있는 api중에 관련된 것을 찾았는데, 도서정보를 찾는 API(get_product_list)가 선택되었습니다.
3) "미국 여행 가이드 북"을 검색해서 도서 정보를 얻었습니다.
4) 가이드된 도서 목록에서 미국 서부 지역, 뉴욕을 선택하고 추가 정보를 찾으려고 합니다.
5) API중에 get_weather_info을 선택해서, Los Angeles 를 검색합니다.
6) 가이드 북과 날씨 정보를 조합하여 Final Answer로 여행 계획을 추천합니다. 

아래는 LangSmith에서 추출한 로그 입니다. 

```text
Question: 미국 여행을 하려고 해. 추천해줘 어떻게 여행하는게 좋아?
Thought:Thought: 미국 여행에 대한 구체적인 정보가 필요할 것 같습니다. 여행 일정, 방문 도시, 관심사 등에 대해 더 자세히 알아보는 것이 좋겠습니다.
Action: get_product_list
Action Input: 미국 여행 가이드북
Observation: 추천 도서는 아래와 같습니다.
[국내도서]미국 서부 셀프트래블(2022-2023), URL: https://product.kyobobook.co.kr/detail/S000060626443
[eBook]미국 서부 셀프트래블(2022-2023), URL: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000002960164
[국내도서]버킷리스트 미국 서부, URL: https://product.kyobobook.co.kr/detail/S000001981279
[eBook]살아보고 싶다면, 포틀랜드, URL: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000003234513
[국내도서]뉴욕 영화 가이드북, URL: https://product.kyobobook.co.kr/detail/S000001913689

Thought: Thought: 여행 가이드북 목록을 보니 미국 서부 지역과 뉴욕에 대한 정보가 있습니다. 하지만 구체적인 여행 계획을 세우기 위해서는 더 자세한 정보가 필요할 것 같습니다.
Action: get_weather_info
Action Input: Los Angeles
Observation: 로스앤젤레스의 현재 날씨의 특징은 Clouds이며, 현재 온도는 13.85도 이고, 최저온도는 11.85도, 최고 온도는 15.6도 입니다. 현재 습도는 85% 이고, 바람은 초당 2.06 미터 입니다. 구름은 100% 입니다.
Thought: 미국 여행에 대한 구체적인 정보를 모으기 위해 여러 도구를 사용해 보았습니다. 가이드북 목록을 통해 인기 여행지와 관심 분야를 파악할 수 있었고, 날씨 정보를 통해 여행 시기와 준비물을 고려할 수 있었습니다. 이제 이 정보들을 종합하여 여행 계획을 세울 수 있을 것 같습니다.

Final Answer: 미국 여행을 계획할 때는 다음 사항들을 고려하는 것이 좋습니다.

1. 여행 지역 선정: 가이드북 목록에서 보았듯이 서부 지역(로스앤젤레스, 포틀랜드 등)과 뉴욕이 인기 여행지입니다. 관심사와 일정에 맞춰 방문 도시를 정하세요.

2. 계절 및 날씨: 로스앤젤레스 날씨 정보를 보니 현재 구름이 많고 온도는 10도 중반입니다. 여행 시기에 따라 날씨가 다르므로 이에 맞춰 옷차림과 활동을 계획하는 것이 좋습니다. 

3. 교통 및 숙박: 렌터카 이용이 편리할 수 있습니다. 도시 간 이동 시에는 기차나 항공편을 이용할 수 있습니다. 숙박 시설은 호텔, 모텔, 에어비앤비 등 다양한 선택지가 있습니다.

4. 관광명소 및 활동: 뉴욕 영화 가이드북에서 보듯 영화 촬영지 방문, 미술관 관람 등 취향에 맞는 활동을 계획하세요.
```

이때의 결과는 아래와 같습니다. 

<img width="858" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/75987226-b3d2-481c-a9ba-efc62bfbe4ca">

- "서울 여행에 대한 책을 추천해줘"로 입력후 결과를 확인합니다.

<img width="848" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/0213de6b-2580-4598-a2fc-b671aea43a37">

아래와 같이 get_book_list를 이용해 얻어온 도서 정보와 search_by_tavily로 얻어진 정보를 통합하였음을 알 수 있습니다.

![image](https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/6b33eb2d-11bc-4959-81d0-9ba76ca55ab2)

- 다양한 API사용해 보기 위하여 "서울에서 부산으로 여행하려고 하고 있어. 서울과 부산의 온도를 비교해줘. 그리고 부산가면서 읽을 책 추천해주고, 부산가서 먹을 맛집도 찾아줘."로 입력 후 결과를 확인합니다. 

  ![image](https://github.com/kyopark2014/llm-agent/assets/52392004/05eb0ab0-fa84-487e-b008-d8517d53105c)

LangSmith의 로그를 보면 아래와 같이 get_weather_info로 서울/부산의 날씨를 검색하고, get_book_list을 이용해 도서 목록을 가져오고, search_by_tavily로 맛집 검색한 결과를 보여주고 있습니다. 

<img width="293" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/dc0db14a-dcd2-486b-b0f5-3fae8a7b60bb">

- [error_code.pdf](./contents/error_code.pdf)를 다운로드 한 후에 채팅창의 파일 아이콘을 선택하여 업로드 합니다. 이후 "보일러 에러코드에 대해 설명해줘."라고 입력하몬 RAG에서 얻어진 결과를 이용해 아래와 같이 답변합니다. 

<img width="852" alt="image" src="https://github.com/kyopark2014/multimodal-on-aws/assets/52392004/16ee0cdc-73d2-4e03-9d23-129b209af4ea">

LangSmith의 로그를 보면 아래와 같이 search_by_opensearch(RAG)를 호출하여 얻은 정보로 답변을 생성했음을 알 수 있습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/6f9db7f5-4ab1-44b5-aa8f-5c158ee12381)



## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "rest-api-for-llm-agent", "ws-api-for-llm-agent"을 삭제합니다.

2) [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cd ~/environment/llm-agent/cdk-llm-agent/ && cdk destroy --all
```

## 결론

LangChain과 LangGraph를 이용해 한국어 Chatbot Agent을 만들었습니다. Agent를 사용함으로써 다양한 API를 문맥(Context)에 따라 활용할 수 있었습니다. 다만 API를 여러번 호출함으로 인한 지연시간이 증가하고, prompt에 넣을 수 있는 Context 길이 제한으로 검색이나 RAG 결과를 일부만 넣게 되는 제한이 있습니다.

## Reference

[Building Context-Aware Reasoning Applications with LangChain and LangSmith](https://www.youtube.com/watch?app=desktop&v=Hy08dbsfJGg)

[Using LangChain ReAct Agents for Answering Multi-hop Questions in RAG Systems](https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e)

[Intro to LLM Agents with Langchain: When RAG is Not Enough](https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834)

[LangChain 🦜️🔗 Tool Calling and Tool Calling Agent 🤖 with Anthropic](https://medium.com/@dminhk/langchain-%EF%B8%8F-tool-calling-and-tool-calling-agent-with-anthropic-467b0fb58980)

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

[llama3 로 #agent 🤖 만드는 방법 + 8B 오픈 모델로 Agent 구성하는 방법](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)

[LLM-powered autonomous agent system](https://lilianweng.github.io/posts/2023-06-23-agent/)

