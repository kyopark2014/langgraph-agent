# Reflection Agent

[Reflection Agents](https://www.youtube.com/watch?v=v5ymBTXNqtk)에서는 Reflection Agent에 대해 설명하고 있습니다. 이와 관련된 [Blog - Reflection Agents](https://blog.langchain.dev/reflection-agents/)을 참조합니다. 

Reflection은 Agent을 포함한 AI 시스템의 품질과 성공률을 높이기 위해 사용되는 프롬프트 전략(prompting strategy)입니다. 

LangGraph를 사용하여 3가지 반영 기술을 구축하는 방법을 설명하고 있으며, Reflexion과 Language Agent Tree Search의 구현 방법도 포함되어 있습니다. 

## Simple Reflection

[agent-reflection-kor.ipynb](./agent/agent-reflection-kor.ipynb)에서는 Reflection을 구현하는 방법에 대해 설명합니다. 이때의 개념도는 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/2a77a177-5be9-4a7d-97a8-4d5a19f9709e)

### 참고문헌

- [agent-reflection.ipynb](./agent/agent-reflection.ipynb) 에서는 MessageGraph()로 LangGraph Agent 만드는것을 설명합니다.

- [reflection.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/reflection/reflection.ipynb)에서는 LangGraph로 Reflection을 이용한 Agent를 설명하고 있습니다. 이것은 re-planning, search, evalution에 활용될 수 있습니다. 

### Node의 정의

에세이 형태의 Prompt를 구성합니다. 

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 5문단의 에세이 작성을 돕는 작가이고 이름은 서연입니다"
            "사용자의 요청에 대해 최고의 에세이를 작성하세요."
            "사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요."
            "완성된 에세이는 <result> tag를 붙여주세요.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | chat
```

Reflect를 위한 Prompt를 정의합니다.

```python
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
            "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
            "특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | chat
```

Workflow를 위한 Node 함수를 정의합니다.

```python
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, List, Union

class ChatAgentState(TypedDict):
    # messages: Annotated[Sequence[BaseMessage], operator.add]
    messages: Annotated[list, add_messages]

def generation_node(state: ChatAgentState):    
    response = chain.invoke(state["messages"])
    return {"messages": [response]}

def reflection_node(state: ChatAgentState):
    messages = state["messages"]
    
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = reflect.invoke({"messages": translated})    
    response = HumanMessage(content=res.content)    
    return {"messages": [response]}

def should_continue(state: ChatAgentState):
    messages = state["messages"]
    
    if len(messages) >= 6:   # End after 3 iterations        
        return "end"
    else:
        return "continue"
```

StateGraph를 이용해 workflow를 정의합니다.

```python
from langgraph.graph import START, END, StateGraph

workflow = StateGraph(ChatAgentState)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)
workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "reflect",
        "end": END,
    },
)

workflow.add_edge("reflect", "generate")
app_reflection = workflow.compile()
```

구현된 workflow는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/b2cccf4d-8a91-4955-9e32-330f77182cff)

이제 아래와 같이 실행합니다.

```python
query = "한국 인공지능 발전을 어떤 준비를 해야할지 설명하세요. 특히 한국의 현황과 향후 중국, 일본, 미국과 어떻게 경쟁해야할지 기술하세요."
inputs = [HumanMessage(content=query)]

for event in app_reflection.stream({"messages": inputs}, stream_mode="values"):   
    message = event["messages"][-1]
    if message.content and len(event["messages"])>1:
        print('generate: ', message.content)
```

아래는 생성된 에세이 초안입니다. 

```text
여기 한국 인공지능 발전을 위한 5문단 에세이 초안입니다:

1. 한국은 인공지능 기술 발전을 위해 정부 차원의 전략적 투자와 지원이 필요합니다. 정부는 인공지능 연구개발에 많은 예산을 할당하고 우수 인재 육성 프로그램을 마련해야 합니다. 또한 인공지능 기업에 대한 세제 혜택과 규제 완화 등 기업 친화적 정책을 펼쳐야 합니다.

2. 한국 기업들은 인공지능 기술 개발에 적극적으로 나서야 합니다. 대기업은 인공지능 연구소를 설립하고 스타트업 인수 등 투자를 아끼지 말아야 합니다. 중소기업도 인공지능 기술을 자사 제품과 서비스에 접목하는 노력이 필요합니다. 기업들이 정부와 긴밀히 협력하여 기술 상용화에 박차를 가해야 합니다.

3. 한국의 우수한 인력을 활용하는 것도 중요합니다. 한국은 세계 수준의 과학기술 인재를 많이 보유하고 있습니다. 정부와 기업은 이들 인재가 국내에서 연구할 수 있는 환경을 만들어야 합니다. 동시에 해외 우수 인재를 적극 유치하여 국내 인공지능 생태계를 더욱 풍부하게 해야 합니다.

4. 중국, 일본, 미국 등 주요국과의 협력도 필수적입니다. 인공지능 기술은 국가 간 경계를 넘나드는 분야입니다. 한국은 이들 국가와 인재 교류, 공동 연구개발, 기술 이전 등의 협력을 강화해야 합니다. 상호 윈윈할 수 있는 전략적 제휴를 맺어 경쟁력을 높여야 합니다.

5. 마지막으로 인공지능 기술 발전에 따른 사회적 영향에도 대비해야 합니다. 인공지능으로 인한 일자리 변화, 윤리적 문제 등에 선제적으로 대응하는 정책과 제도를 마련해야 합니다. 인공지능 기술이 국가 발전과 국민 행복에 기여할 수 있도록 사회 전반의 인프라를 구축해야 합니다.
```

최종적으로 완성된 에세이는 아래와 같습니다.

```python
인공지능 기술 발전에 따른 사회적 영향에도 대비해야 합니다. 인공지능으로 인한 일자리 변화, 윤리적 문제 등에 선제적으로 대응하는 정책과 제도를 마련해야 합니다. 인공지능 기술이 국가 발전과 국민 행복에 기여할 수 있도록 사회 전반의 인프라를 구축해야 합니다."

1. 한국은 인공지능 기술 발전을 위해 정부 차원의 전략적 투자와 지원이 필요합니다. 정부는 인공지능 연구개발에 많은 예산을 할당해야 합니다. 또한 우수 인재 육성 프로그램을 마련하여 인공지능 전문가를 체계적으로 양성해야 합니다. 인공지능 기업에 대해서는 세제 혜택과 규제 완화 등 기업 친화적 정책을 펼쳐 활발한 기술 혁신이 일어날 수 있도록 해야 합니다.

2. 한국 기업들 역시 인공지능 기술 개발에 적극적으로 나서야 합니다. 대기업은 인공지능 전문 연구소를 설립하고, 유망 인공지능 스타트업에 대한 인수 및 투자를 아끼지 말아야 합니다. 중소기업도 자사 제품과 서비스에 인공지능 기술을 접목하는 노력이 필요합니다. 기업들이 정부와 긴밀히 협력하여 인공지능 기술의 상용화를 앞당겨야 합니다.

3. 한국은 우수한 과학기술 인재를 많이 보유하고 있지만, 아직 인공지능 분야 전문 인력이 부족한 실정입니다. 정부와 기업은 이들 인재가 국내에서 연구할 수 있는 환경을 조성해야 합니다. 동시에 해외 우수 인공지능 인재를 적극 유치하여 국내 인공지능 생태계를 더욱 풍부하게 해야 합니다.

4. 중국, 일본, 미국 등 주요국과의 협력도 필수적입니다. 중국의 정부 주도 인공지능 육성 정책에 대응하여 민간 기업의 자율성을 보장하는 정책을 펼쳐야 합니다. 일본의 제조업 인공지능 기술 강점을 인정하고 상호 기술교류를 활성화해야 합니다. 미국의 선도적 인공지능 기업들과 전략적 제휴를 맺어 기술 격차를 줄여나가야 합니다. 인재 교류, 공동 연구개발, 기술 이전 등 다각적인 협력을 통해 상호 윈윈할 수 있어야 합니다.

5. 한국은 우수한 ICT 인프라와 기술력을 바탕으로 인공지능 발전의 잠재력이 크지만, 아직 주요 기술과 인재가 부족한 실정입니다. 또한 인공지능 기업에 대한 투자와 정부 지원이 미흡하여 글로벌 경쟁력이 뒤처지고 있습니다.

6. 인공지능 기술 발전에 따른 사회적 영향에도 대비해야 합니다. 인공지능으로 인한 일자리 변화, 윤리적 문제 등에 선제적으로 대응하는 정책과 제도를 마련해야 합니다. 인공지능 기술이 국가 발전과 국민 행복에 기여할 수 있도록 사회 전반의 인프라를 구축해야 합니다.
```

LangSmith로 확인해 보면 아래와 같이 여러 단계를 거쳐서 generation/reflection이 수행되고 있음을 알 수 있습니다.

![image](https://github.com/user-attachments/assets/d40b049f-3fc3-4e26-909c-d04236b36c27)





