# LangGraph로 구현하는 Agent

## LangGraph Agent 

[langgraph-agent.md](./langgraph-agent.md)에서는 LangGraph Agent의 기본 구성을 설명하고 있습니다.

LLM을 사용할 때 다양한 API로부터 얻은 결과를 사용하여 더 정확한 결과를 얻고 싶을 때에 Agent을 사용합니다. 어떤 상황에 어떤 API를 쓸지를 판단하기 위해서는 상황 인식(Context-Aware)에 기반한 Reasoning(추론: 상황에 대한 인식을 바탕으로 새로운 사실을 유도)이 필요합니다. 여기에서는 Agent를 이용하여 여러개의 API를 선택적으로 사용하는 한국어 Chatbot을 구현합니다. 이를 위한 Architecture는 아래와 같습니다. 

1) 사용자가 채팅창에서 질문을 입력하면 WebSocket 방식으로 Lambda(chat)에 전달됩니다.
2) Lambda(chat)은 Agent 동작을 수행하는데, Action - Observation - Thought - Final Answer의 동작을 수행합니다. 만약 Thought에서 Final Answer를 얻지 못하면 Action부터 다시 수행합니다.
3) Agent의 Action은 API를 이용해 필요한 정보를 얻어옵니다. 이때 사용하는 API에는 도서 추천, 날씨정보, 검색엔진이 있을 수 있습니다. 또한 시스템 시간을 가져오는 동작은 별도 API가 아닌 내부 함수를 이용해 구현할 수 있습니다.
4) 만약 RAG의 정보가 필요한 경우에는 Action의 하나로 RAG을 이용하여 필요한 정보를 조회합니다.
5) Observation/Thought/Final Answer를 위해 Agent는 prompt를 이용해 LLM에 요청을 보내고 응답을 받습니다.
6) Agent가 Final Answer을 구하면 사용자에게 전달합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c372c125-4e05-41f8-b691-784e4c2028af)

아래에서 구현한 Agent는 zero-shot agent로 사용자의 질문에 따라 tools 리스트로부터 적절한 tool을 선택하여 활용합니다. tool은 함수 또는 API로 구현됩니다. 선택된 tool로 원하는 작업을 완료하지 못하면 다른 tool을 추가로 활용합니다.


## Agent Use Cases

### Reflection

Reflection을 통해 LLM의 응답을 향상시키고 충분한 컨텐츠를 제공할 수 있습니다. [reflection-agent.md](./reflection-agent.md)에서는 LangGraph를 이용해 Reflection을 반영하는 Agent를 생성하는 방법을 설명하고 있습니다. 이에 대한 activity diagram은 아래와 같습니다. 

1) "START"가 "generat"에 사용자의 input을 전달하면, "generate"는 LLM으로 답변(generation)을 생성하여 "should_continue"에 전달합니다.
2) "should_continue"는 일정 숫자만큼만 reflection을 수행하도록 제어합니다. 여기서는 메시지의 길이가 6이하인 경우에는 "reflect"에 보내고, 이상인 경우에는 "END"로 보내서 동작을 종료하도록 하고 있습니다. 메시지는 Human/AI 형태로 저장되므로 6이하라면 모두 3회 반복하게 됩니다. 
3) "reflect"는 "generate"가 생성한 답변(generation)을 받아서, LLM Prompt를 이용해 향상된 답변(reflection)을 생성합니다.
4) 이 동작을 3회 반복 수행한 후에 "END"에 최종 답변을 전달합니다. 

<img src="https://github.com/user-attachments/assets/8ca02018-56b8-4272-a8d1-f71aa6b6f0ca" width="350">
   

### Corrective RAG

[corrective-rag-agent.md](./corrective-rag-agent.md)에서는 Corrective RAG을 이용한 RAG 성능 강화에 대해 설명합니다. Corrective RAG는 Vector Store에서 가져온 문서를 Refine하고 관련성이 적은 문서는 제외하고, 다른 데이터 소스나 Web 검색을 통해 RAG의 성능을 향상시킬 수 있습니다. 아래 그림은 Corrective RAG에 대한 activity diagram입니다. 

1) "retrieve"는 질문(Question)을 이용하여, RAG의 Vector Store로 조회(retrieve) 동작을 수행합니다. 이때 k개의 관련된 문서(relevant docuements)을 가져옵니다.
2) "grade_documents"는 LLM prompt를 이용하여 Vector Store에서 가져온 문서가 실제로 관련이 있는지 확인합니다. 관련이 있으면 "yes", 없으면 "no"를 판별(grade)하는데, "no"인 경우에 관련된 문서에서 제외합니다. 만약 관련된 문서가 관련성이 없어 제외되면, "web_search"를 True로 설정합니다. 
3) "decide_to_generate"는 Vector Store에서 가져온 모든 문서가 관련이 있다면, "web_search"를 "yes"로 설정하고, 아니라면 "no로 설정합니다. 이와같이 관련된 문서중에 일부라도 관련이 적다고 판정되면, 웹 검색을 수행하여 관련된 문서를 보강합니다.
4) "web_search"가 "yes"라면 (웹 검색이 필요한 경우), 기존 질문으로 부터 향상된 질문(better_question)을 생성하는 re-write를 동작을 수행합니다. 이를 위해 "rewrite"는 LLM Prompt를 이용하여, 충분히 의도(sementic intent)와 의미(meaning)을 가지도록 향상된 질문(better_question)을 생성합니다.
6) "web search"는 기존 문서(filtered_document)에 웹 검색으로 얻어진 새로운 관련된 문서를 추가해서 문서(documents)를 생성합니다. 
7) "generate"에서는 관련된 문서(documents)를 context로 활용하여 적절한 답변을 생성합니다. 

<img src="https://github.com/user-attachments/assets/996d6671-1782-4968-be4f-0ade60b0316d" width="300">


### Self RAG

Self RAG는 RAG의 Vector Store에서 얻어진 문서들의 관련성을 확인(Grade)하여 관련성이 적은 문서를 제외합니다. 또한 얻어진 답변이 환각(Hallucination)인지, 충분한 잘 작성된 답변인지 확인하여, 답변이 충분하지 않으면 질문을 re-write하여 RAG 동작을 재수행합니다. 이를 통해 RAG의 결과를 향상 시킬수 있습니다. 상세한 내용은 [Self RAG](https://github.com/kyopark2014/llm-agent/blob/main/self-rag.md)에서 설명합니다. 아래는 Self RAG에 대한 activity diagram입니다. 

1) "retrive"는 질문(question)을 이용하여 Vector Store에 관련된 문서를 조회(retrieve)합니다.
2) "grade_documents"는 LLM Prompt를 이용하여 문서(documents)의 관련성을 확인(grade)합니다. 관련이 없는 문서는 제외하여 "filtered documents"로 제조합합니다. 
3) "decide_to_generate"는 "filtered document"를 "generate"로 보내서 답변을 생성하도록 합니다. "filtered document"가 없다면 새로운 질문을 생성하기 위해 "rewrite" 동작을 수행하도록 요청합니다.
4) "rewrite"는 기존 질문(question)을 이용하여 LLM Prompt로 새로운 질문을 생성합니다. 새로운 질문(better question)은 "retrieve"에 전달되어, 새로운 질문으로 RAG 동작을 재수행할 수 있습니다.  
5) "generate"는 "filtered documents"를 이용하여 적절한 답변(generation)을 생성합니다.
6) "grade_generation"은 생성된 답변이 환각(hallucination)인지 확인하여, 만약 환각이라면 "generator"에 보내 다시 답변을 생성하고, 환각이 아니라면 답변이 적절한지 "answer_question"로 검증합니다. 이때, 답변이 적절하다면(useful) 최종 결과를 전달하고, 적절하지 않다면(not_useful) 질문을 새로 생성하기 위해 "rewrite"합니다. 이후로 새로 생성된 질문은 "retrieve"에 전달되어 RAG 조회 과정을 반복합니다.
   
![image](https://github.com/user-attachments/assets/55672f1a-0b8e-4566-a604-6e5534d9e7d9)

### Self-Corrective RAG

Self-Corrective RAG는 Corrective RAG처럼 Vector Store로 부터 얻어진 문서의 관련성을 확인하여 관련성이 없는 문서를 제외하고 웹 검색을 통해 결과를 보강합니다. 또한, Self RAG처럼 RAG의 결과가 환각(Hallucination)인지, 적절한 답변인지 검증하는 절차를 가지고 있습니다. 상세한 내용은 [self-corrective-rag.md](./self-corrective-rag.md)에서 설명합니다. 아래는 Self-Corrective RAG에 대한 acitivity diagram입니다. 

1) "retrieve"는 질문(question)과 관련된 문서를 Vector Store를 통해 조회합니다. 이때, "grade_generation" 동작을 위해 "web_fallback"을 True로 초기화합니다.
2) "generator"는 Vector Store에서 얻어진 관련된 문서(documents)를 이용하여 답변(generation)을 생성합니다. 이때, retries count를 증가시킵니다.
3) "grade_generation"은 "web_fallback"이 True이라면, "hallucination"과 "answer_question"에서 환각 및 답변의 적절성을 확인합니다. 환각일 경우에, 반복 횟수(retries)가 "max_retries"에 도달하지 않았다면 "generate"보내서 답변을 다시 생성하고, "max_retires"에 도달했다면 "websearch"로 보내서 웹 검색을 수행합니다. 또한 답변이 적절하지 않다면, 반복 횟수가 "max_reties"에 도달하기 않았다면, "rewrite"로 보내서 향상된 질문(better question)을 생성하고, 도달하였다면 "websearch"로 보내서 웹 검색을 수행합니다.
4) "websearch"는 웹 검색을 통해 문서를 보강하고, "generate"에 보내서 답변을 생성합니다. 이때, "web_fallback"을 False로 설정하여 "grade_generation"에서 "finalized_response"로 보내도록 합니다.
5) "rewrite"는 새로운 질문(better question)을 생성하여, "retrieve"에 전달합니다. 새로운 질문으로 전체 RAG 동작을 재수행합니다. 전체 RAG 동작은 무한 루프를 방지하기 위하여, "max_retries"만큼 수행할 수 있습니다.
6) "finalize_response"는 최종 답변을 전달합니다.

![image](https://github.com/user-attachments/assets/5769e8ed-6e76-4fda-a932-a1d3c461de50)

## Reference

- [planning-agents.md](./planning-agents.md)에서는 plan-and-execution 형태의 agent를 생성하는 방법을 설명합니다.

- [reflexion-agent.md](./reflexion-agent.md)에서는 Reflexion방식의 Agent에 대해 설명합니다.

- [langgraph-agent.md](./langgraph-agent.md)에서는 LangGraph를 이용해 Agent를 생성하는 방법을 설명합니다. 

- [reflection-agent.md](./reflection-agent.md)에서는 reflection을 이용해 성능을 향상시키는 방법에 대해 설명합니다.

- [persistence-agent.md](./persistence-agent.md)에서는 checkpoint를 이용해 이전 state로 돌아가는 것을 보여줍니다.

- [olympiad-agent.md](./olympiad-agent.md)에서는 Reflection, Retrieval, Human-in-the-loop를 이용해 Olympiad 문제를 푸는것을 설명합니다.

- [code-agent.md](./code-agent.md)에서는 LangGraph를 이용해 code를 생성하는 예제입니다.

- [email-agent.md](./email-agent.md)에서는 LangGraph를 이용해 email을 생성하는 예제입니다.

- [support-bot-agent.md](./support-bot-agent.md)에서는 고객 지원하는 Bot을 Agent로 생성합니다.

- [language-agent-tree-search.md](./language-agent-tree-search.md)에서는 Tree Search 방식의 Agent를 만드는것을 설명합니다.

- [rewoo.md](./rewoo.md)에서는 Reasoning without Observation 방식의 Agent에 대해 설명합니다.

- [llm-compiler.md](./llm-compiler.md)에서는 "An LLM Compiler for Parallel Function Calling"을 구현하는것에 대해 설명합니다. 

- [multi-agent.md](./multi-agent.md)에서는 여러개의 Agent를 이용하는 방법에 대해 설명합니다. 

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

LangGraph를 이용해 한국어 Chatbot Agent을 만들었습니다. Agent를 사용함으로써 다양한 API를 문맥(Context)에 따라 활용할 수 있었습니다. 다만 API를 여러번 호출함으로 인한 지연시간이 증가하고, prompt에 넣을 수 있는 Context 길이 제한으로 검색이나 RAG 결과를 일부만 넣게 되는 제한이 있습니다.

## Reference

[Building Context-Aware Reasoning Applications with LangChain and LangSmith](https://www.youtube.com/watch?app=desktop&v=Hy08dbsfJGg)

[Using LangChain ReAct Agents for Answering Multi-hop Questions in RAG Systems](https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e)

[Intro to LLM Agents with Langchain: When RAG is Not Enough](https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834)

[LangChain 🦜️🔗 Tool Calling and Tool Calling Agent 🤖 with Anthropic](https://medium.com/@dminhk/langchain-%EF%B8%8F-tool-calling-and-tool-calling-agent-with-anthropic-467b0fb58980)

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

[llama3 로 #agent 🤖 만드는 방법 + 8B 오픈 모델로 Agent 구성하는 방법](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)

[LLM-powered autonomous agent system](https://lilianweng.github.io/posts/2023-06-23-agent/)

