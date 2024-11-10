# LangGraph로 구현하는 Conversational Chatbot

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Flanggraph-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


## LangGraph Agent 

[Agent란](https://terms.tta.or.kr/dictionary/dictionaryView.do?word_seq=171384-1%29) 주변 환경을 탐지하여 자율적으로 동작하는 장치 또는 프로그램을 의미합니다. agent의 라틴어 어원인 [agere의 뜻](https://m.blog.naver.com/skyopenus/221783830658)은 to do 또는 to act의 의미를 가지고 있습니다. 

LangGraph는 agent를 생성하고 여러개의 agent가 있을때의 흐름을 관리하기 위한 LangChain의 Extention입니다. 이를 통해 cycle flow를 생성할 수 있으며, 메모리가 내장되어 Agent를 생성에 도움을 줍니다. 상세한 내용은 [LangGraph guide](https://langchain-ai.github.io/langgraph/how-tos/)을 참조합니다. 

아래와 같이 agent는 serverless architecture로 구현할 수 있습니다.

1) 사용자기 질문을 입력하면 webSocket 방식으로 API Gateway로 질문이 전달됩니다.
2) Lambda에서는 LangGraph 형태로 agent를 정의합니다. 여기에서는 Corrective RAG, Self RAG, Self corrective RAG를 구현하였습니다.
3) 질문에 관련된 문서를 OpenSearch를 통해 검색합니다. 이때, child/parent chunking을 이용해 작은 크기의 chunk를 검색하여 성능을 향상시키면서도 Parent Chunk를 문서로 활용하여 풍부한 context를 제공합니다.
4) OpenSearch로 얻어진 관련된 문서는 LLM Prompt를 이용해 관련도를 평가(grade) 합니다. 평가 결과에 따라 웹검색을 이용한 fallback 동작을 수행할 수 있습니다.
5) LLM에 관련된 문서를 context로 제공하여 적절한 답변을 생성합니다. 답변이 환각(hallucination)인지, 적절한 답변이 생성되었는지를 LLM prompt를 이용해 평가(grade)할 수 있습니다. 

<img width="640" alt="image" src="https://github.com/user-attachments/assets/4c18b02c-588d-4a3c-b024-39ac58d20dec">


[langgraph-agent.md](./langgraph-agent.md)에서는 LangGraph Agent의 기본 구성을 설명하고 있습니다.



## Agentic Workflow Design Patterns 

여기서는 대표적인 agentic workflow의 4가지 패턴인 tool use, reflection, plan and execute, multi-agent collaboration에 대해 설명합니다. 

### Tool Use

[agent-executor.md](./agent-executor.md)에서는 LangGraph를 이용해 각종 Tool을 실행하는 agent를 만드는 방법을 설명하고 있습니다. 아래는 agent executor의 activity diagram입니다.

<img src="https://github.com/user-attachments/assets/836d7458-1dab-40db-8cde-fae0becbcce6" width="350">
  
### Reflection

Reflection을 통해 LLM의 응답을 향상시키고 충분한 컨텐츠를 제공할 수 있습니다. [knowledge-guru.md](./knowledge-guru.md)에서는 OpenSearch와 Tavily 검색으로 얻어진 정보로 초안(draft)를 만들고, 검색을 위한 keyword를 추출하여 다시 검색하는 방법으로 답변을 향상사키는(revise) 방법에 대해 설명합니다. 

![image](https://github.com/user-attachments/assets/59f022a3-5946-4cb1-93bf-46b159635f9f)


### Plan and Execute

[plan-and-execuite.md](./plan-and-execuite.md)에서는 Plan and Execute 방식의 Agent를 생성하는 방법에 대해 설명합니다. 

![image](https://github.com/user-attachments/assets/8e6340d8-a49f-4af7-ae4e-167909ff6f7c)
  
### Multi-agent Collaboration 

[long-form-writing.md](./long-form-writing.md)에서는 plan and execute 패턴을 가지는 agent와 reflection을 수행하는 agent를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. Multi agent 구조로 구성함으로써 복잡한 workflow를 단순하게 구현할 수 있습니다.

<img src="https://github.com/user-attachments/assets/6007e82d-f617-48a9-b23a-38a156f4abd1" width="800">


## Case Studies

### Reflection

[reflection-agent.md](./reflection-agent.md)에서는 LangGraph를 이용해 Reflection을 반영하는 Agent를 생성하는 방법을 설명하고 있습니다. 

<img src="https://github.com/user-attachments/assets/8ca02018-56b8-4272-a8d1-f71aa6b6f0ca" width="400">

### Agentic RAG

Agentic RAG는 tool_condition을 통해 RAG에 retrival을 선택하고, 문서를 평가(grade)하여, 검색 결과가 만족스럽지 않다면 re-write를 통해 새로운 질문(better question)을 생성할 수 있습니다. 상세한 내용은 [agentic-rag.md](./agentic-rag.md)을 참조합니다. 

![image](https://github.com/user-attachments/assets/093c3ce9-195e-4b65-a000-fd21089884d7)

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

Self RAG는 RAG의 vector store에서 얻어진 문서들의 관련성을 확인(Grade)하여 관련성이 적은 문서를 제외합니다. 또한 얻어진 답변이 환각(hallucination)인지, 충분한 잘 작성된 답변인지 확인하여, 답변이 충분하지 않으면 질문을 re-write하여 RAG 동작을 재수행합니다. 이를 통해 RAG의 결과를 향상 시킬수 있습니다. 상세한 내용은 [Self RAG](./self-rag.md)에서 설명합니다. 아래는 Self RAG에 대한 activity diagram입니다. 

1) "retrive"는 질문(question)을 이용하여 Vector Store에 관련된 문서를 조회(retrieve)합니다.
2) "grade_documents"는 LLM Prompt를 이용하여 문서(documents)의 관련성을 확인(grade)합니다. 관련이 없는 문서는 제외하여 "filtered documents"로 제조합합니다. 
3) "decide_to_generate"는 "filtered document"를 "generate"로 보내서 답변을 생성하도록 합니다. "filtered document"가 없다면 새로운 질문을 생성하기 위해 "rewrite" 동작을 수행하도록 요청합니다.
4) "rewrite"는 기존 질문(question)을 이용하여 LLM Prompt로 새로운 질문을 생성합니다. 새로운 질문(better question)은 "retrieve"에 전달되어, 새로운 질문으로 RAG 동작을 재수행할 수 있습니다.  
5) "generate"는 "filtered documents"를 이용하여 적절한 답변(generation)을 생성합니다.
6) "grade_generation"은 생성된 답변이 환각(hallucination)인지 확인하여, 만약 환각이라면 "generator"에 보내 다시 답변을 생성하고, 환각이 아니라면 답변이 적절한지 "answer_question"로 검증합니다. 이때, 답변이 적절하다면(useful) 최종 결과를 전달하고, 적절하지 않다면(not_useful) 질문을 새로 생성하기 위해 "rewrite"합니다. 이후로 새로 생성된 질문은 "retrieve"에 전달되어 RAG 조회 과정을 반복합니다.
   
![image](https://github.com/user-attachments/assets/55672f1a-0b8e-4566-a604-6e5534d9e7d9)

### Self-Corrective RAG

Self-Corrective RAG는 Corrective RAG처럼 vector store로 부터 얻어진 문서의 관련성을 확인하여 관련성이 없는 문서를 제외하고 웹 검색을 통해 결과를 보강합니다. 또한, Self RAG처럼 RAG의 결과가 환각(hallucination)인지, 적절한 답변인지 검증하는 절차를 가지고 있습니다. 상세한 내용은 [self-corrective-rag.md](./self-corrective-rag.md)에서 설명합니다. 아래는 Self-Corrective RAG에 대한 acitivity diagram입니다. 

1) "retrieve"는 질문(question)과 관련된 문서를 Vector Store를 통해 조회합니다. 이때, "grade_generation" 동작을 위해 "web_fallback"을 True로 초기화합니다.
2) "generator"는 Vector Store에서 얻어진 관련된 문서(documents)를 이용하여 답변(generation)을 생성합니다. 이때, retries count를 증가시킵니다.
3) "grade_generation"은 "web_fallback"이 True이라면, "hallucination"과 "answer_question"에서 환각 및 답변의 적절성을 확인합니다. 환각일 경우에, 반복 횟수(retries)가 "max_retries"에 도달하지 않았다면 "generate"보내서 답변을 다시 생성하고, "max_retires"에 도달했다면 "websearch"로 보내서 웹 검색을 수행합니다. 또한 답변이 적절하지 않다면, 반복 횟수가 "max_reties"에 도달하기 않았다면, "rewrite"로 보내서 향상된 질문(better question)을 생성하고, 도달하였다면 "websearch"로 보내서 웹 검색을 수행합니다.
4) "websearch"는 웹 검색을 통해 문서를 보강하고, "generate"에 보내서 답변을 생성합니다. 이때, "web_fallback"을 False로 설정하여 "grade_generation"에서 "finalized_response"로 보내도록 합니다.
5) "rewrite"는 새로운 질문(better question)을 생성하여, "retrieve"에 전달합니다. 새로운 질문으로 전체 RAG 동작을 재수행합니다. 전체 RAG 동작은 무한 루프를 방지하기 위하여, "max_retries"만큼 수행할 수 있습니다.
6) "finalize_response"는 최종 답변을 전달합니다.

![image](https://github.com/user-attachments/assets/5769e8ed-6e76-4fda-a932-a1d3c461de50)



### Human-in-the-loop (HIL)

Workflow에서 Human의 interation을 구현하기 위한 checkpoint를 활용합니다. [Human-in-the-loop (HIL)](./Human-in-the-loop.md)에서는 Human-in-the-loop (HIL)을 구현하는것을 설명합니다. Plan and Execute는 [Plan-and-Solve](https://arxiv.org/pdf/2305.04091)와 같이 CoT 방식의 성능을 향상시키는 방법입니다. 


### Easy Writer

[Essay Writer](./essay-writer.md)에서는 essay를 작성하는 Agent를 생성합니다.

![image](https://github.com/user-attachments/assets/b876e10b-aa5f-4b12-8e0d-a1f803cd7647)


![image](https://github.com/user-attachments/assets/7207687c-5649-4e2d-b4ab-c51ebf9c60f4)

### AgentWrite

[AgentWrite LangGraph](https://github.com/samwit/agent_tutorials/tree/main/agent_write)는 Agent를 이용한 Write 과정에 대해 설명하고 있습니다. 상세한 설명은 [AgentWrite with LangGraph - Youtube](https://www.youtube.com/watch?v=nK9K8UPraXk)를 참고합니다.

   

### Multi Agent 

[multi-agent.md](./multi-agent.md)에서는 여러개의 Agent를 이용하는 방법에 대해 설명합니다. 

![image](https://github.com/user-attachments/assets/7adb21cb-8bd7-4fca-af71-2a8ccc536f49)

### Map Reduce

[Map Reduce](https://github.com/kyopark2014/langgraph-agent/blob/main/map-reduce-parallel-processing.md)에서는 Map Reduce 패턴의 병렬처리 방법에 대해 설명하고 있습니다. 

### RAG with Reflection

[RAG with Reflection](https://github.com/kyopark2014/rag-with-reflection/tree/main?tab=readme-ov-file#rag-with-reflection)에서는 Reflection으로 RAG의 성능을 강화합니다.

<img src="https://github.com/user-attachments/assets/226c1c29-ac7a-4d5d-a515-d91451cc6b81" width="400">

### RAG with Query Transformation

[RAG with Query Transformation](https://github.com/kyopark2014/rag-with-reflection/tree/main?tab=readme-ov-file#query-transformation)에서는 RAG의 전처리로 rewrite와 decompse를 수행함으로써 RAG의 성능을 강화합니다.

<img src="https://github.com/user-attachments/assets/e69f19d3-49e9-435c-bf5f-99fb9a78b2da" width="400">


## Reference

- [reflexion-agent.md](./reflexion-agent.md)에서는 Reflexion방식의 Agent에 대해 설명합니다.

- [persistence-agent.md](./persistence-agent.md)에서는 checkpoint를 이용해 이전 state로 돌아가는 것을 보여줍니다.

- [olympiad-agent.md](./olympiad-agent.md)에서는 Reflection, Retrieval, Human-in-the-loop를 이용해 Olympiad 문제를 푸는것을 설명합니다.

- [code-agent.md](./code-agent.md)에서는 LangGraph를 이용해 code를 생성하는 예제입니다.

- [email-agent.md](./email-agent.md)에서는 LangGraph를 이용해 email을 생성하는 예제입니다.

- [support-bot-agent.md](./support-bot-agent.md)에서는 고객 지원하는 Bot을 Agent로 생성합니다.

- [language-agent-tree-search.md](./language-agent-tree-search.md)에서는 Tree Search 방식의 Agent를 만드는것을 설명합니다.

- [rewoo.md](./rewoo.md)에서는 Reasoning without Observation 방식의 Agent에 대해 설명합니다.

- [llm-compiler.md](./llm-compiler.md)에서는 "An LLM Compiler for Parallel Function Calling"을 구현하는것에 대해 설명합니다. 

- [stome-agent.md](./stome-agent.md)에서는 풍부한 기사를 생성(richer article generation) Storm Agent에 대해 설명합니다.

- [GPT Newspape](https://www.youtube.com/watch?v=E7nFHaSs3q8)에서는 신문요약에 대해 설명하고 있습니다. ([github](https://github.com/rotemweiss57/gpt-newspaper/tree/master) 링크)



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


## 실행결과

채팅 메뉴에서 아래와 같이 agent의 동작방식을 선택하여 결과를 확인할 수 있습니다. 

![image](https://github.com/user-attachments/assets/28cc84db-ffa7-4774-aa82-73d5d699eb31)


<!--
RAG를 위해 아래의 파일들을 다운로드한 후에 채팅창 아래의 파일 아이콘을 선택하여 업로드합니다. 

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)

- [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/pdf/2310.11511)

파일 업로드를 하면 아래와 같이 Summary 결과를 보여줍니다. 

![image](https://github.com/user-attachments/assets/2c333876-2f91-49b1-8379-07374e374d83)

-->

[book_SMMDINLM239.pdf](./contents/book_SMMDINLM239.pdf) 파일을 다운로드 한 후에 채팅창 하단의 파일 아이콘을 이용하여 업로드 합니다. 다량의 문서 파일을 업로드 하는 경우에는 S3로 직접 업로드합니다.

문서에서 그림파일이 있는 경우에 검색의 정확도를 위해 페이지 단위로 요약을 수행합니다. 따라서, 파일 업로드후에 [SQS](https://us-west-2.console.aws.amazon.com/sqs/v3/home?region=us-west-2#/queues)에 접속해서 업로드 상황을 확인합니다. 아래와 같이 파일이 S3로 업로드되면, 파일 업로드 이벤트를 SQS에 저장후 하나씩 꺼내서, Mulimodal인 Claude Sonnet을 이용해 이미지를 분석합니다. 

![image](https://github.com/user-attachments/assets/4a9eeb13-cb9f-425e-b7e8-5ab6c17b2701)

### LangGraph Agent

메뉴에서 "Agent Executor (LangGraph)"를 선택하여 "교보 다이렉트 보험의 장단점을 비교하여 설명하세요."라고 입력합니다. Agent Executor의 Tools에 OpenSearch가 Tool로 등록되어 있으므로 RAG 동작을 확인할 수 있습니다. 

![image](https://github.com/user-attachments/assets/8a2a77a2-8bf6-41e1-a04a-01b114e2213f)

28초가 소요가 소요되었는데, "교보 다이렉트 보험 장단점"에 대해 tavily와 OpenSearch를 검색한 결과를 이용해 응답을 얻었습니다. 이때 OpenSearch에서 얻어진 결과는 아래와 같이 LLM을 이용해 관련도 검사를 한 후에 활용합니다.

![image](https://github.com/user-attachments/assets/efd13f05-d32c-4424-9920-ea7ae3c50a49)

### Corrective RAG 사용시

메뉴에서 "Corrective RAG"를 선택하여 "교보 다이렉트 보험의 장단점을 비교하여 설명하세요."라고 입력합니다. 결과는 LangGraph Agent와 유사하지만, 미리 지정된 Workflow에 따라 원하는 작업을 수행할 수 있습니다.

![image](https://github.com/user-attachments/assets/0a1568e5-88ff-4f74-b138-82182283f4b2)


전체 25초가 소요되었고, retrieve로 얻어진 문서에 대한 평가(grade)를 하고, 조건에 따라 웹검색을 수행할 수 있습니다.

![image](https://github.com/user-attachments/assets/0305302d-2da0-46a9-ae41-32ba92f18df0)


### Self RAG 사용시

메뉴에서 "Self RAG"를 선택하여 "교보 다이렉트 보험의 장단점을 비교하여 설명하세요."라고 입력합니다.

![image](https://github.com/user-attachments/assets/245ac8bf-b61e-4805-a918-34af6d7b6691)

전체 22초가 소요되었고 관련된 문서에 대한 평가후에 환각 및 답변의 유용성을 평가하고 있습니다. 

![image](https://github.com/user-attachments/assets/34ce4f4f-ca24-489c-8d07-04041a208b2a)


### Self Corrective RAG 사용시

메뉴에서 "Self-Corrective RAG"를 선택하여 "교보 다이렉트 보험의 장단점을 비교하여 설명하세요."라고 입력합니다.

![image](https://github.com/user-attachments/assets/500cbed1-45fb-48d2-b3f4-7ca889f044e8)


전체 16초가 소요되었고, 생성된 답변에 대한 환각(Hallucination)과 답변의 유용성을 평가합니다. 

![image](https://github.com/user-attachments/assets/b4f46226-cf5f-444c-95e5-f4524b834e12)


"교보 다이렉트 자동차 보험의 장단점을 비교하여 설명하세요."라고 입력합니다.

![image](https://github.com/user-attachments/assets/727c39e0-6a54-47cc-82fd-ba492c81d784)


### Reflection Agent

메뉴에서 "Reflection Agent"를 선택하고, "생성형 AI 기술에 대해 설명해주세요."라고 입력합니다. 이때, 아래와 같이 90초 시간이 소요되었습니다. 

![image](https://github.com/user-attachments/assets/7fc6f563-5e24-4be5-9569-9c50ba35f1f7)

먼저 LLM이 생성한 초안은 아래와 같습니다. 

```text
여기 생성형 AI 기술에 대한 5문단 에세이 초안이 있습니다:

1문단: 생성형 AI(Generative AI)는 기존 데이터를 기반으로 새로운 콘텐츠를 생성할 수 있는 인공지능 기술을 말합니다. 이는 텍스트, 이미지, 오디오, 코드 등 다양한 형태의 콘텐츠 생성이 가능합니다. 생성형 AI는 기계학습 알고리즘을 사용하여 대량의 데이터에서 패턴을 학습하고 이를 바탕으로 새로운 콘텐츠를 만들어냅니다.

2문단: 대표적인 생성형 AI 기술로는 GPT-3, DALL-E, 뮤직 제너레이터 등이 있습니다. GPT-3는 OpenAI에서 개발한 대규모 언어 모델로, 주어진 텍스트 프롬프트를 기반으로 다양한 형식의 텍스트를 생성할 수 있습니다. DALL-E는 OpenAI의 이미지 생성 AI로, 텍스트 설명을 입력하면 해당하는 이미지를 생성해냅니다. 뮤직 제너레이터는 기존 음원 데이터를 학습하여 새로운 멜로디와 가사를 만들어냅니다.
```

생성된 결과에 대한 평가는 아래와 같습니다.

```text
개선사항:

1. 각 문단의 길이를 200자 이상으로 늘리고, 관련 예시와 구체적인 설명을 추가합니다.

2. 2문단에서 생성형 AI의 작동 원리인 기계학습 알고리즘의 종류(예: 순환 신경망, 변분 자기 부호기 등)와 학습 과정에 대해 자세히 설명합니다.

3. 3문단에서 생성형 AI의 활용 분야를 설명할 때 실제 사례나 구체적인 예시를 들어 설명합니다. 예를 들어 작가의 경우 어떤 방식으로 AI를 활용할 수 있는지, 마케팅 분야에서는 어떤 개인화 콘텐츠를 생성할 수 있는지 등을 구체적으로 설명합니다.

4. 전체적으로 문장 구조와 연결을 자연스럽게 하고, 전문 용어에 대한 설명을 추가하여 가독성을 높입니다.
```

총 3회 수정하여 얻어진 최종 결과는 아래와 같습니다. 초안보다 더 풍부하고 충분한 예제를 제공하고 있습니다. 

```text
감사합니다. 피드백을 반영하여 에세이를 수정하고 보완하겠습니다.

1문단: 생성형 AI(Generative AI)는 기존 데이터를 기반으로 새로운 콘텐츠를 생성할 수 있는 인공지능 기술을 말합니다. 텍스트, 이미지, 오디오, 코드 등 다양한 형태의 콘텐츠 생성이 가능합니다. 생성형 AI는 기계학습 알고리즘, 특히 순환 신경망(Recurrent Neural Network)이나 변분 자기부호기(Variational Autoencoder) 등을 사용하여 대량의 데이터에서 패턴을 학습합니다. 그리고 이렇게 학습된 모델을 바탕으로 새로운 콘텐츠를 만들어냅니다.

2문단: 대표적인 생성형 AI 기술로는 GPT-3, DALL-E, 뮤직 제너레이터 등이 있습니다. GPT-3는 OpenAI에서 개발한 거대한 언어 모델로, 175억 개의 매개변수를 가진 순환 신경망 기반의 모델입니다. 이 모델은 주어진 텍스트 프롬프트를 기반으로 다양한 형식의 텍스트를 생성할 수 있습니다. DALL-E는 OpenAI의 이미지 생성 AI로, 변분 자기부호기 기반의 모델입니다. 이 모델은 텍스트 설명을 입력받아 해당하는 이미지를 생성해냅니다. 뮤직 제너레이터는 기존 음원 데이터를 순환 신경망으로 학습하여 새로운 멜로디와 가사를 만들어냅니다.
```

### Easy Writer

메뉴에서 "Easy Writer"를 선택하고 나서, "즐겁게 사는 방법"와 같이 입력하고 결과를 확인하면 아래와 같습니다. 아래의 경우에는 138초가 소요되었습니다. 

![easy-writer](https://github.com/user-attachments/assets/6fda99c8-a902-49c8-a82a-994569429932)

### Knowledge Guru

메뉴에서 "Knowledge Guru"를 선택하고 생성형 AI를 위해서는 데이터가 중요한데요. 데이터를 어떻게 가공할수 있는지 알려주세요."와 같이 검색합니다. 이때 OpenSearch에 생성형 AI와 관련된 데이터들이 있어야 합니다. "Knowledge Guru"의 경우에 OpenSearch의 검색 결과를 LLM으로 평가하여 관련된 문서만을 추출하여 답변을 생성합니다. 또한 추출된 답변에 추가 데이터를 제공하기 위하여 중요 키워드를 추출한 후에 OpenSearch를 이용해 추가 데이터를 수집하여 답변을 향상(Revise)합니다. 

![noname](https://github.com/user-attachments/assets/ebd7fa92-ee56-4380-890b-6b3ec40ee1d6)



## 결론

여러 방식의 agent를 구현하고 동작 방식을 선택하여 사용할 수 있었습니다. Agent의 활용 목적에 따라서 결과와 실행 시간의 trade off를 고려하여야 합니다. 

- RAG 검색을 Tool로 등록하고 LangGraph Agent를 실행하면, ReAct 방식으로 질문을 평가하면서 여러번 질의를 수행할 수 있습니다.
- Reflection Agent 방식은 시간이 오래 소요가 되지만 가장 풍부한 답변을 얻을 수 있었습니다.
- Corrective RAG는 상대적으로 시간이 적게 소요가 되지만 RAG의 Vector Search에서 얻어진 문서를 평가하여 적절한 문서를 고르고 검색으로 얻어진 결과를 모아서 RAG만 검색했을 때보다 좋은 결과를 얻을 수 있습니다.
- Self RAG와 Corrective Self RAG는 관련된 문서의 평가뿐 아니라, 환각과 답변의 적절성을 평가할 수 있습니다. 웹검색 필요에 따라서 Self RAG와 Self Corrective RAG를 선택적으로 사용할 수 있습니다. 


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-east-1.console.aws.amazon.com/apigateway/main/apis?region=us-east-1)로 접속하여 "rest-api-for-langgraph-agent", "ws-api-for-langgraph-agent"을 삭제합니다.

2) [Cloud9 Console](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/langgraph-agent/cdk-langgraph-agent/ && cdk destroy --all
```
