# Self Corrective RAG

여기에서는 Self Corrective RAG를 구현합니다.

## Self Corrective RAG

Self-Corrective RAG는 Corrective RAG처럼 Vector Store로 부터 얻어진 문서의 관련성을 확인하여 관련성이 없는 문서를 제외하고 웹 검색을 통해 결과를 보강합니다. 또한, Self RAG처럼 RAG의 결과가 환각(Hallucination)인지, 적절한 답변인지 검증하는 절차를 가지고 있습니다. 아래는 Self-Corrective RAG에 대한 acitivity diagram입니다. 

1) "retrieve"는 질문(question)과 관련된 문서를 Vector Store를 통해 조회합니다. 이때, "grade_generation" 동작을 위해 "web_fallback"을 True로 초기화합니다.
2) "generator"는 Vector Store에서 얻어진 관련된 문서(documents)를 이용하여 답변(generation)을 생성합니다. 이때, retries count를 증가시킵니다.
3) "grade_generation"은 "web_fallback"이 True이라면, "hallucination"과 "answer_question"에서 환각 및 답변의 적절성을 확인합니다. 환각일 경우에, 반복 횟수(retries)가 "max_retries"에 도달하지 않았다면 "generate"보내서 답변을 다시 생성하고, "max_retires"에 도달했다면 "websearch"로 보내서 웹 검색을 수행합니다. 또한 답변이 적절하지 않다면, 반복 횟수가 "max_reties"에 도달하기 않았다면, "rewrite"로 보내서 향상된 질문(better question)을 생성하고, 도달하였다면 "websearch"로 보내서 웹 검색을 수행합니다.
4) "websearch"는 웹 검색을 통해 문서를 보강하고, "generate"에 보내서 답변을 생성합니다. 이때, "web_fallback"을 False로 설정하여 "grade_generation"에서 "finalized_response"로 보내도록 합니다.
5) "rewrite"는 새로운 질문(better question)을 생성하여, "retrieve"에 전달합니다. 새로운 질문으로 전체 RAG 동작을 재수행합니다. 전체 RAG 동작은 무한 루프를 방지하기 위하여, "max_retries"만큼 수행할 수 있습니다.
6) "finalize_response"는 최종 답변을 전달합니다.


![image](https://github.com/user-attachments/assets/5769e8ed-6e76-4fda-a932-a1d3c461de50)

이때 생성되는 Graph는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/766c27fe-a943-4262-a72b-486c69578f83)



## Self Corrective RAG by LangGraph

[Self-Corrective RAG in LangGraph](https://github.com/vbarda/pandas-rag-langgraph/blob/main/demo.ipynb)을 참조합니다.

아래와 같이 Hallucination인지 관련된 문서인지를 LLM을 통해 판별합니다. 설정된 루프보다 더 많은 task를 수행하면, 인터넷 검색을 통해 결과를 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/b94d70a6-e740-44b0-9918-770c3ea64f2a)
