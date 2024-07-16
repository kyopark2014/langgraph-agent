# Self Corrective RAG

여기에서는 Self Corrective RAG를 구현합니다.

## Self Corrective RAG

![image](https://github.com/user-attachments/assets/9aeb268b-59e3-4749-9a36-0aeec22bb60e)



## Self Corrective RAG by LangGraph

[Self-Corrective RAG in LangGraph](https://github.com/vbarda/pandas-rag-langgraph/blob/main/demo.ipynb)을 참조합니다.

아래와 같이 Hallucination인지 관련된 문서인지를 LLM을 통해 판별합니다. 설정된 루프보다 더 많은 task를 수행하면, 인터넷 검색을 통해 결과를 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/b94d70a6-e740-44b0-9918-770c3ea64f2a)
