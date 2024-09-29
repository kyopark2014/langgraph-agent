# Query Transformation

## 관련 참고문서

[Query Transformations for Improved Retrieval in RAG Systems](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)에서는 query를 개선하여 RAG의 성능을 향상시키는 방법으로 Query Rewriting, Step-back Prompting, Sub-query Decomposition을 설명하고 있습니다.

[RAG의 Query Transformation](https://medium.com/@krtarunsingh/advanced-rag-techniques-unlocking-the-next-level-040c205b95bc)은 여러개의 Subquery를 이용하여 RAG의 성능을 향상시키는 방법입니다. 

![image](https://github.com/user-attachments/assets/ea32be3d-9d19-473e-840d-9ebf0b4cdf28)


쿼리 변환 기술은 LLM을 활용하여 사용자 입력을 변경하거나 개선함으로써 정보 검색의 품질과 관련성을 높입니다. 이러한 변환은 다양한 형태를 취할 수 있습니다: 

### 하위 쿼리 분해 (Sub-query Decompostion)

- 복잡한 쿼리를 간단한하고 쉬운 하위 쿼리로 분해합니다. 
- 각 하위 쿼리를 독립적으로 처리한 후에 종합하여 포괄적인 응답을 형성합니다. 
- LangChain과 LlamaIndex 모두 이 프로세스를 촉진하는 Multi Query Retriever 및 Sub Question Query Engine과 같은 도구를 제공

### 단계별 프롬프팅 (Step-back Prompting)

- 복잡한 원래 쿼리에서 더 넓거나 일반적인 쿼리를 생성하기 위해 LLM을 사용합니다.
- 구체적인 쿼리에 답변하기 위한 기반이 될 수 있는 상위 수준의 컨텍스트를 검색을 목표로 합니다.
- 원래 쿼리와 일반화된 쿼리에서 얻은 컨텍스트를 결합하여 최종 답변 생성을 향상시킵니다.
- LangChain의 [Step Back Prompting](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/step_back/)을 참조합니다.

```python
class StepBackQuery(BaseModel):
    step_back_question: str = Field(
        ...,
        description="Given a specific user question about one or more of these products, write a more generic question that needs to be answered in order to answer the specific question.",
    )
```
  

### 쿼리 재작성 (Query Rewriting)

- 초기 쿼리를 개선하여 검색 프로세스를 향상시키기 위해 LLM을 사용합니다.
- LangChain과 LlamaIndex 모두 이 전략을 구현하지만 접근 방식에는 차이가 있습니다. 특히 LlamaIndex는 검색 효율성을 크게 향상시키는 강력한 구현으로 주목받고 있습니다. (확인 필요)

- [langchain/cookbook/rewrite.ipynb](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb?ref=blog.langchain.dev)

```python
template = (
  "Provide a better search query for web search engine to answer the given question,"
  "end the queries with ’**’."
  "Question: {x} Answer:"
)
rewrite_prompt = ChatPromptTemplate.from_template(template)
```

[langchain-ai/rewrite](https://smith.langchain.com/hub/langchain-ai/rewrite?tab=0)를 참조합니다.

```python
Provide a better search query for web search engine to answer the given question, end the queries with ’**’.  Question {x} Answer:
```


