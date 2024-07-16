# Self RAG

[LangGraph - Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb?ref=blog.langchain.dev)와 같이 Self RAG는 RAG를 grade 한 후에 얻어진 결과가 환각(hallucination)을 하는지 확인하는 절차를 포함합니다. 결과가 만족하지 않을 경우에는 cycle을 통해 반복적으로 Answer를 찾습니다.

<img width="934" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/d066967c-b92c-4951-973f-753d24e3e491">


### Grader 

RAG를 통해 조회된 결과를 평가해서 yes, no를 찾는 Prompt 입니다.

```python
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})
```

### Rewriter

답변을 rewrite해서 새로운 질문을 만들어 RAG 조회의 전과정을 반복합니다.

```python
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
```

