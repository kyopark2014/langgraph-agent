# Self RAG

[LangGraph - Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb?ref=blog.langchain.dev)와 같이 Self RAG는 RAG를 grade 한 후에 얻어진 결과가 환각(hallucination)을 하는지 확인하는 절차를 포함합니다. 결과가 만족하지 않을 경우에는 cycle을 통해 반복적으로 Answer를 찾습니다. 아래는 Self RAG에 대한 activity diagram입니다. 

1) "retrive"는 질문(question)을 이용하여 Vector Store에 관련된 문서를 조회(retrieve)합니다.
2) "grade_documents"는 LLM Prompt를 이용하여 문서(documents)의 관련성을 확인(grade)합니다. 관련이 없는 문서는 제외하여 "filtered documents"로 제조합합니다. 
3) "decide_to_generate"는 "filtered document"를 "generate"로 보내서 답변을 생성하도록 합니다. "filtered document"가 없다면 새로운 질문을 생성하기 위해 "rewrite" 동작을 수행하도록 요청합니다.
4) "rewrite"는 기존 질문(question)을 이용하여 LLM Prompt로 새로운 질문을 생성합니다. 새로운 질문(better question)은 "retrieve"에 전달되어, 새로운 질문으로 RAG 동작을 재수행할 수 있습니다.  
5) "generate"는 "filtered documents"를 이용하여 적절한 답변(generation)을 생성합니다.
6) "grade_generation"은 생성된 답변이 환각(hallucination)인지 확인하여, 만약 환각이라면 "generator"에 보내 다시 답변을 생성하고, 환각이 아니라면 답변이 적절한지 "answer_question"로 검증합니다. 이때, 답변이 적절하다면(useful) 최종 결과를 전달하고, 적절하지 않다면(not_useful) 질문을 새로 생성하기 위해 "rewrite"합니다. 이후로 새로 생성된 질문은 "retrieve"에 전달되어 RAG 조회 과정을 반복합니다.

![image](https://github.com/user-attachments/assets/55672f1a-0b8e-4566-a604-6e5534d9e7d9)


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

