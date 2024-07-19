# Self Corrective RAG

## Basic 

여기에서는 [Self RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)와 [Corrective RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)를 결합하여 RAG의 성능을 향상시키는 방법에 대해 설명합니다. Self-Corrective RAG는 Corrective RAG처럼 Vector Store로 부터 얻어진 문서의 관련성을 확인하여 관련성이 없는 문서를 제외하고 웹 검색을 통해 결과를 보강합니다. 또한, Self RAG처럼 RAG의 결과가 환각(Hallucination)인지, 적절한 답변인지 검증하는 절차를 가지고 있습니다. 아래는 Self-Corrective RAG에 대한 acitivity diagram입니다. 

1) "retrieve"는 질문(question)과 관련된 문서를 Vector Store를 통해 조회합니다. 이때, "grade_generation" 동작을 위해 "web_fallback"을 True로 초기화합니다.
2) "generator"는 Vector Store에서 얻어진 관련된 문서(documents)를 이용하여 답변(generation)을 생성합니다. 이때, retries count를 증가시킵니다.
3) "grade_generation"은 "web_fallback"이 True이라면, "hallucination"과 "answer_question"에서 환각 및 답변의 적절성을 확인합니다. 환각일 경우에, 반복 횟수(retries)가 "max_retries"에 도달하지 않았다면 "generate"보내서 답변을 다시 생성하고, "max_retires"에 도달했다면 "websearch"로 보내서 웹 검색을 수행합니다. 또한 답변이 적절하지 않다면, 반복 횟수가 "max_reties"에 도달하기 않았다면, "rewrite"로 보내서 향상된 질문(better question)을 생성하고, 도달하였다면 "websearch"로 보내서 웹 검색을 수행합니다.
4) "websearch"는 웹 검색을 통해 문서를 보강하고, "generate"에 보내서 답변을 생성합니다. 이때, "web_fallback"을 False로 설정하여 "grade_generation"에서 "finalized_response"로 보내도록 합니다.
5) "rewrite"는 새로운 질문(better question)을 생성하여, "retrieve"에 전달합니다. 새로운 질문으로 전체 RAG 동작을 재수행합니다. 전체 RAG 동작은 무한 루프를 방지하기 위하여, "max_retries"만큼 수행할 수 있습니다.
6) "finalize_response"는 최종 답변을 전달합니다.

![image](https://github.com/user-attachments/assets/5769e8ed-6e76-4fda-a932-a1d3c461de50)

## 상세 구현

상세 코드는 [lambda_function.py](lambda-chat-ws)을 참조합니다. 동작 결과는 [cself-corrective-rag.ipynb](./agent/self-corrective-rag.ipynb)에서 확인할 수 있습니다.

Self Corrective RAG를 위한 class와 환경 설정을 위한 config를 아래와 같이 정의합니다.  

```python
class SelfCorrectiveRagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    candidate_answer: str
    retries: int
    web_fallback: bool

class GraphConfig(TypedDict):
    max_retries: int    
```

환각(Hallucination)을 평가하기 위한 get_hallucination_grader()을 정의합니다. 

```python
def get_hallucination_grader():
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    chat = get_chat()
    structured_llm_grade_hallucination = chat.with_structured_output(GradeHallucinations)
    
    hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
    return hallucination_grader
```

답변의 유용성을 평가하기 위한 get_answer()를 정의합니다. 

###  Node 

```python
def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
    
    chat = get_chat()
    structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
    
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader
```

답변을 생성하기 위한 generate()를 정의합니다. 

```python
def generate_with_retires(state: CragState):
    print("###### generate ######")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1
    
    # RAG generation
    rag_chain = get_reg_chain()
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    print('generation: ', generation.content)
    
    return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}
```

관련된 문서들에서 각 문서별로 관련도를 LLM으로 평가합니다.

```python
def grade_documents_with_count(state: SelfRagState):
    print("###### grade_documents ######")
    question = state["question"]
    documents = state["documents"]
    count = state["count"] if state.get("count") is not None else -1
    
    # Score each doc
    filtered_docs = []
    retrieval_grader = get_retrieval_grader()
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            continue
    print('len(docments): ', len(filtered_docs))    
    return {"question": question, "documents": filtered_docs, "count": count + 1}
```

최종 답변에서 응답을 추출하기 위한 Node입니다. 

```python
def finalize_response(state: SelfCorrectiveRagState):
    return {"messages": [AIMessage(content=state["candidate_answer"])]}
```

### Conditional Edge

생성된 문서의 관련도 평가를 기준으로 적절한 동작을 수행할 수 있도록 conditinal edge를 정의합니다.

```python
def decide_to_generate_with_retires(state: SelfRagState, config):
    print("###### decide_to_generate ######")
    filtered_documents = state["documents"]
    
    count = state["count"] if state.get("count") is not None else -1
    max_count = config.get("configurable", {}).get("max_counts", MAX_RETRIES)
    print("count: ", count)
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "no document" if count < max_count else "not available"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "document"
```

답변이 환각인지, 유용한 답변인지 확인해서 적절한 동작을 수행하기 위한 conditional edge를 정의합니다.

```python
def grade_generation(state: SelfRagState, config):
    print("###### grade_generation ######")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    hallucination_grader = get_hallucination_grader()
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    hallucination_grade = score.binary_score
    
    print("hallucination_grade: ", hallucination_grade)
    print("retries: ", retries)

    # Check hallucination
    answer_grader = get_answer_grader()    
    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.binary_score        
        print("answer_grade: ", answer_grade)

        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful" 
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful" if retries < max_retries else "not available"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported" if retries < max_retries else "not available"
```

### Graph

이제 Workflow를 정의하기 위한 Graph를 선언합니다.

```python
def buildSelCorrectivefRAG():
    workflow = StateGraph(SelfCorrectiveRagState)
        
    # Define the nodes
    workflow.add_node("retrieve", retrieve_for_scrag)  
    workflow.add_node("generate", generate_for_scrag) 
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("websearch", web_search)
    workflow.add_node("finalize_response", finalize_response)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("finalize_response", END)

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "generate": "generate",
            "websearch": "websearch",
            "rewrite": "rewrite",
            "finalize_response": "finalize_response",
        },
    )

    # Compile
    return workflow.compile()

scrag_app = buildSelfRAG()
```

이때 생성되는 Graph는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/766c27fe-a943-4262-a72b-486c69578f83)


아래와 같이 Self Corrective RAG를 실행합니다.

```python
def run_self_corrective_rag(connectionId, requestId, app, query):
    global langMode
    langMode = isKorean(query)
    
    isTyping(connectionId, requestId)
    
    inputs = {"question": query}
    config = {"recursion_limit": 50}
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}:")
            print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["generation"].content)
    
    return value["generation"].content
```



## Reference

[Self-Corrective RAG in LangGraph](https://github.com/vbarda/pandas-rag-langgraph/blob/main/demo.ipynb)을 참조합니다.

아래와 같이 Hallucination인지 관련된 문서인지를 LLM을 통해 판별합니다. 설정된 루프보다 더 많은 task를 수행하면, 인터넷 검색을 통해 결과를 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/b94d70a6-e740-44b0-9918-770c3ea64f2a)
