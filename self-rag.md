# Self RAG

[LangGraph - Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb?ref=blog.langchain.dev)와 같이 Self RAG는 Vector Store의 검색 결과의 관련성을 LLM으로 검증(grade)하고, 생성된 결과가 환각(hallucination)인지, 답변이 적절한지를 검증하는 절차를 포함합니다. 결과가 만족하지 않을 경우에는 cycle을 통해 반복적으로 Answer를 찾습니다. 

## Basic

Self RAG는 Self Reflection을 베이스로 [(2023.10) Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)에서 제안되었습니다. retrieve, critique, generate text passages 과정을 통해 전반적인 품질(Overall quality), 사실성(factuality), 검증 가능성(verifiability)을 향상시킵니다. 

![image](https://github.com/user-attachments/assets/ad83adf8-600b-4c3b-b601-9ad6f48f235b)


## Self RAG의 구현

아래는 Self RAG에 대한 activity diagram입니다. 

1) "retrive"는 질문(question)을 이용하여 Vector Store에 관련된 문서를 조회(retrieve)합니다.
2) "grade_documents"는 LLM Prompt를 이용하여 문서(documents)의 관련성을 확인(grade)합니다. 관련이 없는 문서는 제외하여 "filtered documents"로 제조합합니다. 
3) "decide_to_generate"는 "filtered document"를 "generate"로 보내서 답변을 생성하도록 합니다. "filtered document"가 없다면 새로운 질문을 생성하기 위해 "rewrite" 동작을 수행하도록 요청합니다.
4) "rewrite"는 기존 질문(question)을 이용하여 LLM Prompt로 새로운 질문을 생성합니다. 새로운 질문(better question)은 "retrieve"에 전달되어, 새로운 질문으로 RAG 동작을 재수행할 수 있습니다.  
5) "generate"는 "filtered documents"를 이용하여 적절한 답변(generation)을 생성합니다.
6) "grade_generation"은 생성된 답변이 환각(hallucination)인지 확인하여, 만약 환각이라면 "generator"에 보내 다시 답변을 생성하고, 환각이 아니라면 답변이 적절한지 "answer_question"로 검증합니다. 이때, 답변이 적절하다면(useful) 최종 결과를 전달하고, 적절하지 않다면(not_useful) 질문을 새로 생성하기 위해 "rewrite"합니다. 이후로 새로 생성된 질문은 "retrieve"에 전달되어 RAG 조회 과정을 반복합니다.

![image](https://github.com/user-attachments/assets/55672f1a-0b8e-4566-a604-6e5534d9e7d9)

### 재시도 숫자 제한의 필요성

답변을 얻지 못하면 recursion_limit만큼 반복한 후에 exception error와 함께 실패하게 됩니다. 따라서 아래와 같이 retries, count를 이용해 재시도 숫자를 제한하였습니다. 

- 질문으로 Vector Store 조회시 max_count만큼 "rewrite"를 반복해도 관련된 문서(docuemnts)를 얻지 못하는 경우에는 문서 없이 "generate"에서 답변을 생성합니다.
- "generate"에서 생성한 답변이 환각(Hallucination) 또는 적절하지 않은(not usueful)이라면, max_retries만큼 재시도하다가 생성된 답변을 최종 답변으로 전달합니다. 

![image](https://github.com/user-attachments/assets/4dc8a762-70f5-4e1b-9d2e-d082cc9e74a5)

### 상세 구현

구현된 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)에서 확인할 수 있습니다. 또한, 동작 방식은 [self-rag.ipynb](./agent/self-rag.ipynb)을 참조합니다. 

Self RAG에 대한 class와 재시도에 대한 환경값을 위해 아래와 같이 정의합니다. 

```python
class SelfRagState(TypedDict):
    question : str
    generation : str
    retries: int  # number of generation 
    count: int # number of retrieval
    documents : List[str]
    
class GraphConfig(TypedDict):
    max_retries: int    
    max_count: int
```

환각(Hallucination)을 평가(Grade)하는 함수를 정의합니다. 환각이 없는 경우는 "yes", 있는 경우는 "no"로 결과를 얻습니다. 이를 위해 Chat Model과 함께 [structed output](./structured-output.md)을 활용하였습니다. 

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

RAG의 얻어진 답변을 평가(grade)합니다.

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

RAG의 Vector Store로 부터 문서를 조회하는 retrieve를 정의합니다. 여기서는 Vector Store로 OpenSearch를 활용하였고 Parent/Child Chunking을 이용한 Chunk strategy로 검색 정확도를 향상시키고 있습니다. 

```python
def retrieve(state: CragState):
    print("###### retrieve ######")
    question = state["question"]

    # Retrieval
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = "idx-*", # all
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    top_k = 4
    docs = []    
    if enalbeParentDocumentRetrival == 'true':
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, question, top_k)

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
            excerpt, name, uri, doc_level = get_parent_document(parent_doc_id) # use pareant document
            print(f"parent: name: {name}, uri: {uri}, doc_level: {doc_level}")
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'doc_level': doc_level,
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = question,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content        
            uri = document[0].metadata['uri']
                            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                    },
                )
            )    
    return {"documents": docs, "question": question}
```

관련된 문서를 활용해 답변을 생성하는 generate를 정의합니다.

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

관련된 문서들을 하나씩 꺼내서 질문과 관련도를 LLM prompt를 이용해 확인합니다. 

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

답변이 유용하지 않을때는 질문을 re-write하여 재검색을 수행합니다.

```python
def rewrite(state: CragState):
    print("###### rewrite ######")
    question = state["question"]
    documents = state["documents"]

    # Prompt
    question_rewriter = get_rewrite()
    
    better_question = question_rewriter.invoke({"question": question})
    print("better_question: ", better_question.question)

    return {"question": better_question.question, "documents": documents}
```

Conditional Edge인 decide_to_generate()를 아래와 같이 정의합니다. 무한 루프를 방지하기 위하여 max_count를 활용합니다. 

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

grade_generation()은 conditional edge로서 답변이 환각(Hallucination)인지 적절한 답변(answer_grade)인지 확인하여 적절한 동작을 선택합니다. 

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

Workflow를 위한 Graph를 정의합니다.

```python
def buildSelfRAG():
    workflow = StateGraph(SelfRagState)
        
    # Define the nodes
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents_with_count)
    workflow.add_node("generate", generate_with_retires)
    workflow.add_node("rewrite", rewrite)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_with_retires,
        {
            "no document": "rewrite",
            "document": "generate",
            "not available": "generate",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "rewrite",
            "not available": END,
        },
    )

    return workflow.compile()

srag_app = buildSelfRAG()
```

이때 생성된 Graph는 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/266de2c8-1927-4d02-81ca-bad7de2237fe)

아래와 같이 실행할 수 있습니다. 

```python
def run_self_rag(connectionId, requestId, app, query):
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

