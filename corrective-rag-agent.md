# Corrective RAG Agent

## Corrective RAG 란?

Corrective-RAG(CRAG)는 검색된 문서에 대한 Self Refection / Self Grading을 포함하고 있는 RAG Strategy입니다. 

- [LangGraph Corrective RAG (CRAG)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)

- [논문: Corrective Retrieval Augmented Generation - 2024/02](https://arxiv.org/pdf/2401.15884)
    - 문서로 답변을 생성하기 전에 knowledge refinement을 수행합니다. 이를 위해 문서를 knowledge strip으로 분할하고 평가(grade)하여 관련 없는 문서는 제외합니다. 모든 문서가 임계치 이하이거나 평가를 확신할 수 없는 경우에는 Knowledge Search를 하거나 웹 검색(Web search)를 수행합니다.
      
<img src="https://github.com/user-attachments/assets/1e065d21-88fb-43fa-b904-9d42b50f5762" width="600">

- [Corrective RAG (CRAG) - Mistral](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/corrective_rag_mistral.ipynb)


## Corrective RAG의 구현

여기서 구현하려는 Corrective RAG의 형태는 아래와 같습니다. 상세한 코드는 [lambda_function.py](lambda-chat-ws)을 참조합니다. 더불어 동작을 [corrective-rag.ipynb](./agent/corrective-rag.ipynb)와 [corrective-rag-kor.ipynb](./agent/corrective-rag-kor.ipynb)를 이용해 확인할 수 있습니다.

Corrective RAG의 동작 Flow는 아래와 같습니다. 

1) 사용자가 질문(Question)을 읽어오면 RAG의 Vector Store로 retrieve 동작을 수행합니다. 이때 k개의 관련된 문서(relevant docuements)을 가져옵니다.
2) grade_document()는 LLM prompt를 이용하여 Vector Store에가 가져온 문서가 실제로 관련이 있는지 확인합니다. 관련이 있으면 "yes", 없으면 "no"를 판별(grade)하는데, "no"인 경우에 관련된 문서에서 제외합니다. 만약 관련된 문서가 관련성이 없어 제외되면, "websearch"를 True로 설정합니다. 
3) decide_to_geenerate()는 모든 문서가 관련성이 있으면, 답변을 생성하는 generate()로 이동하고, 하나의 문서라도 제외되면 websearch로 이동합니다.
4) web search가 필요할 경우에 기존 질문이 충분히 의도(sementic intent)와 의미(meaning)을 가지도록 새로운 질문으로 변경(re-write)합니다.
5) web search()에서는 인터넷을 검색하여 새로운 관련된 문서를 찾아 추가합니다.
6) generate()에서는 관련된 문서를 context로 활용하여 적절한 답변을 생성합니다. 

<img src="https://github.com/user-attachments/assets/dc495cc0-2912-4bb3-a1f8-807d05d7b35a" width="200">

기본 채팅을 위한 chat을 정의합니다. 

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            "다음의 Human과 Assistant의 친근한 이전 대화입니다."
            "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | chat
```

GradeDocuments Class를 정의하고 structed out을 이용하여, document가 관련된 문서인지를 yes/no로 응답하도록 합니다. 

```python
**class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = chat.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)**

retrieval_grader = grade_prompt | structured_llm_grader
```

RAG을 위한 Prompt를 정의합니다.

```python
def get_reg_chain():
    if langMode:
        system = (
        """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

        <context>
        {context}
        </context>""")
    else: 
        system = (
        """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        <context>
        {context}
        </context>""")
        
    human = "{question}"
        
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                    
    chat = get_chat()
    rag_chain = prompt | chat
    return rag_chain
```

ReWrite를 위한 Prompt를 정의합니다.

```python
def get_rewrite():
    class RewriteQuestion(BaseModel):
        """rewrited question that is well optimized for retrieval."""

        question: str = Field(description="The new question is optimized for web search")
    
    chat = get_chat()
    structured_llm_rewriter = chat.with_structured_output(RewriteQuestion)
    
    print('langMode: ', langMode)
    
    if langMode:
        system = """당신은 웹 검색에 최적화된 더 나은 버전의 Question으로 변환하는 질문 re-writer입니다. 질문의 의도와 의미을 잘 표현할 수 있는 한국어 질문을 생성하세요."""
    else:
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
    print('system: ', system)
        
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    question_rewriter = re_write_prompt | structured_llm_rewriter
    return question_rewriter
```

Graph State를 정의합니다.

```python
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
```

OpenSeach를 이용해 vector 검색으로 관련된 문서를 찾습니다. 여기서는 관련된 문서를 parent/child chunking을 한 후에 child chunk를 이용해 검색 정확도를 높이고, parent chunk를 이용해 context를 풍부하게 활용합니다. 

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
    return {"documents": docs, "question": question}
```

각 노드를 정의합니다. 

```python
def grade_documents(state: CragState):
    print("###### grade_documents ######")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    
    retrieval_grader = get_grader()
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
            web_search = "Yes"
            continue
    print('len(docments): ', len(filtered_docs))
    print('web_search: ', web_search)

def decide_to_generate(state: CragState):
    print("###### decide_to_generate ######")
    web_search = state["web_search"]
    
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "rewrite"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def generate(state: CragState):
    print("###### generate ######")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    rag_chain = get_reg_chain()
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    print('generation: ', generation.content)
    
    return {"documents": documents, "question": question, "generation": generation}

def rewrite(state: CragState):
    print("###### rewrite ######")
    question = state["question"]

    # Prompt
    question_rewriter = get_rewrite()
    
    better_question = question_rewriter.invoke({"question": question})
    print("better_question: ", better_question.question)

    return {"question": better_question.question}

def web_search(state: CragState):
    print("###### web_search ######")
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    print("web_results: ", web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"question": question, "documents": documents}
```

이제 Graph를 이용하여 Workflow를 정의합니다.

```python
def buildCorrectiveAgent():
    workflow = StateGraph(CragState)
        
    # Define the nodes
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("websearch", web_search)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    workflow.add_edge("rewrite", "websearch")
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

crag_app = buildCorrectiveAgent()
```

아래와 같이 실행할 수 있습니다.

```python
def run_corrective_rag(connectionId, requestId, app, query):
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

이때의 결과의 예는 아래와 같습니다.

```text
###### retrieve ######
document[0]:  page_content='주요 기능
 이미지 분석:' metadata={'source': 'https://docs.aws.amazon.com/ko_kr/rekognition/latest/dg/what-is.html', 'title': 'Amazon Rekognition이란 무엇인가요? - Amazon Rekognition', 'description': '딥 러닝 이미지 분석 서비스인 Amazon Rekognition의 개요.', 'language': 'ko-KR'}
'Finished running: retrieve:'
###### grade_documents ######
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
len(docments):  4
web_search:  No
###### decide_to_generate ######
---DECISION: GENERATE---
'Finished running: grade_documents:'
###### generate ######
generation:  네, Amazon Rekognition은 딥러닝 기술을 활용한 이미지 및 비디오 분석 서비스입니다. 주요 기능에 대해 설명드리겠습니다.

1. 물체, 장면, 개념 감지: 이미지에서 사물, 장면, 개념, 유명인을 감지하고 분류할 수 있습니다.

2. 텍스트 감지: 다양한 언어로 된 이미지에서 인쇄된 텍스트와 손으로 쓴 텍스트를 감지하고 인식합니다.

3. 안전하지 않은 콘텐츠 탐지: 노골적이거나 부적절하고 폭력적인 콘텐츠와 이미지를 탐지하고 필터링할 수 있습니다.

4. 유명인 인식: 정치인, 운동선수, 배우, 음악가 등 다양한 분야의 유명인을 인식합니다.

5. 얼굴 분석: 성별, 나이, 감정 등의 얼굴 특성을 분석하고 얼굴을 감지 및 비교할 수 있습니다.

6. 사용자 지정 레이블: 로고, 제품, 문자 등 사용자 지정 분류기를 만들어 원하는 객체를 탐지할 수 있습니다.

7. 이미지 속성 분석: 품질, 색상, 선명도, 대비 등의 이미지 속성을 분석합니다.

Amazon Rekognition은 AWS 서비스와 통합되어 있고 확장성과 보안을 제공하며, 사용한 만큼만 비용을 지불하는 저렴한 가격 정책을 가지고 있습니다.
'Finished running: generate:'
```


## Reference

### Mistral LLM을 이용한 Corrective RAG 구현 

[corrective_rag_mistral.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/corrective_rag_mistral.ipynb)에서는 문서를 검색할 때에 self-reflection /self-grading을 적용합니다.

<img src="https://github.com/kyopark2014/llm-agent/assets/52392004/dcb682f5-35e4-4478-8189-5db5cdbb266d" width="600">
   

```python
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]

def retrieve(state):
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"
```

Graph을 생성합니다.

```python
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("websearch", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```


[langgraph_crag_mistral.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/langgraph_crag_mistral.ipynb)에서는 Self Reflection을 이용해 RAG의 성능을 강화합니다.


<img src="https://github.com/kyopark2014/llm-agent/assets/52392004/3a2618d0-0e81-4900-976e-78d30fd19a0e" width="600">
  

아래와 같이 Graph를 생성합니다.

```python
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```
