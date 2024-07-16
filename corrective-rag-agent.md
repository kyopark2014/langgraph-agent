# Corrective RAG Agent

## Corrective RAG 란?

Corrective-RAG(CRAG)는 검색된 문서에 대한 Self Refection / Self Grading을 포함하고 있는 RAG Strategy입니다. 

- [LangGraph Corrective RAG (CRAG)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)

- [논문: Corrective Retrieval Augmented Generation - 2024/02](https://arxiv.org/pdf/2401.15884)
    - 문서로 답변을 생성하기 전에 knowledge refinement을 수행합니다. 이를 위해 문서를 knowledge strip으로 분할하고 평가(grade)하여 관련 없는 문서는 제외합니다. 모든 문서가 임계치 이하이거나 평가를 확신할 수 없는 경우에는 Knowledge Search를 하거나 웹 검색(Web search)를 수행합니다.
      
<img src="https://github.com/user-attachments/assets/1e065d21-88fb-43fa-b904-9d42b50f5762" width="600">

- [Corrective RAG (CRAG) - Mistral](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/corrective_rag_mistral.ipynb)


## Corrective RAG의 구현

여기서 구현하려는 Corrective RAG의 형태는 아래와 같습니다. 상세한 코드는 [corrective-rag.ipynb](./agent/corrective-rag.ipynb)와 [corrective-rag-kor.ipynb](./agent/corrective-rag-kor.ipynb)를 참조합니다. 

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
class GradeDocuments(BaseModel):
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
)

retrieval_grader = grade_prompt | structured_llm_grader
```

RAG을 위한 Prompt를 정의합니다.

```python
system = (
"""다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
                        
<context>
{context}
</context>""")
human = "{question}"
    
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                   
rag_chain = prompt | chat
```

ReWrite를 위한 Prompt를 정의합니다.

```python
class RewriteQuestion(BaseModel):
    """rewrited question that is well optimized for retrieval."""

    question: str = Field(description="The new question is optimized for web search")
    
structured_llm_rewriter = chat.with_structured_output(RewriteQuestion)

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

question_rewriter = re_write_prompt | structured_llm_rewriter
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

각 노드를 정의합니다.

```python
def retrieve(state: GraphState):
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(doc)
        else:
            web_search = "Yes"
            continue
    return {"question": question, "documents": filtered_docs, "web_search": web_search}

def decide_to_generate(state: GraphState):
    question = state["question"]
    filtered_documents = state["documents"]
    web_search = state["web_search"]
    
    if web_search == "Yes":
        return "rewrite"
    else:
        return "generate"

def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": docs, "question": question})    
    return {"documents": documents, "question": question, "generation": generation}

def rewrite(state: GraphState):
    question = state["question"]

    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question.question}

def web_search(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]    
    return {"question": question, "documents": documents}
```

이제 Graph를 이용하여 Workflow를 정의합니다.

```python
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

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

# Compile
app = workflow.compile()
```

아래와 같이 실행할 수 있습니다.

```python
from pprint import pprint
inputs = {"question": "이미지를 분석하기 위한 서비스에 대해 설명해줘."}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
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
