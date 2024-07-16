# Corrective RAG Agent

[Advance RAG control flow with Mistral and LangChain: Corrective RAG, Self-RAG, Adaptive RAG](https://www.youtube.com/watch?v=sgnrL7yo1TE)에서는 Self Reflection을 이용해 RAG의 성능을 향상시킵니다.

## Corrective RAG

여기서 구현하련느 Corrective RAG의 형태는 아래와 같습니다. 상세한 코드는 [corrective-rag.ipynb](./agent/corrective-rag.ipynb)을 참조합니다. 

![image](https://github.com/user-attachments/assets/63e3caa9-0b97-472a-bbcc-2132afce46f5)

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
#"""다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
"""Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        
<context>
{context}
</context>""")
human = "{question}"
    
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                   
rag_chain = prompt | chat
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
        return "websearch"
    else:
        return "generate"

def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": docs, "question": question})    
    return {"documents": documents, "question": question, "generation": generation}

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
app = workflow.compile()
```

아래와 같이 실행할 수 있습니다.

```python
inputs = {"question": "Aver의 Glue는 무엇이지?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
```


## 예제: Mistral LLM을 이용한 Corrective RAG 구현 

[corrective_rag_mistral.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/corrective_rag_mistral.ipynb)에서는 문서를 검색할 때에 self-reflection /self-grading을 적용합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/dcb682f5-35e4-4478-8189-5db5cdbb266d)

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

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a2618d0-0e81-4900-976e-78d30fd19a0e)


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
