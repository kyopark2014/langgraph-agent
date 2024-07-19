# Agentic RAG

Agentic RAG는 tool_condition을 통해 RAG에 retrival을 선택하고, 문서를 평가(grade)하여, 검색 결과가 만족스럽지 않다면 re-write를 통해 새로운 질문(better question)을 생성할 수 있습니다. 상세한 내용은 [langgraph_agentic_rag.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)를 참조합니다. 

Agentic RAG의 activity diagram은 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/a7371a2e-59b6-4806-a2e4-14bf75560cb1)

```python
from langgraph.prebuilt import tools_condition
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_agent",
    "Search and return information on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]
```

아래와 같이 Graph를 정의합니다. 

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node(
    "generate", generate
)  
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()
```

