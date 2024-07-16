# Code Agent

[Self-correcting code assistants with Codestral](https://www.youtube.com/watch?v=zXFxmI9f06M)에서는 LangGraph를 이용해 code 생성하는것을 보여줍니다.

[langgraph_code_assistant_mistral.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/langgraph_code_assistant_mistral.ipynb)에서는 상세한 코드를 보여줍니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/edf75f76-93b2-439a-8977-146248091e9d)

```python
class GraphState(TypedDict):
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

Graph는 아래와 같이 정의합니다.

```python
builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("generate", generate)  # generation solution
builder.add_node("check_code", code_check)  # check code

# Build graph
builder.set_entry_point("generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)
```

이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/b021ea5e-13b3-46b2-a889-4cf5dcc4304a)


