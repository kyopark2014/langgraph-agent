# Customer Support Bot

[Build a Customer Support Bot | LangGraph](https://www.youtube.com/watch?v=b3XsvoFWp4c)에서는 고객 지원 App 개발을 위한 Agent를 설명합니다.

[Blog - Build a Customer Support Bot](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)에서는 Flights/ Car Rental / Hotels에 대한 Tool에 대해 설명하고 있습니다.

[customer-support.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/customer-support/customer-support.ipynb)에서는 Customer Support Bot을 Agent로 구현하는 방법에 대해 설명하고 있습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/1e595125-8e22-478c-9eb8-e4ebb301d8a1)

```python
@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully booked."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            passenger_id = config.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
```

아래와 같이 Graph을 설정합니다.

```python
builder = StateGraph(State)

builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = SqliteSaver.from_conn_string(":memory:")
part_1_graph = builder.compile(checkpointer=memory)
```

이렇게 생성된 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/650e7b5f-c2de-48e1-835f-37df14b89ae0)


