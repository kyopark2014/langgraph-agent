# Reflexion Agent

[reflexion.ipynb](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflexion/reflexion.ipynb)에서는 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)을 기반한 Reflexion을 구현하고 있습니다.

Reflexion의 Diagram은 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/469174cb-5ae9-444f-a19c-68261bab65dd)

feedback과 self-reflection을 이용해 더 높은 성능의 결과를 얻습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/fcaab550-b7ec-4edb-9fcf-576135075391)

Graph의 구현 코드는 아래와 같습니다. 

```python
MAX_ITERATIONS = 5
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)

builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge("draft", "execute_tools") # draft -> execute_tools
builder.add_edge("execute_tools", "revise") # execute_tools -> revise

def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i

def event_loop(state: list) -> Literal["execute_tools", "__end__"]:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)  # revise -> execute_tools OR end
builder.set_entry_point("draft")
graph = builder.compile()
```

이것의 구현된 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/00f6d691-1b19-4fa9-9d1a-6049698d9d00)


[Reflexion](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflexion/reflexion.ipynb)에서는 AnswerQuestion/Reflectin 클래스를 이용하여 문장에서 Reflection에 필요한 정보를 추출합니다.

```python
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
```
