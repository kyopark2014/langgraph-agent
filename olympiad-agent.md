# Olympiad Agent

LangGraph를 이용해 [Olympiad 문제를 푸는 Agent](https://www.youtube.com/watch?v=UqYzzjTmKj8&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&index=16)를 생성합니다.

[Blog0-lympiad Agent](https://langchain-ai.github.io/langgraph/tutorials/usaco/usaco/)에서는 Reflection, Retrieval, Human-in-the-loop 에 대해 설명합니다.

## Part 1: Zero-Shot with Reflection

relection에서는 Solve / Evaluate를 반복해가며 문제를 해결합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/b612ee62-d3f0-4483-9c14-3ca78b9c117f)

Graph 생성코드는 아래와 같습니다. 

```python
builder = StateGraph(State)
builder.add_node("solver", solver)
builder.set_entry_point("solver")
builder.add_node("evaluate", evaluate)
builder.add_edge("solver", "evaluate")


def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solver"


builder.add_conditional_edges("evaluate", control_edge, {END: END, "solver": "solver"})
graph = builder.compile()
```

이것의 구현된 Graph는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/f1446780-ea3b-41c9-b3cd-616a9919f05e)


## Part 2: Few-shot Retrieval

이때의 Graph는 아래와 같습니다.

```python
builder = StateGraph(State)
builder.add_node("draft", draft_solver)
builder.set_entry_point("draft")
builder.add_node("retrieve", retrieve_examples)
builder.add_node("solve", solver)
builder.add_node("evaluate", evaluate)
# Add connectivity
builder.add_edge("draft", "retrieve")
builder.add_edge("retrieve", "solve")
builder.add_edge("solve", "evaluate")


def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solve"


builder.add_conditional_edges("evaluate", control_edge, {END: END, "solve": "solve"})


checkpointer = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=checkpointer)
```

생성된 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/cbb6aa30-f6dc-4ebe-b828-a9c7587c7913)


## Part 3: Human-in-the-loop

```python
builder = StateGraph(State)
prompt = hub.pull("wfh/usaco-draft-solver")
llm = ChatAnthropic(model="claude-3-opus-20240229", max_tokens_to_sample=4000)

draft_solver = Solver(llm, prompt.partial(examples=""))
builder.add_node("draft", draft_solver)
builder.set_entry_point("draft")
builder.add_node("retrieve", retrieve_examples)
solver = Solver(llm, prompt)
builder.add_node("solve", solver)
builder.add_node("evaluate", evaluate)
builder.add_edge("draft", "retrieve")
builder.add_edge("retrieve", "solve")
builder.add_edge("solve", "evaluate")


def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solve"


builder.add_conditional_edges("evaluate", control_edge, {END: END, "solve": "solve"})
checkpointer = SqliteSaver.from_conn_string(":memory:")
```

이때의 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/8e2cde6a-108c-4f1b-b531-88110a55ff29)

이대의 결과는 아래와 같습니다.

Reflection: 이전 결과에 대한 비판(critique)를 장려하여 relection을 구현

Retrieval: "episodic memory"를 높은 품질의 few shot 문제를 해결했습니다.

Human-in-the-loop: 인간의 피드백을 이용해 대부분의 답변을 찾아냈습니다. 



