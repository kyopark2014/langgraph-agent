# Storm Agent

[storm.ipynb]에서는 [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207)을 이용하여, 풍부한 기사를 생성(richer article generation)을 위한 "outline-driven RAG"의 아이디어를 확장합니다. 

Storm의 Overview는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/2d0bc93c-e1ab-4bf2-97be-b558b2127453)

Graph의 구현형태는 아래와 같습니다.

```python
builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question)
builder.add_node("answer_question", gen_answer)
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.set_entry_point("ask_question")
interview_graph = builder.compile().with_config(run_name="Conduct Interviews")
```

이를 도식화하면 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/2f8b165a-497b-41f4-acc8-fbfec5f76b02)


