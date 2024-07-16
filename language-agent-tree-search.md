# Language Agent Tree Search

[lats.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb?ref=blog.langchain.dev)에서는 reflection, evaluation, search을 이용해 전체적인 성능을 높입니다.

참고한 문헌은 [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406)와 같습니다. 

Language Agent Tree Search (LATS)의 형태는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/09f9f7d1-bab2-4609-8ae5-dbe980b366fb)


이것읜 형태는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/92c34cf9-c3a2-4890-bd16-2856ebfde42a)

```python
builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.set_entry_point("start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
)

graph = builder.compile()
````

구현된 결과는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/bf61e626-638d-4e02-9835-5909822ae914)
