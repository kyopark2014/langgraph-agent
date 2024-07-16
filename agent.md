# Agent

여기에서는 Agent에 대해 상세히 다룹니다.

## Schema

### AgentAction

- tool: tool의 이름
- tool_input: tool의 input

### AgentFinish

Agent로 부터의 final result를 의미합니다.

return_values: final agent output을 포함하고 있는 key-value. output key를 가지고 있습니다. 

### Intermediate Steps

이전 Agent action을 나타내는것으로 CURRENT agent의 실행으로 인한 output을 포함하고 있습니다. List[Tuple[AgentAction Any]]로 구성됩니다. 


  

## Tool의 설정

### RAG의 Knowledge store를 이용 (Retriever)

[14-Agent/04-Agent-with-various-models.ipynb](https://github.com/teddylee777/langchain-kr/blob/main/14-Agent/04-Agent-with-various-models.ipynb)을 참조합니다.

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="2023년 12월 AI 관련 정보를 PDF 문서에서 검색합니다. '2023년 12월 AI 산업동향' 과 관련된 질문은 이 도구를 사용해야 합니다!",
)
```

### DB query


## Reasoning 방식

### Reasoning의 정의 

- 상황에 대한 인식을 바탕으로 새로운 사실을 유도
- Reasoning in Artificial Intelligence refers to the process by which AI systems analyze information, make inferences, and draw conclusions to solve problems or make decisions. It is a fundamental cognitive function that enables machines to mimic human thought processes and exhibit intelligent behavior.



## Reference

[Agents with local models](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)


[LangChain Agents & LlamaIndex Tools](https://cobusgreyling.medium.com/langchain-agents-llamaindex-tools-e74fd15ee436)에서는 아래와 같은 cycle을 설명하고 있습니다. 

- 어떤 요청(request)를 받았을때 agent는 LLM이 하려고 하는 어떤 action을 할지 결정할때 이용된다.

- Action이 완료된 후에 Observation하고 이후에 Thought에서는 Final Answer에 도달한지 확인한다. Final answer가 아니라면 다른 action을 수행하는 cycle을 거친다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/6b2032db-c259-43f3-a699-7eca41117d45)


[Introducing LangChain Agents: 2024 Tutorial with Example](https://brightinventions.pl/blog/introducing-langchain-agents-tutorial-with-example/)

- Agent는 언어 모델을 이용하여 일련의 action(sequence of actions)들을 선택한다. 여기서 Agent는 결과를 얻기 위하여 action들을 결정하는데 reasoning engine을 이용하고 있다.

- Agent는 간단한 자동 응답(automated response)로 부터 복잡한(complex), 상황인식(context-aare)한 상호연동(interaction)하는 task들을 처리하는데(handling) 중요하다.

- Agent는 Tools, LLM, Prompt로 구성된다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/e0ab693a-1b7b-492d-a19c-30dd4dddded1)

Tool에는 아래와 같은 종류들이 있습니다. 

- Web search tool: Google Search, Tavily Search, DuckDuckGo
- Embedding search in vector database
- 인터넷 검색, reasoning step
- API integration tool
- Custom Tool

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c746c149-ecee-48fa-9c0c-ce66d03c4f34)

#### Agent와 Chain의 차이점

Chain은 연속된 action들로 hardcoding되어 있어서 다른 path를 쓸수 없습니다. 즉, agent는 관련된 정보를 이용하여 결정을 할 수 있고, 원하는 결과를 얻을때까지 반복적으로 다시 할 수 있습니다.

<img src="https://github.com/kyopark2014/llm-agent/assets/52392004/92233da1-03c4-4949-8636-b91191c975ce" width="700">
