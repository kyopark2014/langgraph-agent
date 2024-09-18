# Map Reduce 방식의 병렬처리

[How to create map-reduce branches for parallel execution](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)에 따라 Map Reduce 형태로 workflow를 생성할 수 있습니다. 상세한 내용은 [LangGraph - Controllability with Map Reduce (Youtube)](https://www.youtube.com/watch?v=JQznvlSatPQ)을 참조합니다. 

![image](https://github.com/user-attachments/assets/549270bb-f24f-454d-8386-17891e145526)

상세한 코드는 [map-reduce.ipynb](./agent/map-reduce.ipynb)을 참조합니다.

generate_topics 노드를 정의합니다. 

```python
class Subjects(BaseModel):
    """List of subjects as a json format"""

    subjects: List[str] = Field(
        description="different subjects to follow"
    )
def generate_topics(state: OverallState):
    topic = state['topic']
    human = (
        "Generate list of between 2 and 5 examples related to: {topic}."
        "Provide the only final answer"
    )
    subjects_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", human),
        ]
    )

    subject = subjects_prompt | chat

    output = subject.invoke({"topic": topic})
    
    structured_llm = chat.with_structured_output(Subjects, include_raw=True)
    response = structured_llm.invoke(output.content)

    return {"subjects": response['parsed'].subjects}
```


Joke 노드를 정의합니다.

```python
class Joke(BaseModel):
    """List of jokes as a json format"""
    
    joke: List[str] = Field(
        description="a list of jokes to follow"
    )
class JokeState(TypedDict):
    subject: str
def generate_joke(state: JokeState):
    subject = state['subject']
    
    human = (
        "Generate a joke about {subject}."
        "Provide the only final answer"
    )
    jok_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", human),
        ]
    )

    subject_chain = jok_prompt | chat

    output = subject_chain.invoke({"subject": subject})
    print('output: ', output)

    structured_llm = chat.with_structured_output(Joke, include_raw=True)
    response = structured_llm.invoke(output.content)

    return {"jokes": response['parsed'].joke}
```

Conditional edge을 정의합니다. 

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```

```python
class BestJoke(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0", ge=0)

def best_joke(state: OverallState):
    topic = state['topic']
    jokes = "\n\n".join(state["jokes"])
    
    human = (
        "Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one."
        "Provide the only final answer."
    )
    best_joke_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", human),
            ("human", "{jokes}")
        ]
    )

    best_joke_chain = best_joke_prompt | chat

    output = best_joke_chain.invoke({
        "topic": topic,
        "jokes": jokes
    })
    print('output: ', output)

    structured_llm = chat.with_structured_output(BestJoke, include_raw=True)
    response = structured_llm.invoke(output.content)
    print('response: ', response)
    
    best_selected_joke = state["jokes"][response['parsed'].id]
    
    return {"best_selected_joke": best_selected_joke}
```

이제 State Diagram을 준비합니다.

```python
from langgraph.constants import Send
from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict, Literal, Sequence, Union
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list

    jokes: Annotated[list, operator.add]
    best_selected_joke: str

graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()
```

완성된 diagram은 아래와 같습니다.

```python
from IPython.display import Image

Image(app.get_graph().draw_mermaid_png())
```

![image](https://github.com/user-attachments/assets/1b1bfa71-30e6-4978-85fd-9f9698b3307b)

아래와 같이 실행합니다.

```python
for s in app.stream({"topic": "animals"}):
    print(s)
```

이때의 결과는 아래와 같습니다.

```text
{'generate_topics': {'subjects': ['cat', 'dog', 'lion', 'giraffe']}}
output:  content="Here's a joke about cats:\n\nWhy did the cat go to the computer programming session? Because he wanted to learn some new mouse clicks!" additional_kwargs={'usage': {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-579d536a-e960-4905-af4a-8e802c172583-0' usage_metadata={'input_tokens': 19, 'output_tokens': 32, 'total_tokens': 51}
output:  content="Here's a joke about a dog:\n\nWhy did the dog cross the road twice? Because he was a double-crosser!" additional_kwargs={'usage': {'prompt_tokens': 19, 'completion_tokens': 30, 'total_tokens': 49}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 19, 'completion_tokens': 30, 'total_tokens': 49}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-eee580ab-7ae9-4393-b2d8-f007e7ec5953-0' usage_metadata={'input_tokens': 19, 'output_tokens': 30, 'total_tokens': 49}
output:  content="Here's a joke about a lion:\n\nWhy did the lion eat the tightrope walker? Because he wanted a well-balanced meal!" additional_kwargs={'usage': {'prompt_tokens': 19, 'completion_tokens': 33, 'total_tokens': 52}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 19, 'completion_tokens': 33, 'total_tokens': 52}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-665f8d45-ca86-4539-a948-0129b59ce4ac-0' usage_metadata={'input_tokens': 19, 'output_tokens': 33, 'total_tokens': 52}
{'generate_joke': {'jokes': ['Why did the cat go to the computer programming session? Because he wanted to learn some new mouse clicks!']}}
{'generate_joke': {'jokes': ['Why did the dog cross the road twice? Because he was a double-crosser!']}}
output:  content='Why did the giraffe get a neck massage? Because it was a little stiff!' additional_kwargs={'usage': {'prompt_tokens': 21, 'completion_tokens': 22, 'total_tokens': 43}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 21, 'completion_tokens': 22, 'total_tokens': 43}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-8a55789c-a90d-496f-8232-b2c90d2ad55c-0' usage_metadata={'input_tokens': 21, 'output_tokens': 22, 'total_tokens': 43}
{'generate_joke': {'jokes': ['Why did the lion eat the tightrope walker? Because he wanted a well-balanced meal!']}}
{'generate_joke': {'jokes': ['Why did the giraffe get a neck massage? Because it was a little stiff!']}}
output:  content='3' additional_kwargs={'usage': {'prompt_tokens': 118, 'completion_tokens': 5, 'total_tokens': 123}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 118, 'completion_tokens': 5, 'total_tokens': 123}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-300a711a-4503-470d-aabd-fd68adb72d1e-0' usage_metadata={'input_tokens': 118, 'output_tokens': 5, 'total_tokens': 123}
response:  {'raw': AIMessage(content='', additional_kwargs={'usage': {'prompt_tokens': 330, 'completion_tokens': 33, 'total_tokens': 363}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, response_metadata={'usage': {'prompt_tokens': 330, 'completion_tokens': 33, 'total_tokens': 363}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, id='run-0f6e9bfe-e66e-4180-9846-46e51264db00-0', tool_calls=[{'name': 'BestJoke', 'args': {'id': 3}, 'id': 'toolu_bdrk_01ScMKVP3CC4zwE4QCR6hRcn', 'type': 'tool_call'}], usage_metadata={'input_tokens': 330, 'output_tokens': 33, 'total_tokens': 363}), 'parsed': BestJoke(id=3), 'parsing_error': None}
{'best_joke': {'best_selected_joke': 'Why did the giraffe get a neck massage? Because it was a little stiff!'}}
```
