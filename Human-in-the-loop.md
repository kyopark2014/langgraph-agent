# Human-in-the-loop (HIL)

여기에서는 LangGraph의 [Human-in-the-loop (HIL)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/)을 따라서 구현해보고자 합니다. 

상세 구현 내용은 [human-in-the-loop.ipynb](./agent/human-in-the-loop.ipynb)을 참조합니다.

## LangGraph Agent

Chat Model을 아래와 같이 준비합니다.

```python
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock
bedrock_region = 'us-east-1'
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
maxOutputTokens = 4096
parameters = {
    "max_tokens":maxOutputTokens,     
    "temperature":0.1,
    "top_k":250,
    "top_p":0.9,
    "stop_sequences": [HUMAN_PROMPT]
}    
chat = ChatBedrock(   
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
)
```

인터넷 검색을 하는 Tool을 선언합니다.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

tool_node = ToolNode(tools)
model = chat.bind_tools(tools)
```

LangGraph를 위한 함수를 준비합니다.

```python
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    print('response: ', response)

    return {"messages": [response]}
```


Agent를 준비합니다.

```python
import operator
from typing import Annotated, Sequence, TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile()
```

이때 생성된 Workflow는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/2f51bf01-67c0-4de9-87cf-4fe981688dc5)

Workflow를 실행한 결과는 아래와 같습니다.

```python
inputs = [HumanMessage(content="현재 서울의 날씨는?")]
for event in app.stream({"messages": inputs}, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

이때 실행 결과는 아래와 같습니다.

```text
================================ Human Message =================================

현재 서울의 날씨는?
response:  content='' additional_kwargs={'usage': {'prompt_tokens': 275, 'completion_tokens': 59, 'total_tokens': 334}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 275, 'completion_tokens': 59, 'total_tokens': 334}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-9b01e846-0b60-46eb-b2e5-4ecb14ae6f37-0' tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'seoul weather'}, 'id': 'toolu_bdrk_01JZtYLPUGHozpu9LKsyMyry', 'type': 'tool_call'}] usage_metadata={'input_tokens': 275, 'output_tokens': 59, 'total_tokens': 334}
================================== Ai Message ==================================
Tool Calls:
  tavily_search_results_json (toolu_bdrk_01JZtYLPUGHozpu9LKsyMyry)
 Call ID: toolu_bdrk_01JZtYLPUGHozpu9LKsyMyry
  Args:
    query: seoul weather
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'Seoul', 'region': '', 'country': 'South Korea', 'lat': 37.57, 'lon': 127.0, 'tz_id': 'Asia/Seoul', 'localtime_epoch': 1722588587, 'localtime': '2024-08-02 17:49'}, 'current': {'last_updated_epoch': 1722588300, 'last_updated': '2024-08-02 17:45', 'temp_c': 31.2, 'temp_f': 88.2, 'is_day': 1, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/day/176.png', 'code': 1063}, 'wind_mph': 6.7, 'wind_kph': 10.8, 'wind_degree': 250, 'wind_dir': 'WSW', 'pressure_mb': 1003.0, 'pressure_in': 29.63, 'precip_mm': 0.59, 'precip_in': 0.02, 'humidity': 67, 'cloud': 72, 'feelslike_c': 37.1, 'feelslike_f': 98.7, 'windchill_c': 31.2, 'windchill_f': 88.2, 'heatindex_c': 37.1, 'heatindex_f': 98.7, 'dewpoint_c': 24.2, 'dewpoint_f': 75.6, 'vis_km': 9.0, 'vis_miles': 5.0, 'uv': 7.0, 'gust_mph': 7.9, 'gust_kph': 12.8}}"}]
response:  content='서울의 현재 날씨는 화씨 88.2도(섭씨 31.2도)이며 근처에 부분적인 비가 내리고 있습니다. 풍속은 시속 10.8km(6.7마일)이고 습도는 67%입니다. 전반적으로 무더운 날씨이지만 부분적인 비로 인해 다소 선선할 수 있습니다.' additional_kwargs={'usage': {'prompt_tokens': 829, 'completion_tokens': 135, 'total_tokens': 964}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 829, 'completion_tokens': 135, 'total_tokens': 964}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-2cb405cb-83f4-4d1a-8438-1687dfb07b16-0' usage_metadata={'input_tokens': 829, 'output_tokens': 135, 'total_tokens': 964}
================================== Ai Message ==================================

서울의 현재 날씨는 화씨 88.2도(섭씨 31.2도)이며 근처에 부분적인 비가 내리고 있습니다. 풍속은 시속 10.8km(6.7마일)이고 습도는 67%입니다. 전반적으로 무더운 날씨이지만 부분적인 비로 인해 다소 선선할 수 있습니다.
```

## Checkpoint

checkpoint를 지정하고 "action"에서 멈추도록 workflow를 구성합니다.

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

아래와 같이 thread를 지정하고 실행합니다.

```python
thread = {"configurable": {"thread_id": "5"}}
inputs = [HumanMessage(content="현재 서울의 날씨는?")]

for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

아래와 같이 전체가 실행되지 않고, "action" node 실행전에 workflow가 멈춥니다.

```text
================================ Human Message =================================

현재 서울의 날씨는?
response:  content='' additional_kwargs={'usage': {'prompt_tokens': 275, 'completion_tokens': 59, 'total_tokens': 334}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 275, 'completion_tokens': 59, 'total_tokens': 334}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-12de4ab3-4397-40d8-9c03-2c9bf1612704-0' tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'seoul weather'}, 'id': 'toolu_bdrk_017MExPG887HkKDR36uiM8HL', 'type': 'tool_call'}] usage_metadata={'input_tokens': 275, 'output_tokens': 59, 'total_tokens': 334}
================================== Ai Message ==================================
Tool Calls:
  tavily_search_results_json (toolu_bdrk_017MExPG887HkKDR36uiM8HL)
 Call ID: toolu_bdrk_017MExPG887HkKDR36uiM8HL
  Args:
    query: seoul weather
```

아래와 같이 tool에 전달된 메시지를 확인할 수 있습니다.

```python
current_state = app.get_state(thread)
last_message = current_state.values["messages"][-1]
last_message.tool_calls[0]["args"]
```

아래와 같이 재실행하면 Resume 동작을 수행할 수 있습니다.

```python
for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

이때의 결과는 아래와 같습니다.

```text
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'Seoul', 'region': '', 'country': 'South Korea', 'lat': 37.57, 'lon': 127.0, 'tz_id': 'Asia/Seoul', 'localtime_epoch': 1722588587, 'localtime': '2024-08-02 17:49'}, 'current': {'last_updated_epoch': 1722588300, 'last_updated': '2024-08-02 17:45', 'temp_c': 31.2, 'temp_f': 88.2, 'is_day': 1, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/day/176.png', 'code': 1063}, 'wind_mph': 6.7, 'wind_kph': 10.8, 'wind_degree': 250, 'wind_dir': 'WSW', 'pressure_mb': 1003.0, 'pressure_in': 29.63, 'precip_mm': 0.59, 'precip_in': 0.02, 'humidity': 67, 'cloud': 72, 'feelslike_c': 37.1, 'feelslike_f': 98.7, 'windchill_c': 31.2, 'windchill_f': 88.2, 'heatindex_c': 37.1, 'heatindex_f': 98.7, 'dewpoint_c': 24.2, 'dewpoint_f': 75.6, 'vis_km': 9.0, 'vis_miles': 5.0, 'uv': 7.0, 'gust_mph': 7.9, 'gust_kph': 12.8}}"}]
response:  content='서울의 현재 날씨는 화씨 88.2도(섭씨 31.2도)이며 부분적으로 비가 내리고 있습니다. 풍속은 시속 10.8km(6.7마일)이고 습도는 67%입니다. 전반적으로 무더운 날씨이지만 가끔 소나기가 내리고 있는 상황입니다.' additional_kwargs={'usage': {'prompt_tokens': 829, 'completion_tokens': 129, 'total_tokens': 958}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} response_metadata={'usage': {'prompt_tokens': 829, 'completion_tokens': 129, 'total_tokens': 958}, 'stop_reason': 'end_turn', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'} id='run-e4208f03-6b2b-4548-a3b6-06ae22612e6b-0' usage_metadata={'input_tokens': 829, 'output_tokens': 129, 'total_tokens': 958}
================================== Ai Message ==================================

서울의 현재 날씨는 화씨 88.2도(섭씨 31.2도)이며 부분적으로 비가 내리고 있습니다. 풍속은 시속 10.8km(6.7마일)이고 습도는 67%입니다. 전반적으로 무더운 날씨이지만 가끔 소나기가 내리고 있는 상황입니다.
```

