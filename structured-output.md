# Structured Output

## Chat 지정

Bedrock을 이용해 chat을 설정합니다.

```python
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

## 사용 방법

### 사용자 정보 추출하기

```python
class User(BaseModel):
    name: str
    age: int
    
structured_llm = chat.with_structured_output(User, include_raw=True)

info = structured_llm.invoke("Jason is 25 years old.")

user_info = info['parsed']

print('name: ', user_info.name)
print('age: ', user_info.age)
```

이때의 결과는 아래와 같습니다. 

```text
name:  Jason
age:  25
```

LLM의 parsing 결과는 아래와 같습니다. 

```java
{
   "raw":"AIMessage(content=""",
   "additional_kwargs="{
      "usage":{
         "prompt_tokens":322,
         "completion_tokens":50,
         "total_tokens":372
      },
      "stop_reason":"tool_use",
      "model_id":"anthropic.claude-3-sonnet-20240229-v1:0"
   },
   "response_metadata="{
      "usage":{
         "prompt_tokens":322,
         "completion_tokens":50,
         "total_tokens":372
      },
      "stop_reason":"tool_use",
      "model_id":"anthropic.claude-3-sonnet-20240229-v1:0"
   },
   "id=""run-94e916c0-db53-4bd7-aeca-eb30605981a6-0",
   "tool_calls="[
      {
         "name":"User",
         "args":{
            "name":"Jason",
            "age":25
         },
         "id":"toolu_bdrk_01RQkKNJefn6bogdDzQWHhrq"
      }
   ],
   "usage_metadata="{
      "input_tokens":322,
      "output_tokens":50,
      "total_tokens":372
   }")",
   "parsed":"User(name=""Jason",
   age=25),
   "parsing_error":"None"
}
```

### 정보의 추출

```python
class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str
    
structured_llm = chat.with_structured_output(AnswerWithJustification, include_raw=True)

info = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

parsed_info = info['parsed']

print('answer: ', parsed_info.answer)
print('justification: ', parsed_info.justification)
```

이때의 결과는 아래와 같습니다. 

```text
answer:  A pound of bricks and a pound of feathers weigh the same.
justification:  A pound is a unit of weight or mass, not volume. Since a pound of bricks and a pound of feathers both have the same mass (one pound), they must weigh the same amount. The fact that bricks are denser and take up less volume than feathers for the same weight is irrelevant - their weights are equal when the mass is the same. This is often used as a riddle to trick people into thinking the bricks would be heavier due to their greater density, but by definition of the pound unit, equal masses must have equal weights.
```

### Plan의 추출

```python
from typing import Annotated, List, Tuple, TypedDict

class Plan(BaseModel):
    """List of steps. The updated plan should be in the following format:
<plan>
[\"<step>\", \"<step>\", ...]
</plan>"""
    steps: List[str] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

structured_llm = chat.with_structured_output(Plan, include_raw=True)

info = structured_llm.invoke("To find the hometown of the 2024 Australian Open winner, we would need to follow these steps:\n\n1. Wait for the 2024 Australian Open tennis tournament to take place (typically in January 2024 in Melbourne, Australia).\n2. Identify the winner of the men's singles or women's singles tournament.\n3. Research biographical information about the 2024 Australian Open winner to determine their hometown or place of birth.\n4. The hometown or birthplace of the 2024 Australian Open winner is the final answer.\n\nSince the 2024 Australian Open has not happened yet, we cannot provide the actual hometown until the tournament takes place and the winner is determined. The key steps are to wait for the event, identify the winner, and then research their biographical details to find their hometown or birthplace.")

parsed_info = info['parsed']
parsed_info.steps
```

이때의 결과는 아래와 같습니다. 

```text
{'raw': AIMessage(content='', additional_kwargs={'usage': {'prompt_tokens': 561, 'completion_tokens': 131, 'total_tokens': 692}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, response_metadata={'usage': {'prompt_tokens': 561, 'completion_tokens': 131, 'total_tokens': 692}, 'stop_reason': 'tool_use', 'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'}, id='run-b6259c7e-ffbd-4ef2-956f-e816ea4a5d66-0', tool_calls=[{'name': 'Plan', 'args': {'steps': ['Wait for the 2024 Australian Open tennis tournament to take place in January 2024.', "Identify the winner of the men's singles or women's singles tournament.", 'Research biographical information about the 2024 Australian Open winner.', 'Determine the hometown or place of birth of the winner from the biographical information.', 'Response: The hometown of the 2024 Australian Open winner is [their hometown/birthplace].']}, 'id': 'toolu_bdrk_01E89VxYe4bPmT7Nmc12rjB9'}], usage_metadata={'input_tokens': 561, 'output_tokens': 131, 'total_tokens': 692}),
 'parsed': Plan(steps=['Wait for the 2024 Australian Open tennis tournament to take place in January 2024.', "Identify the winner of the men's singles or women's singles tournament.", 'Research biographical information about the 2024 Australian Open winner.', 'Determine the hometown or place of birth of the winner from the biographical information.', 'Response: The hometown of the 2024 Australian Open winner is [their hometown/birthplace].']),
 'parsing_error': None}

['Wait for the 2024 Australian Open tennis tournament to take place in January 2024.',
 "Identify the winner of the men's singles or women's singles tournament.",
 'Research biographical information about the 2024 Australian Open winner.',
 'Determine the hometown or place of birth of the winner from the biographical information.',
 'Response: The hometown of the 2024 Australian Open winner is [their hometown/birthplace].']
```


