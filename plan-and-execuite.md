# Plan-and-Execute Agent
[LangGraph: Planning Agents](https://www.youtube.com/watch?v=uRya4zRrRx4)에서는 3가지 plan-and-execution 형태의 agent를 설명하고 있습니다. [plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb)에서는 [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)에 대한 Agent를 정의하고 있습니다.

LangGraph은 stateful하고 multi-actor 애플리케이션을 만들 수 있도록 돕는 오픈 소스 framework입니다. 이를 통해 빠르게 실행하고, 비용을 효율적으로 사용하고 성능을 향상 시킬 수 있습니다. 

## Plan-and-Execute 

[plan-and-execute.ipynb](./agent/plan-and-execute.ipynb)와 같이 Plan-and-Execute 동작을 수행하는 Agent를 만들 수 있습니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다. 


![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a97d0764-2891-4454-8854-522ef3249e44)

이때의 activity diagram은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/04b2168d-8bd6-481e-8b9c-5282562118cc)


## 상세 구현

Plan and execute의 State 클래스와 workflow는 아래와 같습니다. Plan 노드에서 생성된 draft 형태의 plan은 매 plan이 실행될때마다 업데이트 되고 이때 얻어진 결과는 info에 array로 저장됩니다. 

```python
class State(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[List[Tuple], operator.add]
    info: Annotated[List[Tuple], operator.add]
    answer: str

def buildPlanAndExecute():
    workflow = StateGraph(State)
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("replaner", replan_node)
    workflow.add_node("final_answer", final_answer)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replaner")
    workflow.add_conditional_edges(
        "replaner",
        should_end,
        {
            "continue": "executor",
            "end": "final_answer",
        },
    )
    workflow.add_edge("final_answer", END)

    return workflow.compile()
```

Plan 노드는 아래와 같습니다.

```python
class Plan(BaseModel):
    """List of steps as a json format"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

def get_planner():
    system = (
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps."
        "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."
    )
        
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{messages}"),
        ]
    )
    
    chat = get_chat()   
    
    planner = planner_prompt | chat
    return planner

def plan_node(state: State, config):
    print("###### plan ######")
    
    inputs = [HumanMessage(content=state["input"])]

    planner = get_planner()
    response = planner.invoke({"messages": inputs})
    
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Plan, include_raw=True)
        info = structured_llm.invoke(response.content)
        
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            return {
                "input": state["input"],
                "plan": parsed_info.steps
            }
    
    print('parsing_error: ', info['parsing_error'])
    return {"plan": []}          
```


Plan을 실행하기 위한 execution 노드는 아래와 같습니다.

```python
def execute_node(state: State, config):
    print("###### execute ######")
    plan = state["plan"]
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    
    task = plan[0]
    task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    request = HumanMessage(content=task_formatted)
    
    chat = get_chat()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | chat
    
    response = chain.invoke({"messages": [request]})
    result = response.content
    output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
    
    return {
        "input": state["input"],
        "plan": state["plan"],
        "info": [output],
        "past_steps": [task],
    }
```

아래와 같이 replan 노드를 정의합니다.

```python
class Response(BaseModel):
    """Response to user."""
    response: str
    
class Act(BaseModel):
    """Action to perform as a json format"""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
    
def get_replanner():
    replanner_prompt = ChatPromptTemplate.from_template(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer."
        "Do not add any superfluous steps."
        "The result of the final step should be the final answer."
        "Make sure that each step has all the information needed - do not skip steps."

        "Your objective was this:"
        "{input}"

        "Your original plan was this:"
        "{plan}"

        "You have currently done the follow steps:"
        "{past_steps}"

        "Update your plan accordingly."
        "If no more steps are needed and you can return to the user, then respond with that."
        "Otherwise, fill out the plan."
        "Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."
    )
    
    chat = get_chat()
    replanner = replanner_prompt | chat
    
    return replanner

def replan_node(state: State, config):
    print('#### replan ####')
    
    update_state_message("replanning...", config)
    
    replanner = get_replanner()
    output = replanner.invoke(state)
    print('replanner output: ', output.content)
    
    result = None
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Act, include_raw=True)    
        info = structured_llm.invoke(output.content)
        print(f'attempt: {attempt}, info: {info}')
        
        if not info['parsed'] == None:
            result = info['parsed']
            print('act output: ', result)
            break
                
    if result == None:
        return {"response": "답을 찾지 못하였습니다. 다시 해주세요."}
    else:
        if isinstance(result.action, Response):
            return {
                "response": result.action.response,
                "info": [result.action.response]
            }
        else:
            return {"plan": result.action.steps}
```

반복 동작을 위해 should_end() 을 정의합니다.

```python
def should_end(state: State) -> Literal["continue", "end"]:
    print('#### should_end ####')
    print('state: ', state)
    if "response" in state and state["response"]:
        return "end"
    else:
        return "continue"    
```

최종 답변을 생성합니다. 

```python
def final_answer(state: State) -> str:
    print('#### final_answer ####')
    
    context = state['info']
    query = state['input']
    
    if isKorean(query)==True:
        system = (
            "Assistant의 이름은 서연이고, 질문에 대해 친절하게 답변하는 도우미입니다."
            "다음의 <context> tag안의 참고자료를 이용하여 질문에 대한 답변합니다."
            "답변의 이유를 풀어서 명확하게 설명합니다."
            "결과는 <result> tag를 붙여주세요."
            
            "<context>"
            "{context}"
            "</context>"
        )
    else: 
        system = (
            "Here is pieces of context, contained in <context> tags."
            "Provide a concise answer to the question at the end."
            "Explains clearly the reason for the answer."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "Put it in <result> tags."
            
            "<context>"
            "{context}"
            "</context>"
        )

    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                
    chat = get_chat()
    chain = prompt | chat
    
    try: 
        response = chain.invoke(
            {
                "context": context,
                "input": query,
            }
        )
        result = response.content
        output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
        print('output: ', output)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)      
        
    return {"answer": output}  
```



## 실행 결과

아래와 같이 CoT 문제를 쉽게 해결할 수 있습니다.

"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까? "

![image](https://github.com/user-attachments/assets/9c7d9765-6148-4666-9318-677d7dd568e4)


"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"

![image](https://github.com/user-attachments/assets/c8d9807e-15a4-424c-999f-4b92e8799a76)


"저는 초등학교 4학년이에요. 의사가 되려면 어떻게 해야하나요?"로 질문합니다.

![image](https://github.com/user-attachments/assets/aaf828cc-6618-4978-8473-03892b6a9ef1)



아래와 같이 "넌센스 큐즈니 너무 고민하지 말고 대답해봐. 아빼 개구리는 깨굴깨굴 울고 엄마 개구리는 가굴가굴 울고 있는데, 아기 개구리는 어떻게 울까?"라고 질문을 했을때에 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/d018433c-a4ff-413d-88f5-159c50ca5f23)

(45+23x2+345)/2 로 수학문제를 내고 결과를 확인합니다. 

![image](https://github.com/user-attachments/assets/0bdae5d2-0a9a-4221-b253-b29bd0ebc260)

"토끼와 거북이중에 누가 더 빠르지?"라고 질문하고 결과를 확인합니다. 

![image](https://github.com/user-attachments/assets/d88255fc-f194-4812-b049-80f700f70b7d)

이때의 LangSmith 로그는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/a949b540-9e8c-4de9-a350-fdb5bc6cefc9)




"닭이 먼저인지 달걀이 먼저인지 알려줘."로 질문합니다.

![image](https://github.com/user-attachments/assets/7b117a51-47ad-4eaf-b2ae-67fd1fe486f8)

이때의 동작을 LangSmith로 확인하면 아래와 같습니다.

![image](https://github.com/user-attachments/assets/35b5e9e1-9f3c-4b5e-8f8c-8ae2ed08d711)

이때, plan draft는 아래와 같습니다.

```java
  "plan": [
    "1. 생명의 기원에 대한 과학적 증거를 검토합니다. 화석 기록과 진화론적 증거를 살펴봅니다.",
    "2. 현재 지구상에 존재하는 조류의 기원과 진화 과정을 추적합니다.",
    "3. 달걀을 낳는 조류의 기원을 파악합니다. 달걀을 낳기 전 조상이 있었는지 확인합니다.",
    "4. 달걀을 낳는 조류의 조상이 달걀을 낳지 않았다면, 그 조상이 먼저 존재했다는 것을 의미합니다. 따라서 닭이 먼저였습니다.",
    "5. 반대로 달걀을 낳는 조류의 조상도 달걀을 낳았다면, 달걀이 먼저 존재했다는 의미입니다.",
    "6. 과학적 증거를 종합하여 닭과 달걀 중 어느 것이 먼저 존재했는지 결론을 내립니다."
  ]
```

### How many r's in Strawberry?

Plan and execute 패턴으로 "How many r's are there in "strawberry"?" 문제를 풀어 보았습니다. 이 문제는 OpenAI community에 나왔던 유명한 문제입니다. Tokenizer의 특징으로 two로 답을 하지만, [Incorrect count of ‘r’ characters in the word “strawberry“](https://community.openai.com/t/incorrect-count-of-r-characters-in-the-word-strawberry/829618)에서는 "Veryfy with code"로 문제를 해결하기도 했습니다.

![image](https://github.com/user-attachments/assets/d015742c-e44d-4033-ad11-3d7f423a1d2d)

로그로 확인해보면 전체 15번의 plan을 생성한 후에 최종 값인 '3'을 얻을 수 있었습니다. 

#### 실행 1 (Plan Draft의 생성)

```python
Step 1: Write down the word "strawberry".
Step 2: Scan the word from left to right, counting the number of times the letter 'r' appears.
Step 3: Keep a running count of the number of 'r's found.
Step 4: After scanning the entire word, the final count is the number of 'r's in the word.

executor output:
strawberry
```

#### 실행 2 (Replan후 실행 결과)

```python
'1. Write a function that takes a word as input', 
'2. Initialize a counter variable to 0', 
'3. Loop through each character in the word',
"4. If the character is 'r', increment the counter", 
'5. After the loop ends, return the final counter value', 
"6. Call the function with the user's word and print the result"

executor output:
def count_r(word):
```

#### 실행 3

```python
'2. Initialize a counter variable to 0',
'3. Loop through each character in the word',
"4. If the character is 'r', increment the counter",
'5. After the loop ends, return the final counter value',
'6. Call the function with the word "strawberry" and print the result'

executor output:  
counter = 0
```

#### 실행 4

```python
'1. Define a function that takes a word as input',
'2. Initialize a counter to 0',
"3. Loop through each character in the word 'strawberry'",
"4. If the character is 'r', increment the counter",
'5. After the loop ends, return the final counter value',
"6. Call the function with the word 'strawberry' and print the result"

executor output:  
def count_r(word):
```

#### 실행 5

```python
'3. Initialize a counter variable to 0',
"4. Loop through each character in the word 'strawberry'",
"5. If the character is 'r', increment the counter",
'6. After the loop ends, return the final counter value',
'7. Define a function that takes a word as input and follows the above steps',
"8. Call the function with the word 'strawberry' and print the result"

executor output:  
counter = 0
```

#### 실행 6

```python
'1. Define a function that takes a word as input',
'2. Initialize a counter to 0',
'3. Loop through each character in the input word',
"4. If the character is 'r', increment the counter",
'5. After the loop ends, return the final counter value',
"6. Call the function with the word 'strawberry'",
'7. Print the result'

executor output:  
def count_r(word):
```

#### 실행 7

```python
'3. Loop through each character in the input word',
"4. If the character is 'r', increment the counter",
'5. After the loop ends, return the final counter value',
"6. Call the function with the word 'strawberry'",
'7. Print the result']

executor output: 
word = "strawberry"
for char in word:
# 각 문자에 대한 작업을 여기에 추가
print(char)
```

#### 실행 8

```python
'4. Create a function that takes a string as input',
'5. Initialize a counter to 0',
'6. Loop through each character in the string',
"7. If the character is 'r', increment the counter",
'8. After the loop ends, return the final counter value',
"9. Call the function with the word 'strawberry'",
'10. Print the result'

executor output: 
def count_r(string):
```

#### 실행 9 

```python
'5. Initialize a counter to 0',
'6. Loop through each character in the string',
"7. If the character is 'r', increment the counter",
'8. After the loop ends, return the final counter value',
"9. Call the function with the word 'strawberry'",
'10. Print the result']

executor output: 
counter = 0
```

#### 실행 10

```python
'6. Loop through each character in the string',
"7. If the character is 'r', increment the counter",
'8. After the loop ends, return the final counter value',
"9. Call the function with the word 'strawberry'",
'10. Print the result'

executor output: 
for char in 'strawberry':
print(char)
Output:
s
t
r
a
w
b
e
r
r
y
```

#### 실행 11 

```python
"1. Iterate through each character in the word 'strawberry'",
'2. Initialize a counter to 0',
"3. If the current character is 'r', increment the counter",
'4. After the loop ends, return the final counter value',
"5. Call the function with the word 'strawberry'",
'6. Print the result'

executor output: 
1. Iterate through each character in the word 'strawberry'
executor output:  
for char in 'strawberry':
# 각 문자에 대한 작업을 여기에 작성합니다.
print(char)
# 출력:
# s
# t
# r
# a
# w
# b
# e
# r
# r
# y
```

#### 실행 12

```python
'2. Initialize a counter to 0',
"3. If the current character is 'r', increment the counter",
'4. After the loop ends, return the final counter value',
"5. Call the function with the word 'strawberry'",
'6. Print the result'

executor output: 
counter = 0
```

#### 실행 13 

```python
'1. Initialize a counter to 0',
"2. Loop through each character in the word 'strawberry'",
"3. If the current character is 'r', increment the counter",
'4. After the loop ends, return the final counter value',
"5. Call the function with the word 'strawberry'",
'6. Print the result']

executor output: 
counter = 0
```

#### 실행 14 

```python
'1. Initialize a counter to 0',
"2. Loop through each character in the string 'strawberry'",
"3. If the current character is 'r', increment the counter",
'4. After the loop ends, return the final counter value'

executor output: 
counter = 0
```

#### 실행 15

```python
"1. If the current character is 'r', increment the counter",
'2. After the loop ends, return the final counter value'

executor output: 
1. 문자열의 각 문자를 하나씩 확인합니다.
2. 현재 문자가 'r'이면 카운터를 1 증가시킵니다.
3. 문자열의 모든 문자를 확인할 때까지 1, 2번 과정을 반복합니다.
```

