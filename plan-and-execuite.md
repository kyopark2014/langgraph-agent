# Planning Agent
[LangGraph: Planning Agents](https://www.youtube.com/watch?v=uRya4zRrRx4)에서는 3가지 plan-and-execution 형태의 agent를 설명하고 있습니다. [plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb)에서는 [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)에 대한 Agent를 정의하고 있습니다.

LangGraph은 stateful하고 multi-actor 애플리케이션을 만들 수 있도록 돕는 오픈 소스 framework입니다. 이를 통해 빠르게 실행하고, 비용을 효율적으로 사용하고 성능을 향상 시킬 수 있습니다. 

## Plan-and-Execute

[plan-and-execute.ipynb](./agent/plan-and-execute.ipynb)와 같이 Plan-and-Execute 동작을 수행하는 Agent를 만들 수 있습니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다. 


![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a97d0764-2891-4454-8854-522ef3249e44)

이때의 activity diagram은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/a96b1848-c58e-4a5c-a741-0b541a94f5e6)


## 상세 구현

Plan을 생성하는 Prompt를 준비합니다. 

```python
def get_planner():
    system = """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""
        
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{messages}"),
        ]
    )
    
    chat = get_chat()   
    
    planner = planner_prompt | chat
    return planner

inputs = [HumanMessage(content=state["input"])]
planner = get_planner()
response = planner.invoke({"messages": inputs})
print('response.content: ', response.content)
```

아래와 같이 plan을 생성하고 추출할 수 있습니다.

```python
class Plan(BaseModel):
    """List of steps as a json format"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

chat = get_chat()
structured_llm = chat.with_structured_output(Plan, include_raw=True)
info = structured_llm.invoke(response.content)

parsed_info = info['parsed']
print('steps: ', parsed_info.steps)
```

상기 내용을 적용한 plan() 함수는 아래와 같습니다.

```python
class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

def plan(state: PlanExecuteState):
    print("###### plan ######")
    print('input: ', state["input"])
    
    inputs = [HumanMessage(content=state["input"])]

    planner = get_planner()
    response = planner.invoke({"messages": inputs})
    print('response.content: ', response.content)
    
    chat = get_chat()
    structured_llm = chat.with_structured_output(Plan, include_raw=True)
    info = structured_llm.invoke(response.content)
    print('info: ', info)
    
    if not info['parsed'] == None:
        parsed_info = info['parsed']
        # print('parsed_info: ', parsed_info)        
        print('steps: ', parsed_info.steps)
        
        return {
            "input": state["input"],
            "plan": parsed_info.steps
        }
    else:
        print('parsing_error: ', info['parsing_error'])
        
        return {"plan": []}
```

Plan을 실행하기 위한 execution() 함수는 아래와 같습니다.

```python
def execute(state: PlanExecuteState):
    print("###### execute ######")
    print('input: ', state["input"])
    plan = state["plan"]
    print('plan: ', plan) 
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    #print("plan_str: ", plan_str)
    
    task = plan[0]
    task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    print("request: ", task_formatted)     
    request = HumanMessage(content=task_formatted)
    
    chat = get_chat()
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            "다음의 Human과 Assistant의 친근한 이전 대화입니다."
            "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )
    chain = prompt | chat
    
    agent_response = chain.invoke({"messages": [request]})
    #print("agent_response: ", agent_response)
    
    print('task: ', task)
    print('executor output: ', agent_response.content)
    
    # print('plan: ', state["plan"])
    # print('past_steps: ', task)
    
    return {
        "input": state["input"],
        "plan": state["plan"],
        "past_steps": [task],
    }
```

아래와 같이 replan() 함수를 정의합니다.

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
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. \
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.""")
       
    chat = get_chat()
    replanner = replanner_prompt | chat
     
    return replanner

def replan(state: PlanExecuteState):
    print('#### replan ####')
    
    replanner = get_replanner()
    output = replanner.invoke(state)
    print('replanner output: ', output.content)
    
    chat = get_chat()
    structured_llm = chat.with_structured_output(Act, include_raw=True)    
    info = structured_llm.invoke(output.content)
    # print('info: ', info)
    
    result = info['parsed']
    print('act output: ', result)
    
    if result == None:
        return {"response": "답을 찾지 못하였습니다. 다시 해주세요."}
    else:
        if isinstance(result.action, Response):
            return {"response": result.action.response}
        else:
            return {"plan": result.action.steps}
```

반복 동작을 위해 should_end() 을 정의합니다.

```python
def should_end(state: PlanExecuteState) -> Literal["continue", "end"]:
    print('#### should_end ####')
    print('state: ', state)
    if "response" in state and state["response"]:
        return "end"
    else:
        return "continue"
```

아래와 같이 workflow를 정의합니다.

```python
def buildPlanAndExecute():
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", plan)
    workflow.add_node("executor", execute)
    workflow.add_node("replaner", replan)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replaner")
    workflow.add_conditional_edges(
        "replaner",
        should_end,
        {
            "continue": "executor",
            "end": END,
        },
    )

    return workflow.compile()

plan_and_execute_app = buildPlanAndExecute()

def run_plan_and_exeucute(connectionId, requestId, app, query):
    isTyping(connectionId, requestId)
    
    inputs = {"input": query}
    config = {"recursion_limit": 50}
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["response"])
    
    return value["response"]
```

이렇게 정의한 graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a311023-53d7-464a-b4a0-655c558bc058)



## 실행 결과

아래와 같이 CoT 문제를 쉽게 해결할 수 있습니다.

"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까? "

<img width="857" alt="image" src="https://github.com/user-attachments/assets/6449bf02-a3f4-42d0-8103-b294fe60c729">


"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"

<img width="863" alt="image" src="https://github.com/user-attachments/assets/d29321fc-ddc1-484e-8c9d-c4ce34598eb0">

조금 생각이 필요한 문제를 주더라도 답변을 찾아가는 것을 로그로 확인할 수 있습니다. 그런데 아래와 같이 중간 결과없이 최종 결과를 답변하고 있어서 개선이 필요합니다. (개선 방법 고민중)

<img width="867" alt="image" src="https://github.com/user-attachments/assets/e7d4ee6d-ceb9-4782-9088-178024692977">

![image](https://github.com/user-attachments/assets/05c2784a-814e-4062-b771-7760c42c2974)

아래와 같이 "넌센스 큐즈니 너무 고민하지 말고 대답해봐. 아빼 개구리는 깨굴깨굴 울고 엄마 개구리는 가굴가굴 울고 있는데, 아기 개구리는 어떻게 울까?"라고 질문을 했을때에 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/78bb277d-9adc-46e8-bb88-03b8dc34fb0f)

이때, LamgSmith의 로그를 보면 아래와 같습니다.

![image](https://github.com/user-attachments/assets/71b22451-7dfa-436b-9c9e-da122feaaf40)

```text
1. 아빠 개구리의 울음소리 패턴을 파악합니다: '깨굴깨굴'
2. 엄마 개구리의 울음소리 패턴을 파악합니다: '가굴가굴'
3. 아빠와 엄마의 울음소리 패턴을 비교하여 공통점과 차이점을 찾습니다
4. 공통점: 두 번 반복되는 발음 패턴
5. 차이점: 아빠는 '깨'를, 엄마는 '가'를 발음함
6. 아기 개구리의 울음소리는 아빠와 엄마의 울음소리 패턴을 따르되, 아기 개구리 나름의 발음으로 바꾼다고 가정합니다
7. 아기 개구리 나름의 발음은 '애'라고 가정합니다
8. 따라서 아기 개구리의 울음소리는 '애굴애굴'이 됩니다
```

나름 의미있는 유추이지만 아기 개구리는 울지 못합니다. 
