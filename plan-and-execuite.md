# Planning Agent
[LangGraph: Planning Agents](https://www.youtube.com/watch?v=uRya4zRrRx4)에서는 3가지 plan-and-execution 형태의 agent를 설명하고 있습니다. 

LangGraph은 stateful하고 multi-actor 애플리케이션을 만들 수 있도록 돕는 오픈 소스 framework입니다. 이를 통해 빠르게 실행하고, 비용을 효율적으로 사용하고 성능을 향상 시킬 수 있습니다. 

## Basic Plan-and-Execute

[plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb)에서는 [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)에 대한 Agent를 정의합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a97d0764-2891-4454-8854-522ef3249e44)

이때의 activity diagram은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/a96b1848-c58e-4a5c-a741-0b541a94f5e6)


전체적인 구조는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a311023-53d7-464a-b4a0-655c558bc058)

class와 함수를 정의합니다. 

```python
"system" = """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""

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

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )    

async def plan_step(state: PlanExecute):  # planner

async def execute_step(state: PlanExecute):  # agent

async def replan_step(state: PlanExecute): # replan

def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
```

Graph, Node, Edge를 정의합니다.

```python
from langgraph.graph import StateGraph

workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
)

app = workflow.compile()
```

실행은 아래와 같습니다.

```python
config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```


## 실행 결과


<img width="867" alt="image" src="https://github.com/user-attachments/assets/e7d4ee6d-ceb9-4782-9088-178024692977">

![image](https://github.com/user-attachments/assets/05c2784a-814e-4062-b771-7760c42c2974)



"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까? "

<img width="857" alt="image" src="https://github.com/user-attachments/assets/6449bf02-a3f4-42d0-8103-b294fe60c729">


"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"

<img width="863" alt="image" src="https://github.com/user-attachments/assets/d29321fc-ddc1-484e-8c9d-c4ce34598eb0">


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
