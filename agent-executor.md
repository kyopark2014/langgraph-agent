# Agentic RAG (Agent Executor)

ReAct는 LLM을 다양한 데이터 소스와 실행 가능한 프로그램과 결합할 수 있기 때문에 매우 효과적이지만, agent에 대한 유연성이나 투명성이 부족하여 구체적으로 어떻게 작동하는지 알수 없는 경우가 많습니다. LangGraph와 같이 agent를 그래프로 구성하면, 반복적인 ReAct와 유사한 루프를 생성할 수 있으며, 사용자 정의하기가 훨씬 쉬워집니다. 따라서 더 결정론적인 흐름(deterministic flow)와 계층적 의사 결정(hierarchical decision-making)을 추가할 수 있으며 [더 많은 유연성과 투명성으로 구축](https://www.pinecone.io/learn/langgraph-research-agent/)할 수 있습니다. 

<img src="https://github.com/user-attachments/assets/703b86dd-8e6a-4673-adac-048baf94d35b" width="500">

아래와 같이 LangGraph Agent를 Serverless Architecture로 구현하여 적절한 Tool을 실행할 수 있습니다. 상세한 코드는 [agent-executor.ipynb](./agent/agent-executor.ipynb)을 참조합니다. Agent로 tools를 실행하는 Excueter의 동작은 아래와 같습니다. 

## Chat Agent Executor

LangGraph에서 제공하는 기본 Sample에 한글 Prompt를 적용한 내용을 아래와 같이 반영하였습니다. 한글 Prompt를 부분적으로라도 쓰면, 적절한 한국어 답변을 생성하는데 도움이 됩니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다.

Tool을 정의하고 chat model에 bind 합니다. 또한 tool들을 실행하기 위한 tool_node을 정의합니다. 

```python
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

chatModel = get_chat() 

model = chatModel.bind_tools(tools)

tool_node = ToolNode(tools)
```

state를 위한 ChatAgentState을 정의하고 node와 conditional edge를 구성합니다.

```python
class ChatAgentState(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: ChatAgentState) -> Literal["continue", "end"]:
    messages = state["messages"]    
    # print('(should_continue) messages: ', messages)
    
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:                
        return "continue"

def call_model(state: ChatAgentState):
    question = state["messages"]
    print('question: ', question)
    
    if isKorean(question[0].content)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "You are a conversational AI designed to answer in a friendly way to a question."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "You will be acting as a thoughtful advisor."                
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
        
    response = chain.invoke(question)
    return {"messages": [response]}
```

아래와 같이 Workflow를 정의합니다.

```python
def buildChatAgent():
    workflow = StateGraph(ChatAgentState)

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

    return workflow.compile()
```

이렇게 구성된 workflow를 그려보면 아래와 같습니다.
```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/865ddc21-8492-437d-bbbc-3a9a45728a25)

아래와 같이 실행합니다.

```python
chat_app = buildChatAgent()

def run_agent_executor(connectionId, requestId, app, query):
    isTyping(connectionId, requestId)
    
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    message = ""
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        # print('event: ', event)
        
        message = event["messages"][-1]
        # print('message: ', message)

    msg = readStreamMsg(connectionId, requestId, message.content)

    return msg
```

Stream으로도 실행할 수 있습니다.

```python
from langchain_core.messages import HumanMessage

inputs = [HumanMessage(content="강남역 맛집 알려줘")]

for event in app.stream({"messages": inputs}, stream_mode="values"):    
    event["messages"][-1].pretty_print()
```

## Agent Executor From Scratch

LangGraph에서는 [ReAct 방식의 chat agent](https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/base.ipynb)를 제공하고 있습니다. 그런데 이 example에서 사용하는 ReAct Prompt 방식은 LangChain Agent와 같은 방식으로 RAG와 같이 context가 길어질 경우에 제대로 동작못하는 경우가 있었습니다. 따라서 가능한 LangGraph만으로 구성할 것을 추천 드립니다. 아래는 참고용 셈플 입니다. 

```python
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langgraph.prebuilt.tool_executor import ToolExecutor

def get_react_prompt_template(mode: str): # (hwchase17/react) https://smith.langchain.com/hub/hwchase17/react
    # Get the react prompt template
    
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should use only the tool name from [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    else: 
        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

다음의 format을 사용하세요.:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action로서 [{tool_names}]에서 tool의 name만을 가져옵니다. 
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 5번 반복 할 수 있습니다.)
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

너는 Human에게 해줄 응답이 있거나, Tool을 사용하지 않아도 되는 경우에, 다음 format을 사용하세요.:
'''
Thought: Tool을 사용해야 하나요? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
```

AgentState Class와 Node를 구성합니다. 

```python
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

prompt_template = get_react_prompt_template(mode)
agent_runnable = create_react_agent(chat, tools, prompt_template)

tools = [TavilySearchResults(max_results=1)]

def run_agent(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def execute_tools(state: AgentState):
    agent_action = state["agent_outcome"]
    
    tool_executor = ToolExecutor(tools)
    
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"
```

Workflow를 구성합니다.

```python
def buildAgent():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")
    return workflow.compile()    

app = buildAgent()
```


