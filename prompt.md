# ReAct Prompt

여기서는 다양한 Prompt 예제에 대해 설명합니다.

## ReAct 기본 

ReAct 기본 포맷으로 "hwchase17/react"을 참조합니다. 

```python
PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
```

이를 langchain hub를 이용하려면 아래처럼 hub를 설치합니다. 

```text
pip install langchainhub
```

실제 사용할때는 아래 코드를 이용합니다.

```python
from langchain import hub
prompt_template = hub.pull("hwchase17/react")
```

### 한글화 Prompt

기본 ReAct를 한글화한 Prompt는 아래와 같습니다.

```python
PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

Use the following format:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action으로서 [{tool_names}]중 하나를 선택합니다.
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 3번 반복 할 수 있습니다. 반복이 끝날때까지 정답을 찾지 못하면 마지막 result로 답변합니다.)
... (반복이 끝날때까지 적절한 답변을 얻지 못하면, 마지막 결과를 Final Answer를 전달합니다. )
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
```

## 참고 예제

[Agent 예제](https://github.com/langchain-ai/langchain/issues/12944)에는 아래와 같은 예제가 있습니다.

```python
PREFIX = """
Respond to the human as helpfully and accurately as possible.
Please do not repeat yourself. Start with the following format:
'''
Question: the input question you must answer
Thought: Do I need to use a tool? 
'''
"""

FORMAT_INSTRUCTIONS = """
Consider your actions before and after.

If your answer to 'Thought: Do I need to use a tool?' is\
'No', continue with the following format:

'''
Thought: Do I need to use a tool? No.
Action: return AgentFinish.
'''

If your answer to 'Thought: Do I need to use a tool?' is\
'Yes', continue with the following format until the answer becomes 'No',
at which point you should use the aforementioned format:

Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

"""

SUFFIX = '''
Begin!
Remember, you do not always need to use tools. Do not provide information the user did not ask for.

Question: {input}
Thought: {agent_scratchpad}
'''
```

```python
template = """ 
You are a great AI-Assistant that has access to additional tools in order to answer the following questions as best you can. Always answer in the same language as the user question. You have access to the following tools:

{tools}

To use a tool, please use the following format:

'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
'''

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
```

```python
PREFIX = """
You are a highly intelligent assistant capable of understanding and executing specific actions based on user requests. Your goal is to assist the user as efficiently and accurately as possible without deviating from their instructions.
"""

FORMAT_INSTRUCTIONS = """
Please follow these instructions carefully:

1. If you need to perform an action to answer the user's question, use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: [Specify the action]
Action Input: [Provide the necessary input for the action]
Observation: [Describe the outcome of the action]
'''

2. If you can answer the user's question without performing any additional actions, use the following format:
'''
Thought: Do I need to use a tool? No
Final Answer: [Provide your answer here]
'''

Your responses should be concise and directly address the user's query. Avoid generating new questions or unnecessary information.
"""

SUFFIX = """
End of instructions. Please proceed with answering the user's question following the guidelines provided above.
"""

# Example of how to use this template in a LangChain agent setup
template = PREFIX + FORMAT_INSTRUCTIONS + SUFFIX
```
