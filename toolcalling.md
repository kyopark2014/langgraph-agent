# Tool Calling Agent

여기에서는 Tool Calling Agent에 대해 설명합니다.

## Tool Calling 의 구현

```python
def run_agent_tool_calling(connectionId, requestId, chat, query):
    # toolList = "get_current_time, get_product_list, get_weather_info"
    toolList = ", ".join((t.name for t in tools))
    
    system = f"You are a helpful assistant. Make sure to use the {toolList} tools for information."
    # system = f"You are a helpful assistant. Make sure to use the get_current_time tool for information."
    #system = f"다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변에 필요한 정보는 다움의 tools를 이용해 수집하세요. Tools: {toolList}"
            
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            # ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
        ]
    )
    print('prompt: ', prompt)
    
     # create agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # run agent
    isTyping(connectionId, requestId)
    response = agent_executor.invoke({"input": query})
    print('response: ', response)

    # streaming        
    readStreamMsgForAgent(connectionId, requestId, response['output'])

    msg = response['output']    
    output = removeFunctionXML(msg)
    # print('output: ', output)
            
    return output
```

## Tool Calling Chat의 구현

```python
def run_agent_tool_calling_chat(connectionId, requestId, chat, query):
    toolList = "get_current_time, get_product_list, get_weather_info"
    # system = f"You are a helpful assistant. Make sure to use the {toolList} tools for information."
    system = f"다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변에 필요한 정보는 다움의 tools를 이용해 수집하세요. Tools: {toolList}"
            
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
     # create agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # run agent
    history = memory_chain.load_memory_variables({})["chat_history"]
    
    isTyping(connectionId, requestId)
    response = agent_executor.invoke({
        "input": query,
        "chat_history": history
    })
    print('response: ', response)

    # streaming        
    readStreamMsgForAgent(connectionId, requestId, response['output'])

    msg = response['output']    
    output = removeFunctionXML(msg)
    # print('output: ', output)
            
    return output
```
