# Knowledge Guru

Knowledge Guru의 activity diagram은 아래와 같습니다. 

1) 요청(task)에 따라 tools를 이용하여 draft 답변을 구합니다.
2) draft로 부터 검색에 사용할 keyword를 추출하고, 각 keyword로 search를 포함한 tools에 요청하여 관련된 정보를 얻습니다. 
3) 관련된 정보(content)를 이용하여 답변을 업데이트(revise) 합니다.
4) 2번 3번의 과정을 max_revision 만큼 반복합니다.

![image](https://github.com/user-attachments/assets/009a20ec-0993-4be3-a8ac-a5f9b7f04a68)


## 구현

검색에서 사용하는 함수는 OpenSearch와 Tavily를 모두 활용합니다. 

```python
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

def init_enhanced_search():
    chat = get_chat() 

    model = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
            
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State):
        question = state["messages"]
        print('question: ', question)
            
        if isKorean(question[0].content)==True:
            system = (
                "Assistant는 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
                "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
            )
        else: 
            system = (            
                "You are a researcher charged with providing information that can be used when making answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will be acting as a thoughtful advisor."
                "Put it in <result> tags."
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

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
            
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
    
    return buildChatAgent()

app_enhanced_search = init_enhanced_search()

def enhanced_search(query):
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    print('result: ', result)
            
    message = result["messages"][-1]
    print('enhanced_search: ', message)

    return message.content[message.content.find('<result>')+8:len(message.content)-9]
```

Graph를 위한 State를 정의합니다. 

```python
class State(TypedDict):
    task: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
```

검색을 수행하여 초안(draft)를 생성합니다. 

```python
def generation(state: State):    
    draft = enhanced_search(state['task'])  
    print('draft: ', draft)
        
    return {"draft": draft}
```

draft에서 적절한 검색어를 추출합니다. 이후 추출된 keyword를 이용하여 재검색을 하고, 이때 얻어진 정보로 content를 생성합니다. 

```python    
class Queries(BaseModel):
    """List of quries as a json format"""
    queries: str = Field(description="queries relevant to the question'")
    
def reflection(state: State):
    system = """You are a researcher charged with providing information that can \
be used when writing the following answer. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max. All queries should be words or string without numbers"""
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{answer}"),
        ]
    )
            
    chat = get_chat()
    chain = prompt | chat
    response = chain.invoke({"answer": state['draft']})
    print('response: ', response.content)
        
    chat = get_chat()
    structured_llm = chat.with_structured_output(Queries, include_raw=True)
    info = structured_llm.invoke(response.content)
    print('info: ', info)
        
    content = []
    if not info['parsed'] == None:
        queries = info['parsed']
        print('queries: ', queries.queries)
        
        if useEnhancedSearch:
            for q in json.loads(queries.queries):
                response = enhanced_search(q)     
                print(f'q: {q}, response: {response}')
                content.append(response)                   
        else:
            search = TavilySearchResults(k=2)
            for q in json.loads(queries.queries):
                response = search.invoke(q)     
                for r in response:
                    content.append(r['content'])    
    return {
        "content": content,
        "draft": state["draft"]
    }    
```

검색을 통해 얻어온 content로 답변을 향상시킵니다.

```python        
def revise_answer(state: State):   
    system = """Revise your previous answer using the new information as bellow. Then prvide the final answer with <result> tag.
        
    <information>
    {content}
    </information
    """
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "<answer>{draft}</answer>"),
        ]
    )
                
    chat = get_chat()
    chain = prompt | chat

    response = chain.invoke({
        "content": state['content'],
        "draft": state['draft']
    })
    print('revise_answer: ', response.content)
                
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "draft": response, 
        "revision_number": revision_number + 1
    }
```

max_revisions만큼 반복합니다. 

```python
MAX_REVISIONS = 1
def should_continue(state: State, config):
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
    print("max_revisions: ", max_revisions)
            
    if state["revision_number"] > max_revisions:
        return "end"
    return "contine"
```

Workflow를 Graph로 정의합니다. 

```python
def buildKnowledgeGuru():    
    workflow = StateGraph(State)

    workflow.add_node("generation", generation)
    workflow.add_node("reflection", reflection)
    workflow.add_node("revise_answer", revise_answer)

    workflow.set_entry_point("generation")

    workflow.add_conditional_edges(
        "revise_answer", 
        should_continue, 
        {
            "end": END, 
            "contine": "reflection"
        }
    )

    workflow.add_edge("generation", "reflection")
    workflow.add_edge("reflection", "revise_answer")
        
    app = workflow.compile()
        
    return app
    
app = buildKnowledgeGuru()
```

아래와 같이 실행하여 결과를 스트림으로 표시합니다.

```python        
isTyping(connectionId, requestId)    
inputs = {"task": query}
config = {
    "recursion_limit": 50,
    "max_revisions": MAX_REVISIONS
}
    
for output in app.stream(inputs, config):   
    for key, value in output.items():
        print(f"Finished: {key}")
        #print("value: ", value)
            
print('value: ', value)
        
readStreamMsg(connectionId, requestId, value["draft"].content)
    
return value["draft"].content[value["draft"].content.find('<result>')+8:len(value["draft"].content)-9]
```


### 실행결과

아래는 "생성형 AI를 위한 데이터의 수집 및 분석 방법에 대해 설명해줘."에 대한 결과입니다. 일반적인 Q&A 경우보다 훨싼 상세한 결과를 얻을 수 있습니다. 

![image](https://github.com/user-attachments/assets/fe32f21c-0f3f-481c-ba4c-cc9728677996)


아래와 같이 전체 251초가 소요되었습니다.

![image](https://github.com/user-attachments/assets/b5d31304-5727-4208-b840-28cb3bc548e5)
