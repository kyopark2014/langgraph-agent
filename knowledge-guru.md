# Knowledge Guru

Knowledge Guru의 activity diagram은 아래와 같습니다. 

1) 요청(task)에 따라 tools를 이용하여 draft 답변을 구합니다.
2) draft로 부터 검색에 사용할 keyword와 draft에 대한 reflection을 추출합니다. 추출된 keyword로 search를 포함한 tools에 요청하여 관련된 정보를 얻습니다. 
3) 관련된 정보(content)와 reflection을 이용하여 답변을 업데이트(revise) 합니다.
4) 2번 3번의 과정을 max_revision 만큼 반복합니다.

![image](https://github.com/user-attachments/assets/1bd71fbd-e4c0-428e-a0d7-5f448dc99640)

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
    messages: Annotated[list, add_messages]
    reflection: list
    search_queries: list
```

검색을 수행하여 초안(draft)를 생성합니다. 

```python
def generate(state: State):    
    draft = enhanced_search(state['task'])  
    print('draft: ', draft)
        
    return {
        "task": state['task'],
        "messages": [AIMessage(content=draft)]
    }
```


Reflection과 search_queries를 구하기 위한 Research 클래스를 정의합니다.

```python
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    advisable: str = Field(description="Critique of what is helpful for better answer")
    superfluous: str = Field(description="Critique of what is superfluous")

class Research(BaseModel):
    """Provide reflection and then follow up with search queries to improve the answer."""

    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
```

아래와 같이 reflection, search_queries를 추출합니다. 

```python    
def reflect(state: State):
    print('draft: ', state["messages"][-1].content)
    
    reflection = []
    search_queries = []
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Research, include_raw=True)
            
        info = structured_llm.invoke(state["messages"][-1].content)
        print(f'attempt: {attempt}, info: {info}')
                
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            # print('reflection: ', parsed_info.reflection)                
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            search_queries = parsed_info.search_queries
                
            print('reflection: ', parsed_info.reflection)            
            print('search_queries: ', search_queries)                
            break
        
    return {
        "task": state["task"],
        "messages": state["messages"],
        "reflection": reflection,
        "search_queries": search_queries
    }  
```

답변을 향상시킵니다.

```python        
def revise_answer(state: State):   
    system = """Revise your previous answer using the new information. 
You should use the previous critique to add important information to your answer. provide the final answer with <result> tag. 
<critique>
{reflection}
</critique>

<information>
{content}
</information>"""
                    
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
            
    chat = get_chat()
    reflect = reflection_prompt | chat
            
    messages = [HumanMessage(content=state["task"])] + state["messages"]
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    print('translated: ', translated)
        
    content = []        
    if useEnhancedSearch:
        for q in state["search_queries"]:
            response = enhanced_search(q)     
            print(f'q: {q}, response: {response}')
            content.append(response)                   
    else:
        search = TavilySearchResults(k=2)
        for q in state["search_queries"]:
            response = search.invoke(q)     
            for r in response:
                content.append(r['content'])     
        
    res = reflect.invoke(
        {
            "messages": translated,
            "reflection": state["reflection"],
            "content": content
        }
    )    
                                
    response = HumanMessage(content=res.content[res.content.find('<result>')+8:len(res.content)-9])
    print('response: ', response)
                
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "task": state["task"],
        "messages": [response], 
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

    workflow.add_node("generate", generate)
    workflow.add_node("reflect", reflect)
    workflow.add_node("revise_answer", revise_answer)

    workflow.set_entry_point("generate")

    workflow.add_conditional_edges(
        "revise_answer", 
        should_continue, 
        {
            "end": END, 
            "contine": "reflect"}
    )

    workflow.add_edge("generate", "reflect")
    workflow.add_edge("reflect", "revise_answer")
        
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
        
readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
return value["messages"][-1].content
```


### 실행결과

아래는 "생성형 AI를 위한 데이터의 수집 및 분석 방법에 대해 설명해줘."에 대한 결과입니다. 일반적인 Q&A 경우보다 훨싼 상세한 결과를 얻을 수 있습니다. 

![image](https://github.com/user-attachments/assets/fe32f21c-0f3f-481c-ba4c-cc9728677996)

아래와 같이 전체 251초가 소요되었습니다.

![image](https://github.com/user-attachments/assets/b5d31304-5727-4208-b840-28cb3bc548e5)




아래에서는 "서울에서 관광하기 좋은곳과 가까운 맛집들을 추천해줘."로 검색하였을때의 결과입니다. 일반적인 검색일 경우에는 Tavily를 통해 검색하여 얻은 결과를 활용합니다. 

![noname](https://github.com/user-attachments/assets/a8e557dd-9b70-4a4e-a7d5-3e5eda544339)

이때 전체 94초가 소요 되었습니다.

![image](https://github.com/user-attachments/assets/30b2784a-7ba2-46d2-ac1f-3646e1b649bb)



아래는 "생성형 AI를 위해서는 데이터가 중요한데요. 데이터를 어떻게 가공할수 있는지 알려주세요."와 같이 검색하여 OpenSearch를 중심으로 검색을 수행하였습니다.

![noname](https://github.com/user-attachments/assets/ebd7fa92-ee56-4380-890b-6b3ec40ee1d6)

이때의 실행히간은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/b084379a-54cd-4c6d-a707-b589a25e9aba)


