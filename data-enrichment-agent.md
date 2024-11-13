# Data Enrichment Agent 

LangGraph의 [LangGraph Data Enrichment Template](https://github.com/langchain-ai/data-enrichment)와 [Data Enrichment Agent](https://www.youtube.com/watch?v=mNxAM1ETBvs&t=10s)을 참조합니다.

<img src="https://github.com/user-attachments/assets/232f0b86-7663-4355-bb30-fcefd65b6876" width="700">

이때의 activity diagram은 아래와 같습니다.

![noname](https://github.com/user-attachments/assets/91415efe-ad2e-478f-9ae1-49bd7f5f4eed)


## 상세 구현

아래와 같이 state를 정의합니다.

```python
class State(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]=field(default_factory=list)
    loop_step: Annotated[int,operator.add]=field(default=0)
    topic: str
    extraction_schema: dict[str, Any]
    info: Optional[dict[str, Any]] = field(default=None)
```

사용한 툴에 대해 정의합니다. 아래는 검색을 위해 tavily를 사용합니다.

```python
def search(        
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """        
    print(f"###### [tool] search: {query} ######")
    
    wrapped = TavilySearchResults(max_results=max_search_results)
    result = wrapped.invoke({"query": query})
    
    output = cast(list[dict[str, Any]], result)    
    global reference_docs
    for re in result:  # reference
        doc = Document(
            page_content=re["content"],
            metadata={
                'name': 'WWW',
                'url': re["url"],
                'from': 'tavily'
            }
        )
        reference_docs.append(doc)
    
    return output
```

웹에서 텍스트를 가져오기 위한 툴입니다.

```python
def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:        
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    print(f"###### [tool] scrape_website: {url} ######")
    
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.get_text()
        
        _INFO_PROMPT = (
            "You are doing web research on behalf of a user. You are trying to find out this information:"

            "<info>"
            "{info}"
            "</info>"

            "You just scraped the following website: {url}"

            "Based on the website content below, jot down some notes about the website."

            "<Website content>"
            "{content}"
            "</Website content>"
        )
        
        p = _INFO_PROMPT.format(
            info=json.dumps(state['extraction_schema'], indent=2),
            url=url,
            content=content[:40_000],
        )            
        chat = get_chat()
        result = chat.invoke(p)
        content = str(result.content)
    else:
        content = "Failed to retrieve the webpage. Status code: " + str(response.status_code)
        print(content)
    
    return content
```

Agent 노드를 정의합니다.

```python
def agent_node(state: State) -> Dict[str, Any]:
    print("###### agent_node ######")
    
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state["extraction_schema"],
    }
    
    if isKorean(state["topic"])==True:
        MAIN_PROMPT = (
            "웹 검색을 통해 <info> tag의 schema에 대한 정보를 찾아야 합니다."
            "<info>"
            "{info}"
            "</info>"

            "다음 도구를 사용할 수 있습니다:"
            "- `Search`: call a search tool and get back some results"
            "- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above."
            "- `Info`: call this when you are done and have gathered all the relevant info:"

            "다음은 네가 연구 중인 topic에 대한 정보입니다:"

            "Topic: {topic}"
        )
    else:
        MAIN_PROMPT = (
            "You are doing web research on behalf of a user. You are trying to figure out this information:"
            "<info>"
            "{info}"
            "</info>"

            "You have access to the following tools:"
            "- `Search`: call a search tool and get back some results"
            "- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above."
            "- `Info`: call this when you are done and have gathered all the relevant info:"

            "Here is the information you have about the topic you are researching:"

            "Topic: {topic}"
        )

    p = MAIN_PROMPT.format(
        info=json.dumps(state["extraction_schema"], indent=2), 
        topic=state["topic"]
    )

    messages = [HumanMessage(content=p)] + state["messages"]

    chat = get_chat() 
    tools = [scrape_website, search, info_tool]
    model = chat.bind_tools(tools, tool_choice="any")
    result = model.invoke(messages)    
    response = cast(AIMessage, result)

    info = None
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                print('info: ', info)                    
                break
            
    if info is not None:  # The agent is submitting their answer
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]

    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    
    return {
        "messages": response_messages,
        "info": info,
        "loop_step": 1,
    }
```

Reflection 노드를 정의합니다.

```python
class Reason(BaseModel):
    values: List[str] = Field(
        description="a list of reasons"
    )

class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: Reason = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory."
        " This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
        default=None,
    )

def reflect_node(state: State) -> Dict[str, Any]:
    print("###### reflect_node ######")
    
    last_message = state["messages"][-1]    
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect_node.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
        
    presumed_info = state["info"]
    print('presumed_info: ', presumed_info)
    
    topic = state["topic"]
    # print('topic: ', topic)
    if isKorean(topic)==True:
        system = (
            "아래 정보로 info tool을 호출하려고 합니다."
            "이것이 좋습니까? 그 이유도 설명해 주세요."
            "당신은 특정 URL을 살펴보거나 더 많은 검색을 하도록 어시스턴트에게 요청할 수 있습니다."
            "만약 이것이 좋지 않다고 생각한다면, 어떻게 개선해야할 지 구체적으로 제사합니다."
            "최종 답변에 <result> tag를 붙여주세요."
        )
    else:
        system = (                
            "I am thinking of calling the info tool with the info below."
            "Is this good? Give your reasoning as well."
            "You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches."
            "If you don't think it is good, you should be very specific about what could be improved."
            "Put it in <result> tags."
        )
        
    human = "{presumed_info}"
    
    chat = get_chat()
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    
    response = chain.invoke({
        "presumed_info": json.dumps(presumed_info)
    })
    result = response.content
    output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
    
    response = ""
    reason = []
    is_satisfactory = False
    improvement_instructions = ""
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(InfoIsSatisfactory, include_raw=True)
        info = structured_llm.invoke(output)
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            
            reason = parsed_info.reason.values
            is_satisfactory = parsed_info.is_satisfactory
            improvement_instructions = parsed_info.improvement_instructions                
            
            response = cast(InfoIsSatisfactory, info)
            break                
    
    if is_satisfactory and presumed_info:
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(reason),
                    name="Info",
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Unsatisfactory response:\n{improvement_instructions}",
                    name="Info",
                    status="error",
                )
            ]
        }
```

Conditional edge들을 정의합니다.

```python
def route_after_agent(state: State) -> Literal["reflect", "tools", "agent"]:
    print("###### route_after_agent ######")
    
    last_message = state["messages"][-1]
    print('last_message: ', last_message)
    
    next = ""
    if not isinstance(last_message, AIMessage):
        next = "agent"
    else:
        if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
            next = "reflect"
        else:
            print('tool_calls: ', last_message.tool_calls[0]["name"])
            next = "tools"
    print('next: ', next)
    
    return next

def route_after_checker(state: State) -> Literal["end", "continue"]:
    print("###### route_after_checker ######")
    
    last_message = state["messages"][-1]
    print('last_message: ', last_message)
    
    if state["loop_step"] < max_loops:
        if not state["info"]:
            return "continue"
        
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        
        if last_message.status == "error":
            return "continue"  # Research deemed unsatisfactory
        
        return "end"   # It's great!
    
    else:
        return "end"
```

Workflow를 정의합니다.

```python
def build_data_enrichment_agent():
    workflow = StateGraph(State, output=OutputState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent", 
        route_after_agent,
        {
            "agent": "agent",
            "reflect": "reflect",
            "tools": "tools"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "reflect", 
        route_after_checker,
        {
            "continue": "agent",
            "end": END
        }
    )

    return workflow.compile()        
```

여기서 사용한 schema는 아래와 같습니다.

```java
schema = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Company name"},
                    "technologies": {
                        "type": "string",
                        "description": "Brief summary of key technologies used by the company",
                    },
                    "market_share": {
                        "type": "string",
                        "description": "Overview of market share for this company",
                    },
                    "future_outlook": {
                        "type": "string",
                        "description": "Brief summary of future prospects and developments in the field for this company",
                    },
                    "key_powers": {
                        "type": "string",
                        "description": "Which of the 7 Powers (Scale Economies, Network Economies, Counter Positioning, Switching Costs, Branding, Cornered Resource, Process Power) best describe this company's competitive advantage",
                    },
                },
                "required": ["name", "technologies", "market_share", "future_outlook"],
            },
            "description": "List of companies",
        }
    },
    "required": ["companies"],
}
```

실행후에 결과를 확인합니다.

```python
inputs={
    "topic": text,
    "extraction_schema": schema
}    
config = {
    "recursion_limit": 50,
    "max_loops": max_loops,
    "requestId": requestId,
    "connectionId": connectionId
}

result = app.invoke(inputs, config)
print('result: ', result)

final = text_output(result["info"])
```







## 실행 결과

"Top 5 Chip Providers for LLM Training"로 입력후에 결과를 확인합니다. 

![noname](https://github.com/user-attachments/assets/eda83e20-2efc-42d8-92c7-2234b9fcf717)

이때의 LangSmith 로그를 보면 아래와 같습니다.

![image](https://github.com/user-attachments/assets/3e7b3871-608d-4aab-b397-8bd3a489f3ca)

"한국 주식시장에서 가장 큰 5개 회사"라고 입력후 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/1a551c84-18ab-4a55-a3b6-e5acb6004d55)

이때의 LangSmith 로그는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/013499c8-ea57-4831-8fb6-092669140d12)
