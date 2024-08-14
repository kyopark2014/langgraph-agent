import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import requests
import base64
import operator

from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings
from multiprocessing import Process, Pipe

from langchain.agents import tool
from bs4 import BeautifulSoup
from pytz import timezone
from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image
from opensearchpy import OpenSearch

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict, Literal, Sequence, Union

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
agentLangMode = 'kor'
projectName = os.environ.get('projectName')
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
LLM_for_chat = json.loads(os.environ.get('LLM_for_chat'))
LLM_for_multimodal= json.loads(os.environ.get('LLM_for_multimodal'))
LLM_embedding = json.loads(os.environ.get('LLM_embedding'))
selected_chat = 0
selected_multimodal = 0
selected_embedding = 0
separated_chat_history = os.environ.get('separated_chat_history')
enalbeParentDocumentRetrival = os.environ.get('enalbeParentDocumentRetrival')
enableHybridSearch = os.environ.get('enableHybridSearch')
useParallelRAG = os.environ.get('useParallelRAG', 'true')

reference_docs = []
# api key to get weather information in agent
secretsmanager = boto3.client('secretsmanager')
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    secret = json.loads(get_weather_api_secret['SecretString'])
    #print('secret: ', secret)
    weather_api_key = secret['weather_api_key']

except Exception as e:
    raise e
   
# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# api key to use Tavily Search
tavily_api_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)
    tavily_api_key = secret['tavily_api_key']
except Exception as e: 
    raise e

if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
   
# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 
MSG_LENGTH = 100

# Multi-LLM
def get_chat():
    global selected_chat
    
    profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = profile['max_tokens']
    print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == len(LLM_for_chat):
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = profile['max_tokens']
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    return chat

def get_multimodal():
    global selected_multimodal
    print('LLM_for_chat: ', LLM_for_chat)
    print('selected_multimodal: ', selected_multimodal)
    
    profile = LLM_for_multimodal[selected_multimodal]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = profile['max_tokens']
    print(f'selected_multimodal: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    multimodal = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_multimodal = selected_multimodal + 1
    if selected_multimodal == len(LLM_for_multimodal):
        selected_multimodal = 0
    
    return multimodal

def get_embedding():
    global selected_embedding
    profile = LLM_embedding[selected_embedding]
    bedrock_region =  profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id:{model_id}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  
    
    selected_embedding = selected_embedding + 1
    if selected_embedding == len(LLM_embedding):
        selected_embedding = 0
    
    return bedrock_embedding

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            chat_memory.save_context({"input": text}, {"output": msg})             

def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
    print('result: ', result)
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
            print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                    
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
    # print('lexical query result: ', json.dumps(response))
    print('relevant_documents: ', relevant_documents)
    
    return relevant_documents

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress = True,
    http_auth=(opensearch_account, opensearch_passwd),
    use_ssl = True,
    verify_certs = True,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index="idx-rag", 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('uri: ', metadata['uri'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    return source['text'], metadata['name'], metadata['uri']

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            # \n문자를 replace합니다.
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n"
    
    return answer
    
@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    # print('timestr:', timestr)
    
    return timestr

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

@tool    
def get_system_time() -> str:
    """
    retrive system time to earn the current date and time.
    return: a string of date and time
    """    
    
    function_name = "lambda-datetime-for-llm-agent"
    lambda_region = 'ap-northeast-2'
    
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {}
        print("Payload: ", payload)
            
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise
    
    payload = response['Payload']
    print('payload: ', payload)
    body = json.load(payload)['body']
    print('body: ', body)
    jsonBody = json.loads(body) 
    print('jsonBody: ', jsonBody)    
    timestr = jsonBody['timestr']
    print('timestr: ', timestr)
    
    return timestr

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
    return weather_str

@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general information by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs
    
    answer = ""
    
    if tavily_api_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(k=3)
                    
        output = search.invoke(keyword)
        print('tavily output: ', output)
        
        for result in output:
            print('result: ', result)
            if result:
                content = result.get("content")
                url = result.get("url")
                
                reference_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': 'WWW',
                            'uri': url,
                            'content': content,
                            'from': 'tavily'
                        },
                    )
                )
                
                print('langth of reference_docs: ', len(reference_docs))
            
                answer = answer + f"{content}, URL: {url}\n"
        
    return answer

@tool    
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    

    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = "idx-*", # all
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    answer = ""
    top_k = 2
    
    docs = [] 
    if enalbeParentDocumentRetrival == 'true': # parent/child chunking
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)
                        
        for i, document in enumerate(relevant_documents):
            #print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
            
            excerpt, name, uri = get_parent_content(parent_doc_id) # use pareant document
            #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, uri: {uri}, content: {excerpt}")
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = keyword,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            #print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content        
            uri = document[0].metadata['uri']            
            name = document[0].metadata['name']
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'from': 'vector'
                    },
                )
            )
    
    if enableHybridSearch == 'true':
        docs = docs + lexical_search_for_tool(keyword, top_k)
    
    print('doc length: ', len(docs))
                
    filtered_docs = grade_documents(keyword, docs)
        
    for i, doc in enumerate(filtered_docs):
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content
            
        print(f"filtered doc[{i}]: {text}, metadata:{doc.metadata}")
    
    answer = "" 
    for doc in filtered_docs:
        excerpt = doc.page_content
        uri = doc.metadata['uri']
        
        answer = answer + f"{excerpt}\n\n"
        
    return answer

def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
    # print('result: ', result)
                
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
            print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                        
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                        
                    if len(relevant_documents)>=top_k:
                        break                                
    # print('lexical query result: ', json.dumps(response))
    
    for i, doc in enumerate(relevant_documents):
        #print('doc: ', doc[0])
        #print('doc content: ', doc[0].page_content)
        
        if len(doc[0].page_content)>=100:
            text = doc[0].page_content[:100]
        else:
            text = doc[0].page_content            
        print(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")        

    return relevant_documents

def lexical_search_for_tool(query, top_k):
    # lexical search (keyword)
    min_match = 0
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f'{min_match}%',
                                "operator":  "or",
                            }
                        }
                    },
                ],
                "filter": [
                ]
            }
        }
    }

    response = os_client.search(
        body=query,
        index="idx-*", # all
    )
    # print('lexical query result: ', json.dumps(response))
        
    docs = []
    for i, document in enumerate(response['hits']['hits']):
        if i>=top_k: 
            break
                    
        excerpt = document['_source']['text']
        
        name = document['_source']['metadata']['name']
        # print('name: ', name)

        page = ""
        if "page" in document['_source']['metadata']:
            page = document['_source']['metadata']['page']
        
        uri = ""
        if "uri" in document['_source']['metadata']:
            uri = document['_source']['metadata']['uri']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'page': page,
                        'from': 'lexical'
                    },
                )
            )
    
    for i, doc in enumerate(docs):
        #print('doc: ', doc)
        #print('doc content: ', doc.page_content)
        
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        print(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")   
        
    return docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)       
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    #print("question: ", question)
    #print("doc: ", doc)    
    print_doc(doc)
    
    #print("score: ", score)
    
    grade = score.binary_score    
    print("grade: ", grade)
    
    if grade.lower() == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
    
def grade_documents_using_parallel_processing(question, documents):
    models = [        
        # claude 3.5
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude3.5",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude3.5",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "eu-central-1", # Frankfurt
            "model_type": "claude3.5",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "ap-northeast-1", # Tokyo
            "model_type": "claude3.5",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        }
    ]
    """{   # Claude 3.0
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude3",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude3",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "bedrock_region": "ca-central-1", # Canada
            "model_type": "claude3",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "bedrock_region": "eu-west-2", # London
            "model_type": "claude3",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "bedrock_region": "sa-east-1", # Sao Paulo
            "model_type": "claude3",
            "max_tokens": 4096,
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        }"""
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    selected = 0
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
                    
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected))
        processes.append(process)
        
        selected = selected + 1
        if selected == len(models):
            selected = 0

    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    #print('filtered_docs: ', filtered_docs)
    return filtered_docs

def print_doc(doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"doc: {text}, metadata:{doc.metadata}")
    
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    if useParallelRAG == 'true' or multiRegionGrade == 'enable':  # parallel processing
        print("start grading...")
        filtered_docs = grade_documents_using_parallel_processing(question, documents)

    else:
        # Score each doc    
        chat = get_chat()
        retrieval_grader = get_retrieval_grader(chat)
        for doc in documents:
            # print('doc: ', doc)
            print_doc(doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            print("score: ", score)
            
            grade = score.binary_score
            print("grade: ", grade)
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                continue
    
    global reference_docs
    reference_docs += filtered_docs
    
    # print('len(docments): ', len(filtered_docs))    
    return filtered_docs

def get_references_for_agent(docs):
    reference = "\n\nFrom\n"
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        uri = ""
        if "uri" in doc.metadata:
            uri = doc.metadata['uri']
            #print('uri: ', uri)                
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
           
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            sourceType = "OpenSearch"
        #print('sourceType: ', sourceType)        
        
        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)
        
        # for some of unusual case 
        #excerpt = excerpt.replace('"', '')        
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        print('excerpt(quotation removed): ', excerpt)
        
        if page:                
            reference = reference + f"{i+1}. {page}page in <a href={uri} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
        else:
            reference = reference + f"{i+1}. <a href={uri} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
    return reference

def retrieve(question):
    # Retrieval
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = "idx-*", # all
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    top_k = 4
    docs = []    
    if enalbeParentDocumentRetrival == 'true':
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, question, top_k)

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
            excerpt, name, uri = get_parent_content(parent_doc_id) # use pareant document
            print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, uri: {uri}, content: {excerpt}")
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = question,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content        
            uri = document[0].metadata['uri']
                            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'from': 'vector'
                    },
                )
            )    
    
    if enableHybridSearch=='true':
        # lexical search (keyword)
        min_match = 0
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": question,
                                    "minimum_should_match": f'{min_match}%',
                                    "operator":  "or",
                                }
                            }
                        },
                    ],
                    "filter": [
                    ]
                }
            }
        }

        response = os_client.search(
            body=query,
            index="idx-*", # all
        )
        # print('lexical query result: ', json.dumps(response))
        
        for i, document in enumerate(response['hits']['hits']):
            if i>=top_k: 
                break
                    
            excerpt = document['_source']['text']
            #print(f'## Document(opensearch-keyword) {i+1}: {excerpt}')

            name = document['_source']['metadata']['name']
            # print('name: ', name)

            uri = ""
            if "uri" in document['_source']['metadata']:
                uri = document['_source']['metadata']['uri']
            # print('uri: ', uri)
            
            print(f"lexical search --> doc[{i}]: {excerpt}, name:{name}, uri:{uri}\n")
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'uri': uri,
                        'from': 'lexical'
                    },
                )
            )  
    return docs

def web_search(question, documents):
    global reference_docs
    
    # Web search
    web_search_tool = TavilySearchResults(k=3)
    
    docs = web_search_tool.invoke({"query": question})
    # print('web_search: ', len(docs))
    
    for d in docs:
        print("d: ", d)
        if 'content' in d:
            web_results = "\n".join(d["content"])
            
    #web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    # print("web_results: ", web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    # for reference
    for d in docs:
        content = d.get("content")
        url = d.get("url")
                
        reference_docs.append(
            Document(
                page_content=content,
                metadata={
                    'name': 'WWW',
                    'uri': url,
                    'from': 'tavily'
                },
            )
        )
    return documents

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def get_reg_chain():
    if langMode:
        system = (
        """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

        <context>
        {context}
        </context>""")
    else: 
        system = (
        """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        <context>
        {context}
        </context>""")
        
    human = "{question}"
        
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                    
    chat = get_chat()
    rag_chain = prompt | chat
    return rag_chain

def get_rewrite():
    class RewriteQuestion(BaseModel):
        """rewrited question that is well optimized for retrieval."""

        question: str = Field(description="The new question is optimized to represent semantic intent and meaning of the user")
    
    chat = get_chat()
    structured_llm_rewriter = chat.with_structured_output(RewriteQuestion)
    
    print('langMode: ', langMode)
    
    if langMode:
        system = """당신은 질문 re-writer입니다. 사용자의 의도와 의미을 잘 표현할 수 있도록 질문을 한국어로 re-write하세요."""
    else:
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
    print('system: ', system)
        
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    question_rewriter = re_write_prompt | structured_llm_rewriter
    return question_rewriter

# define tools
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

####################### LangGraph #######################
# Chat Agent Executor
#########################################################
def run_agent_executor(connectionId, requestId, query):
    chatModel = get_chat() 

    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
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
                "다음의 Human과 Assistant의 친근한 이전 대화입니다."
                "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
            )
        else: 
            system = (            
                "Answer friendly for the newest question using the following conversation"
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

    app = buildChatAgent()
        
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

####################### LangGraph #######################
# Reflection Agent
#########################################################
def run_reflection_agent(connectionId, requestId, query):
    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    def generation(state: State):    
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 5문단의 에세이 작성을 돕는 작가이고 이름은 서연입니다"
                    "사용자의 요청에 대해 최고의 에세이를 작성하세요."
                    "사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요."
                    "최종 답변에는 완성된 에세이 전체 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chat = get_chat()
        chain = prompt | chat

        response = chain.invoke(state["messages"])
        return {"messages": [response]}

    def reflection(state: State):
        messages = state["messages"]
        
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
                    "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
                    #"특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다"
                    "각 문단의 길이는 최소 200자 이상이 되도록 관련된 예제를 충분히 포함합니다.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chat = get_chat()
        reflect = reflection_prompt | chat
        
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        print('translated: ', translated)
        
        res = reflect.invoke({"messages": translated})    
        response = HumanMessage(content=res.content)    
        return {"messages": [response]}

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]
        
        if len(messages) >= 6:   # End after 3 iterations        
            return "end"
        else:
            return "continue"

    def buildReflectionAgent():
        workflow = StateGraph(State)
        workflow.add_node("generate", generation)
        workflow.add_node("reflect", reflection)
        workflow.set_entry_point("generate")
        workflow.add_conditional_edges(
            "generate",
            should_continue,
            {
                "continue": "reflect",
                "end": END,
            },
        )

        workflow.add_edge("reflect", "generate")
        return workflow.compile()

    app = buildReflectionAgent()

    isTyping(connectionId, requestId)
    
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    msg = ""
    
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        print('event: ', event)
        
        message = event["messages"][-1]
        print('message: ', message)
        
        if len(event["messages"])>1:
            if msg == "":
                msg = message.content
            else:
                msg = f"{msg}\n\n{message.content}"

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)

    return msg

####################### LangGraph #######################
# Corrective RAG
#########################################################
langMode = False

def run_corrective_rag(connectionId, requestId, query):
    class State(TypedDict):
        question : str
        generation : str
        web_search : str
        documents : List[str]

    def retrieve(state: State):
        print("###### retrieve ######")
        question = state["question"]
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question}

    def grade_documents(state: State):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        web_search = "No"
        
        if useParallelRAG == 'true' or multiRegionGrade == 'enable':  # parallel processing
            print("start grading...")
            filtered_docs = grade_documents_using_parallel_processing(question, documents)
            
            if len(documents) != len(filtered_docs):
                web_search = "Yes"

        else:    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for doc in documents:
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                grade = score.binary_score
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    web_search = "Yes"
                    continue
        print('len(docments): ', len(filtered_docs))
        print('web_search: ', web_search)
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "web_search": web_search}

    def decide_to_generate(state: State):
        print("###### decide_to_generate ######")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "rewrite"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def generate(state: State):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        rag_chain = get_reg_chain()
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
            
        return {"documents": documents, "question": question, "generation": generation}

    def rewrite_for_crag(state: State):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def web_search(state: State):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def buildCorrectiveRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve)  
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite", rewrite_for_crag)
        workflow.add_node("websearch", web_search)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )
        workflow.add_edge("rewrite", "websearch")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    app = buildCorrectiveRAG()
    
    global langMode
    langMode = isKorean(query)
            
    isTyping(connectionId, requestId)
    
    inputs = {"question": query}
    config = {"recursion_limit": 50}
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["generation"].content)
    
    return value["generation"].content

####################### LangGraph #######################
# Self RAG
#########################################################
MAX_RETRIES = 2 # total 3

class GraphConfig(TypedDict):
    max_retries: int    
    max_count: int


def run_self_rag(connectionId, requestId, query):
    class State(TypedDict):
        question : str
        generation : str
        retries: int  # number of generation 
        count: int # number of retrieval
        documents : List[str]

    def get_hallucination_grader():
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""

            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )
        
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        
        chat = get_chat()
        structured_llm_grade_hallucination = chat.with_structured_output(GradeHallucinations)
        
        hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
        return hallucination_grader

    def get_answer_grader():
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no'"
            )
        
        chat = get_chat()
        structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
        
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = answer_prompt | structured_llm_grade_answer
        return answer_grader

    def retrieve(state: State):
        print("###### retrieve ######")
        question = state["question"]
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question}

    def generate(state: State):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        # RAG generation
        rag_chain = get_reg_chain()
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}
            
    def grade_documents(state: State):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        count = state["count"] if state.get("count") is not None else -1
        
        if useParallelRAG == 'true' or multiRegionGrade == 'enable':  # parallel processing
            print("start grading...")
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:    
            # Score each doc
            filtered_docs = []
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for doc in documents:
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                grade = score.binary_score
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    continue
        print('len(docments): ', len(filtered_docs))    
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "count": count + 1}

    def decide_to_generate(state: State, config):
        print("###### decide_to_generate ######")
        filtered_documents = state["documents"]
        
        count = state["count"] if state.get("count") is not None else -1
        max_count = config.get("configurable", {}).get("max_count", MAX_RETRIES)
        print("count: ", count)
        
        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "no document" if count < max_count else "not available"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "document"

    def rewrite(state: State):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def grade_generation(state: State, config):
        print("###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        hallucination_grader = get_hallucination_grader()
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        hallucination_grade = score.binary_score
        
        print("hallucination_grade: ", hallucination_grade)
        print("retries: ", retries)

        # Check hallucination
        answer_grader = get_answer_grader()    
        if hallucination_grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score        
            # print("answer_grade: ", answer_grade)

            if answer_grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful" 
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful" if retries < max_retries else "not available"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported" if retries < max_retries else "not available"
        
    def build():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve)  
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite", rewrite)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "no document": "rewrite",
                "document": "generate",
                "not available": "generate",
            },
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "rewrite",
                "not available": END,
            },
        )

        return workflow.compile()

    app = build()    
    
    global langMode
    langMode = isKorean(query)
    
    isTyping(connectionId, requestId)
    
    inputs = {"question": query}
    config = {"recursion_limit": 50}
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["generation"].content)
    
    return value["generation"].content

####################### LangGraph #######################
# Self-Corrective RAG
#########################################################
def run_self_corrective_rag(connectionId, requestId, query):
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        question: str
        documents: list[Document]
        candidate_answer: str
        retries: int
        web_fallback: bool

    def retrieve(state: State):
        print("###### retrieve ######")
        question = state["question"]
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question, "web_fallback": True}

    def generate(state: State):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        # RAG generation
        rag_chain = get_reg_chain()
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        global reference_docs
        reference_docs += documents
        
        return {"retries": retries + 1, "candidate_answer": generation.content}

    def rewrite(state: State):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def grade_generation(state: State, config):
        print("###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["candidate_answer"]
        web_fallback = state["web_fallback"]
        
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        if not web_fallback:
            return "finalize_response"
        
        print("---Hallucination?---")    
        hallucination_grader = get_hallucination_grader()
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        hallucination_grade = score.binary_score
            
        # Check hallucination
        if hallucination_grade == "no":
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS (Hallucination), RE-TRY---")
            return "generate" if retries < max_retries else "websearch"

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        # Check question-answering
        answer_grader = get_answer_grader()    
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.binary_score     
        print("answer_grade: ", answer_grade)
        
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "finalize_response"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION (Not Answer)---")
            return "rewrite" if retries < max_retries else "websearch"

    def web_search(state: State):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def finalize_response(state: State):
        return {"messages": [AIMessage(content=state["candidate_answer"])]}
        
    def buildSelCorrectivefRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve)  
        workflow.add_node("generate", generate) 
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("websearch", web_search)
        workflow.add_node("finalize_response", finalize_response)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("finalize_response", END)

        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "generate": "generate",
                "websearch": "websearch",
                "rewrite": "rewrite",
                "finalize_response": "finalize_response",
            },
        )

        # Compile
        return workflow.compile()

    app = buildSelCorrectivefRAG()

    global langMode
    langMode = isKorean(query)
    
    isTyping(connectionId, requestId)
    
    inputs = {"question": query}
    config = {"recursion_limit": 50}
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            #print("value: ", value)
            
    #print('value: ', value)
    #print('content: ', value["messages"][-1].content)
        
    readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
    return value["messages"][-1].content

####################### LangGraph #######################
# Plan and Execute
#########################################################
def run_plan_and_exeucute(connectionId, requestId, query):
    class Plan(BaseModel):
        """List of steps as a json format"""

        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )

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

    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        response: str

    def plan(state: State):
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

    def execute(state: State):
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

    def replan(state: State):
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
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        print('state: ', state)
        if "response" in state and state["response"]:
            return "end"
        else:
            return "continue"    

    def buildPlanAndExecute():
        workflow = StateGraph(State)
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

    app = buildPlanAndExecute()    
    
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

####################### LangGraph #######################
# Essay Writer
#########################################################
def run_essay_writer(connectionId, requestId, query):
    class State(TypedDict):
        task: str
        plan: list[str]
        essay: str
        critique: str
        content: List[str]
        revision_number: int
        max_revisions: int

    class Plan(BaseModel):
        """List of steps as a json format"""

        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )
        
    def get_planner():
        system = """You are an expert writer tasked with writing a high level outline of an essay. \
    Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
    or instructions for the sections. \
    Make sure that each session has all the information needed."""
        
        #system = """You are an expert writer tasked with writing a high level outline of an essay.\
    #For the given objective, come up with a simple step by step plan. \
    #This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    #The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""
                
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("placeholder", "{messages}"),
            ]
        )
            
        chat = get_chat()   
            
        planner = planner_prompt | chat
        return planner
        
    def plan(state: State):
        print("###### plan ######")
        print('task: ', state["task"])
            
        task = [HumanMessage(content=state["task"])]

        planner = get_planner()
        response = planner.invoke({"messages": task})
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
                "task": state["task"],
                "plan": parsed_info.steps
            }
        else:
            print('parsing_error: ', info['parsing_error'])
                
            return {"plan": []}  
    
    class Queries(BaseModel):
        queries: List[str]
    
    def research_plan(state: State):
        task = state['task']
        print('task: ', task)
        
        system = """You are a researcher charged with providing information that can \
    be used when writing the following essay. Generate a list of search queries that will gather \
    any relevant information. Only generate 3 queries max."""
            
        research_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{task}"),
            ]
        )
            
        chat = get_chat()   
            
        research = research_prompt | chat
        
        response = research.invoke({"task": task})
        print('response.content: ', response.content)
        
        chat = get_chat()
        structured_llm = chat.with_structured_output(Queries, include_raw=True)
        info = structured_llm.invoke(response.content)
        # print('info: ', info)
        
        if not info['parsed'] == None:
            queries = info['parsed']
            print('queries: ', queries.queries)
            
        content = state["content"] if state.get("content") is not None else []
        search = TavilySearchResults(k=2)
        for q in queries.queries:
            response = search.invoke(q)     
            # print('response: ', response)        
            for r in response:
                content.append(r['content'])
        return {        
            "task": state['task'],
            "plan": state['plan'],
            "content": content,
        }
        
    def generation(state: State):    
        print('content: ', state['content'])
        print('task: ', state['task'])
        print('plan: ', state['plan'])
                            
        content = "\n\n".join(state['content'] or [])
        
    #    system = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
    #Generate the best essay possible for the user's request and the initial outline. \
    #If the user provides critique, respond with a revised version of your previous attempts. \
    #Utilize all the information below as needed: """
    #    system = """당신은 5문단의 에세이 작성을 돕는 작가입니다. \
    #용자의 요청에 대해 최고의 에세이를 작성하세요. \
    #사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요. \
    #최종 답변에는 완성된 에세이 전체 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."""
        system = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
    Generate the best essay possible for the user's request and the initial outline. \
    If the user provides critique, respond with a revised version of your previous attempts. \
    Utilize all the information below as needed: 

    <content>
    {content}
    </content>
    """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{task}\n\nHere is my plan:\n\n{plan}"),
            ]
        )
            
        chat = get_chat()
        chain = prompt | chat

        response = chain.invoke({
            "content": content,
            "task": state['task'],
            "plan": state['plan']
        })
        # print('response: ', response)
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "essay": response, 
            "revision_number": revision_number + 1
        }

    def reflection(state: State):    
        """You are a teacher grading an essay submission. \
    Generate critique and recommendations for the user's submission. \
    Provide detailed recommendations, including requests for length, depth, style, etc."""

        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
                    "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
                    #"특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다"
                    "각 문단의 길이는 최소 200자 이상이 되도록 관련된 예제를 충분히 포함합니다.",
                ),
                ("human", "{essay}"),
            ]
        )
        
        chat = get_chat()
        reflect = reflection_prompt | chat
                
        res = reflect.invoke({"essay": state['essay'].content})    
        response = HumanMessage(content=res.content)    
        
        return {
            "critique": response,
            "revision_number": int(state['revision_number'])
        }
    
    def research_critique(state: State):
        system = """You are a researcher charged with providing information that can \
    be used when making any requested revisions (as outlined below). \
    Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
        
        critique_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{critique}"),
            ]
        )
        
        chat = get_chat()           
        critique = critique_prompt | chat    
        response = critique.invoke({"critique": state['critique']})
        print('response.content: ', response.content)
        
        chat = get_chat()
        structured_llm = chat.with_structured_output(Queries, include_raw=True)
        info = structured_llm.invoke(response.content)
        # print('info: ', info)
        
        content = ""
        if not info['parsed'] == None:
            queries = info['parsed']
            print('queries: ', queries.queries)
            
            content = state["content"] if state.get("content") is not None else []
            search = TavilySearchResults(k=2)
            for q in queries.queries:
                response = search.invoke(q)     
                # print('response: ', response)        
                for r in response:
                    content.append(r['content'])
        return {
            "content": content,
            "revision_number": int(state['revision_number'])
        }
    
    MAX_REVISIONS = 2
    config = {"recursion_limit": 50}
    
    def should_continue(state, config):
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "contine"
    
    def buildEasyWriter():
        workflow = StateGraph(State)

        workflow.add_node("planner", plan)
        workflow.add_node("generation", generation)
        workflow.add_node("reflection", reflection)
        workflow.add_node("research_plan", research_plan)
        workflow.add_node("research_critique", research_critique)

        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "generation", 
            should_continue, 
            {
                "end": END, 
                "contine": "reflection"}
        )

        workflow.add_edge("planner", "research_plan")
        workflow.add_edge("research_plan", "generation")

        workflow.add_edge("reflection", "research_critique")
        workflow.add_edge("research_critique", "generation")
        # graph = builder.compile(checkpointer=memory)

        return workflow.compile()
    
    app = buildEasyWriter()    
    
    isTyping(connectionId, requestId)
    
    inputs = {"task": query}
    config = {
        "recursion_limit": 50,
        "max_revisions": 2
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["essay"])
    
    return value["essay"]
    
#########################################################
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def revise_question(connectionId, requestId, chat, query):    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)

def removeFunctionXML(msg):
    #print('msg: ', msg)
    
    while(1):
        start_index = msg.find('<function_calls>')
        end_index = msg.find('</function_calls>')
        length = 18
        
        if start_index == -1:
            start_index = msg.find('<invoke>')
            end_index = msg.find('</invoke>')
            length = 10
        
        output = ""
        if start_index>=0:
            # print('start_index: ', start_index)
            # print('msg: ', msg)
            
            if start_index>=1:
                output = msg[:start_index-1]
                
                if output == "\n" or output == "\n\n":
                    output = ""
            
            if end_index >= 1:
                # print('end_index: ', end_index)
                output = output + msg[end_index+length:]
                            
            msg = output
        else:
            output = msg
            break

    return output

def readStreamMsgForAgent(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            output = removeFunctionXML(msg)
            # print('output: ', output)
            
            if len(output)>0 and output[0]!='<':
                result = {
                    'request_id': requestId,
                    'msg': output,
                    'status': 'proceeding'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    # print('sendMessage size: ', len(body))
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        # raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def use_multimodal(img_base64, query):    
    multimodal = get_multimodal()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    multi_region = 'disable'
    if "multi_region" in jsonBody:
        multi_region = jsonBody['multi_region']
    print('multi_region: ', multi_region)
    
    global multiRegionGrade
    if multi_region == 'enable':
        multiRegionGrade = 'enable'
    else:
        multiRegionGrade = 'disable'
    
    print('initiate....')
    global reference_docs
    reference_docs = []

    global map_chain, memory_chain
    
    # Multi-LLM
    profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    # print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)
    
    chat = get_chat()    
    
    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
    
    start = int(time.time())    

    msg = ""
    reference = ""
    
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)                  

                elif convType == 'agent-executor':
                    msg = run_agent_executor(connectionId, requestId, text)
                        
                elif convType == 'agent-executor-chat':
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)  
                    msg = run_agent_executor(connectionId, requestId, revised_question)
                        
                elif convType == 'agent-reflection':  # reflection
                    msg = run_reflection_agent(connectionId, requestId, text)      
                    
                elif convType == 'agent-crag':  # corrective RAG
                    msg = run_corrective_rag(connectionId, requestId, text)
                                        
                elif convType == 'agent-srag':  # self RAG 
                    msg = run_self_rag(connectionId, requestId, text)
                    
                elif convType == 'agent-scrag':  # self-corrective RAG
                    msg = run_self_corrective_rag(connectionId, requestId, text)        
                
                elif convType == 'agent-plan-and-execute':  # self-corrective RAG
                    msg = run_plan_and_exeucute(connectionId, requestId, text)        
                                                
                elif convType == 'agent-essay-writer':  # essay writer
                    msg = run_essay_writer(connectionId, requestId, text)      
                
                elif convType == "translation":
                    msg = translate_text(chat, text) 
                
                elif convType == "grammar":
                    msg = check_grammer(chat, text)  
                
                else:
                    msg = general_conversation(connectionId, requestId, chat, text)  
                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                
                if reference_docs:
                    reference = get_references_for_agent(reference_docs)
                                        
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(path, doc_prefix, object)
                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                        
            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'uri': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                
            elif file_type == 'py' or file_type == 'js':
                s3r = boto3.resource("s3")
                doc = s3r.Object(s3_bucket, s3_prefix+'/'+object)
                
                contents = doc.get()['Body'].read().decode('utf-8')
                
                #contents = load_code(file_type, object)                
                                
                msg = summary_of_code(chat, contents, file_type)                  
                
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    img = img.resize((width, height))
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        
                command = ""        
                if 'command' in jsonBody:
                    command  = jsonBody['command']
                    print('command: ', command)
                
                # verify the image
                msg = use_multimodal(img_base64, command)       
                
                # extract text from the image
                text = extract_text(chat, img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
            
            else:
                msg = "uploaded file: "+object
        
        sendResultMessage(connectionId, requestId, msg+reference)
        # print('msg+reference: ', msg+reference)    
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg+reference}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            raise Exception ("Not able to write into dynamodb")               
        #print('resp, ', resp)

    return msg, reference

def lambda_handler(event, context):
    # print('event: ', event)    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg, reference = getResponse(connectionId, jsonBody)

                    print('msg+reference: ', msg+reference)
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }