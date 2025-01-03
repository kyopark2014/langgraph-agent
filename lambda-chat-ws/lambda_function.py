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
import uuid
import functools

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
from multiprocessing import Process, Pipe
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# from langchain.agents import tool
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from pytz import timezone
from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image
from opensearchpy import OpenSearch

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from pydantic.v1 import BaseModel, Field
from typing import Any, List, Tuple, Dict, Optional, cast, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict
from langchain_aws import AmazonKnowledgeBasesRetriever
from tavily import TavilyClient  

from dataclasses import dataclass, field
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
     
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
length_of_models = 1
selected_multimodal = 0
selected_embedding = 0
separated_chat_history = os.environ.get('separated_chat_history')
enableParentDocumentRetrival = os.environ.get('enableParentDocumentRetrival')
enableHybridSearch = os.environ.get('enableHybridSearch')
useParrelWebSearch = True
useEnhancedSearch = True
vectorIndexName = os.environ.get('vectorIndexName')
index_name = vectorIndexName
grade_state = "LLM" # LLM, PRIORITY_SEARCH, OTHERS
numberOfDocs = 2
minDocSimilarity = 400

prompt_flow_name = os.environ.get('prompt_flow_name')
rag_prompt_flow_name = os.environ.get('rag_prompt_flow_name')
knowledge_base_name = os.environ.get('knowledge_base_name')

"""  
multi_region_models = [  # claude sonnet 3.5
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude3.5",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude3.5",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "eu-central-1", # Frankfurt
        "model_type": "claude3.5",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "ap-northeast-1", # Tokyo
        "model_type": "claude3.5",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
]
"""
    
multi_region_models = [   # claude sonnet 3.0
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude3",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude3",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "ca-central-1", # Canada
        "model_type": "claude3",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "eu-west-2", # London
        "model_type": "claude3",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "sa-east-1", # Sao Paulo
        "model_type": "claude3",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    }
]
multi_region = 'enable'

titan_embedding_v1 = [  # dimension = 1536
  {
    "bedrock_region": "us-west-2", # Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  },
  {
    "bedrock_region": "us-east-1", # N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  }
]
priority_search_embedding = titan_embedding_v1
selected_ps_embedding = 0

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
tavily_api_key = []
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    # print('secret: ', secret)
    if secret['tavily_api_key']:
        tavily_api_key = json.loads(secret['tavily_api_key'])
    # print('tavily_api_key: ', tavily_api_key)
except Exception as e: 
    raise e

def check_tavily_secret(tavily_api_key):
    query = 'what is LangGraph'
    valid_keys = ""

    for i, key in enumerate(tavily_api_key):
        try:
            tavily_client = TavilyClient(api_key=key)
            response = tavily_client.search(query, max_results=1)
            # print('tavily response: ', response)
            
            if 'results' in response and len(response['results']):
                print('the valid tavily api keys: ', i)
                valid_keys = key
                break
        except Exception as e:
            print('Exception: ', e)
    # print('valid_keys: ', valid_keys)
    
    return valid_keys

tavily_key = check_tavily_secret(tavily_api_key)
# print('tavily_api_key: ', tavily_api_key)
os.environ["TAVILY_API_KEY"] = tavily_key
      
def tavily_search(query, k):
    docs = []
    try:
        tavily_client = TavilyClient(api_key=tavily_key)
        response = tavily_client.search(query, max_results=k)
        # print('tavily response: ', response)
            
        for r in response["results"]:
            name = r.get("title")
            if name is None:
                name = 'WWW'
            
            docs.append(
                Document(
                    page_content=r.get("content"),
                    metadata={
                        'name': name,
                        'url': r.get("url"),
                        'from': 'tavily'
                    },
                )
            )                   
    except Exception as e:
        print('Exception: ', e)

    return docs

# result = tavily_search('what is LangChain', 2)
# print('search result: ', result)

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

def reflesh_opensearch_index():
    #########################
    # opensearch index (reflesh)
    #########################
    print(f"deleting opensearch index... {index_name}") 
    
    try: # create index
        response = os_client.indices.delete(
            index_name
        )
        print('opensearch index was deleted:', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        #raise Exception ("Not able to create the index")        
    return 
    
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
    
    if multi_region == 'enable':
        profile = multi_region_models[selected_chat]
        length_of_models = len(multi_region_models)
    else:
        profile = LLM_for_chat[selected_chat]
        length_of_models = 1
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
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
    if selected_chat == length_of_models:
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
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
    maxOutputTokens = 4096
    print(f'LLM: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
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
    print(f'selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}')
    
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
    # print('prompt: ', prompt)
    
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
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text' and text and msg:
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg) 
                
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
    if isKorean(query)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "You will be acting as a thoughtful advisor."
            "Using the following conversation, answer friendly for the newest question." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer." 
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)])
    # print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    # print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId, "")  
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

def get_answer_using_opensearch(connectionId, requestId, chat, text):    
    # retrieve
    isTyping(connectionId, requestId, "retrieving...")
    relevant_docs = retrieve_documents_from_opensearch(text, top_k=4)
    
    # grade
    isTyping(connectionId, requestId, "grading...")    
    filtered_docs = grade_documents(text, relevant_docs) # grading    
    filtered_docs = check_duplication(filtered_docs) # check duplication
            
    # generate
    isTyping(connectionId, requestId, "generating...")  
    msg = generate_answer_with_stream(connectionId, requestId, chat, filtered_docs, text)
               
    return msg

def retrieve_documents_from_opensearch(query, top_k=4):
    print("###### retrieve_documents_from_opensearch ######")

    bedrock_embedding = get_embedding()       
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    relevant_docs = []
    if enableParentDocumentRetrival == 'enable':  
        result = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k*2,  
            search_type="script_scoring",
            pre_filter={"term": {"metadata.doc_level": "child"}}
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
                                    
        # print('relevant_documents: ', relevant_documents)    
        for i, doc in enumerate(relevant_documents):
            if len(doc[0].page_content)>=100:
                text = doc[0].page_content[:100]
            else:
                text = doc[0].page_content            
            print(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")
    
        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
            
            content, name, url = get_parent_content(parent_doc_id) # use pareant document
            #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {content}")
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    
    else: 
        print("###### similarity_search_with_score ######")
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k
        )
        
        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    # print('the number of docs (vector search): ', len(relevant_docs))
            
    if enableHybridSearch == 'true':
        relevant_docs += lexical_search(query, top_k)    

    return relevant_docs

def retrieve_documents_from_tavily(query, top_k):
    print("###### retrieve_documents_from_tavily ######")

    relevant_documents = []
    search = TavilySearchResults(
        max_results=top_k,
        include_answer=True,
        include_raw_content=True,
        search_depth="advanced", 
        include_domains=["google.com", "naver.com"]
    )
                    
    try: 
        output = search.invoke(query)
        # print('tavily output: ', output)
            
        for result in output:
            print('result of tavily: ', result)
            if result:
                content = result.get("content")
                url = result.get("url")
                
                relevant_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': 'WWW',
                            'url': url,
                            'from': 'tavily'
                        },
                    )
                )                
    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        # raise Exception ("Not able to request to tavily")   

    return relevant_documents 

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('url: ', metadata['url'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    url = ""
    if "url" in metadata:
        url = metadata['url']
    
    return source['text'], metadata['name'], url

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
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer
    
@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    # print('timestr:', timestr)
    
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
        
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            search_depth="advanced", # "basic"
            include_domains=["google.com", "naver.com"]
        )
                    
        try: 
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
                                'url': url,
                                'from': 'tavily'
                            },
                        )
                    )                
                    answer = answer + f"{content}, URL: {url}\n"
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to tavily")   
        
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
    
    # retrieve
    relevant_docs = retrieve_documents_from_opensearch(keyword, top_k=2)                        
    print('relevant_docs length: ', len(relevant_docs))

    # grade
    filtered_docs = grade_documents(keyword, relevant_docs)
        
    for i, doc in enumerate(filtered_docs):
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content
            
        print(f"filtered doc[{i}]: {text}, metadata:{doc.metadata}")
       
    relevant_context = "" 
    for doc in filtered_docs:
        content = doc.page_content
        
        relevant_context = relevant_context + f"{content}\n\n"
        
    return relevant_context

def lexical_search(query, top_k):
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
        index=index_name
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
        
        url = ""
        if "url" in document['_source']['metadata']:
            url = document['_source']['metadata']['url']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
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

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
                                    
def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == length_of_models:
            selected_chat = 0
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

def web_search_by_tavily(conn, q, k):     
    search = TavilySearchResults(max_results=k) 
    response = search.invoke(q)     
    print('response: ', response)
    
    content = []
    for r in response:
        if 'content' in r:
            content.append(r['content'])
        
    conn.send(content)    
    conn.close()
    
def web_search_using_parallel_processing(qurles):
    content = []    

    processes = []
    parent_connections = []
    
    k = 2
    for i, q in enumerate(qurles):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
                    
        process = Process(target=web_search_by_tavily, args=(child_conn, q, k))
        processes.append(process)
        
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        content += parent_conn.recv()
        
    for process in processes:
        process.join()
    
    return content

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")
    
knowledge_base_id = None
def retrieve_from_knowledge_base(query):
    global knowledge_base_id
    if not knowledge_base_id:        
        client = boto3.client('bedrock-agent')         
        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('response: ', response)
                
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
                    break
    
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}},
        )
        
        relevant_docs = retriever.invoke(query)
        print(relevant_docs)
    
    docs = []
    for i, document in enumerate(relevant_docs):
        #print(f"{i}: {document.page_content}")
        print_doc(i, document)
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]
        # print('score:', score)
        doc_prefix = "knowledge-base"
        
        link = ""
        if "s3Location" in document.metadata["location"]:
            link = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
            
            # print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            # print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in document.metadata["location"]:
            link = document.metadata["location"]["webLocation"]["url"] if document.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "Web Crawler"

        # print('link:', link)                    

        docs.append(
            Document(
                page_content=excerpt,
                metadata={
                    'name': name,
                    'url': link,
                    'from': 'RAG'
                },
            )
        )
    return docs

def check_duplication(docs):
    length_original = len(docs)
    
    contentList = []
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        # print('excerpt: ', doc['metadata']['excerpt'])
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updateed_docs = len(updated_docs)     
    
    if length_original == length_updateed_docs:
        print('no duplication')
    
    return updated_docs

def get_ps_embedding():
    global selected_ps_embedding
    profile = priority_search_embedding[selected_ps_embedding]
    bedrock_region =  profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_ps_embedding: {selected_ps_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}')
    
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
    
    bedrock_ps_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  

    selected_ps_embedding = selected_ps_embedding + 1
    if selected_ps_embedding == len(priority_search_embedding):
        selected_ps_embedding = 0

    return bedrock_ps_embedding

def priority_search(query, relevant_docs, minSimilarity):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        #print('doc: ', doc)

        content = doc.page_content
        # print('content: ', content)

        excerpts.append(
            Document(
                page_content=content,
                metadata={
                    'name': doc.metadata['name'],
                    'url': doc.metadata['url'],
                    'from': doc.metadata['from'],
                    'order':i,
                    'score':0
                }
            )
        )
    #print('excerpts: ', excerpts)

    docs = []
    if len(excerpts):
        embeddings = get_ps_embedding()
        vectorstore_confidence = FAISS.from_documents(
            excerpts,  # documents
            embeddings  # embeddings
        )            
        rel_documents = vectorstore_confidence.similarity_search_with_score(
            query=query,
            k=len(relevant_docs)
        )
        
        for i, document in enumerate(rel_documents):
            print(f'## Document(priority_search) query: {query}, {i+1}: {document}')

            order = document[0].metadata['order']
            name = document[0].metadata['name']
            
            score = document[1]
            print(f"query: {query}, {order}: {name}, {score}")

            relevant_docs[order].metadata['score'] = int(score)

            if score < minSimilarity:
                docs.append(relevant_docs[order])    
        # print('selected docs: ', docs)

    return docs
        
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    print("start grading...")
    print("grade_state: ", grade_state)
        
    if grade_state == "LLM":
        if multi_region == 'enable':  # parallel processing
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
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

    elif grade_state == "PRIORITY_SEARCH" and len(documents):
        filtered_docs = priority_search(question, documents, minDocSimilarity)
    else:  # OTHERS
        filtered_docs = documents
    
    global reference_docs 
    reference_docs += filtered_docs    
    # print('langth of reference_docs: ', len(reference_docs))
    
    # print('len(docments): ', len(filtered_docs))    
    return filtered_docs

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")
    
def get_references(docs):
    reference = "\n\nFrom\n"
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            #print('url: ', url)                
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
           
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            if useEnhancedSearch:
                sourceType = "OpenSearch"
            else:
                sourceType = "WWW"
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
        # print('excerpt(quotation removed): ', excerpt)
        
        if page:                
            reference = reference + f"{i+1}. {page}page in <a href={url} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
        else:
            reference = reference + f"{i+1}. <a href={url} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
    return reference

def web_search(question, documents):
    global reference_docs
    
    # Web search
    web_search_tool = TavilySearchResults(max_results=3)
    
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
                    'url': url,
                    'from': 'tavily'
                },
            )
        )
    return documents

def get_reg_chain(langMode):
    if langMode:
        system = (
            "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."   
        )
    else: 
        system = (
            "Here is pieces of context, contained in <context> tags." 
            "Provide a concise answer to the question at the end." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        )
        
    human = (
        "<question>"
        "{question}"
        "</question>"

        "<context>"
        "{context}"
        "</context>"
    )
        
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
        system = (
            "You a question re-writer that converts an input question to a better version that is optimized"
            "for web search. Look at the input and try to reason about the underlying semantic intent / meaning."
        )
        
    print('system: ', system)
        
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    question_rewriter = re_write_prompt | structured_llm_rewriter
    return question_rewriter

def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
        
    chat = get_chat()
    structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
        
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question."
        "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader

@tool
def grade_answer_for_tool(question: str, answer: str):
    """
    Grade whether the answer is useful or not
    keyword: question and generated answer which could be useful
    return: binary score represented by "yes" or "no"
    """    
    print("###### grade_answer ######")
    print('question: ', question)
    print('answer: ', answer)
                
    answer_grader = get_answer_grader()    
    score = answer_grader.invoke({"question": question, "generation": answer})
    answer_grade = score.binary_score        
    print("answer_grade: ", answer_grade)

    if answer_grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        result_str = f"This answer was varified: {answer}"
        return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return f"This answer is invalid. Try again from retrieval_node"
    
def get_hallucination_grader():    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
        
    system = (
        "You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts."
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts."
    )    
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
    
# define tools
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

def update_state_message(msg:str, config):
    print(msg)
    # print('config: ', config)
    
    requestId = config.get("configurable", {}).get("requestId", "")
    connectionId = config.get("configurable", {}).get("connectionId", "")
    
    isTyping(connectionId, requestId, msg)
    
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

    def call_model(state: State, config):
        question = state["messages"]
        print('question: ', question)
        
        update_state_message("thinking...", config)
            
        if isKorean(question[0].content)==True:
            system = (
                "당신은 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
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
        print('call_model response: ', response.tool_calls)
        
        # state messag
        if response.tool_calls:
            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                print('tool name: ', toolinfo['name'])                    
                update_state_message(f"calling... {toolinfo['name']}", config)
        
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

def enhanced_search(query, config):
    print("###### enhanced_search ######")
    inputs = [HumanMessage(content=query)]
        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    print('result: ', result)
            
    message = result["messages"][-1]
    print('enhanced_search: ', message)

    return message.content[message.content.find('<result>')+8:len(message.content)-9]

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

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
        print("###### should_continue ######")
        messages = state["messages"]    
        print('last_message: ', messages[-1])
        
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State, config):
        print("###### call_model ######")
        # print('state: ', state["messages"])
        
        update_state_message("thinking...", config)
        
        if isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함합니다."
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
            
        response = chain.invoke(state["messages"])
        print('call_model response: ', response.tool_calls)
        
        # state messag
        if response.tool_calls:
            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                print('tool name: ', toolinfo['name'])                    
                update_state_message(f"calling... {toolinfo['name']}", config)
        
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
        
    isTyping(connectionId, requestId, "thinking...")
    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    message = ""
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        # print('event: ', event)
        
        message = event["messages"][-1]
        # print('message: ', message)

    msg = readStreamMsg(connectionId, requestId, message.content)

    #return msg[msg.find('<result>')+8:len(msg)-9]
    return msg

####################### LangGraph #######################
# Chat Agent Executor (v2)
# Reference: https://github.com/kyopark2014/langgraph-agent/blob/main/multi-agent.md
#########################################################
def run_agent_executor2(connectionId, requestId, query):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        sender: str

    tool_node = ToolNode(tools)
            
    def create_agent(chat, tools: str):        
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)
        
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."

            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            "You have access to the following tools: {tool_names}."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | chat.bind_tools(tools)
    
    def agent_node(state, agent, name):
        print(f"###### agent_node:{name} ######")        
        print('state: ', state["messages"])
        
        response = agent.invoke(state["messages"])
        print('response: ', response)
                
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(response, ToolMessage):
            pass
        else:
            response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)            
            
        return {
            "messages": [response],
            "sender": name,
        }
    
    chat = get_chat()

    execution_agent = create_agent(chat, tools)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State) -> Literal["continue", "end"]:
        print("###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
        else:      
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"

    def buildAgentExecutor():
        workflow = StateGraph(State)

        workflow.add_node("agent", execution_agent_node)
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

    app = buildAgentExecutor()
        
    isTyping(connectionId, requestId, "")
    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    message = ""
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        # print('event: ', event)
        
        message = event["messages"][-1]
        # print('message: ', message)

    msg = readStreamMsg(connectionId, requestId, message.content)

    #return msg[msg.find('<result>')+8:len(msg)-9]
    return msg

####################### LangGraph #######################
# Reflection Agent
#########################################################
def run_reflection_agent(connectionId, requestId, query):
    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    def generation_node(state: State, config):    
        print("###### generation ######")      
        update_state_message("generating...", config)
        
        system = (
            "당신은 5문단의 에세이 작성을 돕는 작가이고 이름은 서연입니다"
            "사용자의 요청에 대해 최고의 에세이를 작성하세요."
            "사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요."
            "최종 답변에는 완성된 에세이 전체 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chat = get_chat()
        chain = prompt | chat

        response = chain.invoke({"messages":state["messages"]})
        return {"messages": [response]}

    def reflection_node(state: State, config):
        print("###### reflection ######")
        messages = state["messages"]
        
        update_state_message("reflecting...", config)
        
        system = (
            "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
            "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
            #"특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다"
            "각 문단의 길이는 최소 200자 이상이 되도록 관련된 예제를 충분히 포함합니다."
        )
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
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
        print("###### should_continue ######")
        messages = state["messages"]
        
        if len(messages) >= 6:   # End after 3 iterations        
            return "end"
        else:
            return "continue"

    def buildReflectionAgent():
        workflow = StateGraph(State)
        workflow.add_node("generate", generation_node)
        workflow.add_node("reflect", reflection_node)
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

    isTyping(connectionId, requestId, "")
    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
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

    def retrieve_node(state: State):
        print("###### retrieve ######")
        question = state["question"]
        
        docs = retrieve_documents_from_opensearch(question, top_k=4)
        
        return {"documents": docs, "question": question}

    def grade_documents_node(state: State, config):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("grading...", config)
        
        # Score each doc
        filtered_docs = []
        print("start grading...")
        print("grade_state: ", grade_state)
        
        web_search = "No"
        
        if grade_state == "LLM":
            if multi_region == 'enable':  # parallel processing            
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
            
        elif grade_state == "PRIORITY_SEARCH":
            filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents
        
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

    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
            
        return {"documents": documents, "question": question, "generation": generation}

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("rewriting...", config)

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def web_search_node(state: State, config):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("web searching...", config)

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def buildCorrectiveRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)

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
            
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
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
    
    def retrieve_node(state: State, config):
        print('state: ', state)
        print("###### retrieve ######")
        question = state["question"]
        
        update_state_message("retrieving...", config)
        
        docs = retrieve_documents_from_opensearch(question, top_k=4)
        
        return {"documents": docs, "question": question}
    
    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}
            
    def grade_documents_node(state: State, config):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        count = state["count"] if state.get("count") is not None else -1
        
        print("start grading...")
        print("grade_state: ", grade_state)
        update_state_message("grading...", config)
        
        if grade_state == "LLM":
            if multi_region == 'enable':  # parallel processing            
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

        elif grade_state == "PRIORITY_SEARCH":
            filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents
        
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

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("rewriting...", config)

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
        
        update_state_message("grading...", config)
        
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
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)

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
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
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

    def retrieve_node(state: State, config):
        print("###### retrieve ######")
        question = state["question"]
        
        update_state_message("retrieving...", config)
        
        docs = retrieve_documents_from_opensearch(question, top_k=4)
        
        return {"documents": docs, "question": question, "web_fallback": True}

    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        global reference_docs
        reference_docs += documents
        
        return {"retries": retries + 1, "candidate_answer": generation.content}

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        update_state_message("rewriting...", config)
        
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
        
        update_state_message("grading...", config)
        
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

    def web_search_node(state: State, config):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("web searching...", config)

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def finalize_response_node(state: State):
        print("###### finalize_response ######")
        return {"messages": [AIMessage(content=state["candidate_answer"])]}
        
    def buildSelCorrectivefRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("generate", generate_node) 
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)
        workflow.add_node("finalize_response", finalize_response_node)

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
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
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
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        answer: str

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
        print('input: ', state["input"])
        
        update_state_message("planning...", config)
        
        inputs = [HumanMessage(content=state["input"])]

        planner = get_planner()
        response = planner.invoke({"messages": inputs})
        print('response.content: ', response.content)
        
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Plan, include_raw=True)
            info = structured_llm.invoke(response.content)
            print(f'attempt: {attempt}, info: {info}')
            
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('parsed_info: ', parsed_info)        
                print('steps: ', parsed_info.steps)                
                return {
                    "input": state["input"],
                    "plan": parsed_info.steps
                }
        
        print('parsing_error: ', info['parsing_error'])
        return {"plan": []}          

    def execute_node(state: State, config):
        print("###### execute ######")
        print('input: ', state["input"])
        plan = state["plan"]
        print('plan: ', plan) 
        
        update_state_message("executing...", config)
        
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        #print("plan_str: ", plan_str)
        
        task = plan[0]
        task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        # print("request: ", task_formatted)     
        request = HumanMessage(content=task_formatted)
        
        chat = get_chat()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", (
                    "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
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
        
        print('task: ', task)
        print('executor output: ', output)
        
        # print('plan: ', state["plan"])
        # print('past_steps: ', task)        
        return {
            "input": state["input"],
            "plan": state["plan"],
            "info": [output],
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
            "For the given objective, come up with a simple step by step plan."
            "This plan should involve individual tasks, that if executed correctly will yield the correct answer."
            "Do not add any superfluous steps."
            "The result of the final step should be the final answer."
            "Make sure that each step has all the information needed - do not skip steps."

            "Your objective was this:"
            "<input>"
            "{input}"
            "</input>"
                        
            "Your original plan was this:"
            "<plan>"
            "{plan}"
            "</plan>"

            "You have currently done the follow steps:"
            "<past_steps>"
            "{past_steps}"
            "</past_steps>"

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
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        print('state: ', state)
        if "response" in state and state["response"]:
            return "end"
        else:
            return "continue"    
        
    def final_answer(state: State) -> str:
        print('#### final_answer ####')
        
        # get final answer
        context = state['info']
        print('context: ', context)
        
        query = state['input']
        print('query: ', query)
        
        if isKorean(query)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
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
        # print('prompt: ', prompt)
                    
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

    app = buildPlanAndExecute()    
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"input": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)
    
    readStreamMsg(connectionId, requestId, value["answer"])
        
    return value["answer"]

####################### LangGraph #######################
# Planning (Advanced CoT)
#########################################################
def run_planning(connectionId, requestId, query):
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        answer: str

    def plan_node(state: State, config):
        print("###### plan ######")
        print('input: ', state["input"])
        
        update_state_message("planning...", config)
                
        system = (
            "당신은 user의 question을 해결하기 위해 step by step plan을 생성하는 AI agent입니다."                
            
            "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."                
        )
        
        human = (
            "{question}"
        )
                            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        chat = get_chat()
        planner = planner_prompt | chat
        response = planner.invoke({
            "question": state["input"]
        })
        print('response.content: ', response.content)
        result = response.content
        
        #output = result[result.find('<result>')+8:result.find('</result>')]
        output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        print('planning_steps: ', planning_steps)
        
        return {
            "input": state["input"],
            "plan": planning_steps
        }
        
    def execute_node(state: State, config):
        print("###### execute ######")
        print('input: ', state["input"])
        plan = state["plan"]
        print('plan: ', plan) 
        
        update_state_message("executing...", config)
        chat = get_chat()

        requestId = config.get("configurable", {}).get("requestId", "")
        print('requestId: ', requestId)
        connectionId = config.get("configurable", {}).get("connectionId", "")
        print('connectionId: ', connectionId)

        # retrieve
        isTyping(connectionId, requestId, "retrieving...")
        relevant_docs = retrieve_documents_from_opensearch(plan[0], top_k=4)
        relevant_docs += retrieve_documents_from_tavily(plan[0], top_k=4)
            
        # grade
        isTyping(connectionId, requestId, "grading...")    
        filtered_docs = grade_documents(plan[0], relevant_docs) # grading    
        filtered_docs = check_duplication(filtered_docs) # check duplication
                
        # generate
        isTyping(connectionId, requestId, "generating...")                  
        result = generate_answer(chat, relevant_docs, plan[0])
        
        print('task: ', plan[0])
        print('executor output: ', result)
        
        # print('plan: ', state["plan"])
        # print('past_steps: ', task)        
        return {
            "input": state["input"],
            "plan": state["plan"],
            "info": [result],
            "past_steps": [plan[0]],
        }
            
    def replan_node(state: State, config):
        print('#### replan ####')
        print('state of replan node: ', state)
        
        update_state_message("replanning...", config)

        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
            "당신은 다음의 Question에 대한 적절한 답변을 얻고자합니다."
        )        
        human = (
            "<question>"
            "{input}"
            "</question>"            
                        
            "당신의 원래 계획은 아래와 같습니다." 
            "<plan>"
            "{plan}"
            "</plan>"
            
            "완료한 단계는 아래와 같습니다."
            "<past_steps>"
            "{past_steps}"
            "</past_steps>"
            
            "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
            "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 단계는 계획에 포함하지 마세요."                
            "수정된 계획에는 <plan> tag를 붙여주세요."
            "만약 더 이상 계획을 세우지 않아도 Question의 주어진 질문에 답변할 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
            
            "수정된 계획의 형식은 아래와 같습니다."
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."         
        )                   
        
        replanner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )     
        
        chat = get_chat()
        replanner = replanner_prompt | chat
        
        response = replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        print('replanner output: ', response.content)
        result = response.content

        if result.find('<plan>') == -1:
            return {"response":response.content}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            print('plan output: ', output)

            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            print('planning_steps: ', planning_steps)

            return {"plan": planning_steps}
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        # print('state: ', state)
        
        if "response" in state and state["response"]:
            print('response: ', state["response"])            
            next = "end"
        else:
            print('plan: ', state["plan"])
            next = "continue"
        print(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State) -> str:
        print('#### final_answer ####')
        
        # get final answer
        context = state['info']
        print('context: ', context)
        
        query = state['input']
        print('query: ', query)
        
        if isKorean(query)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "Here is pieces of context, contained in <context> tags."
                "Provide a concise answer to the question at the end."
                "Explains clearly the reason for the answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "Put it in <result> tags."
            )
    
        human = (
            "{input}"

            "<context>"
            "{context}"
            "</context>"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                    
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

            if result.find('<result>')==-1:
                output = result
            else:
                output = result[result.find('<result>')+8:result.find('</result>')]
                
            print('output: ', output)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)      
            
        return {"answer": output}  

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

    app = buildPlanAndExecute()    
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"input": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)
    
    readStreamMsg(connectionId, requestId, value["answer"])
        
    return value["answer"]

####################### LangGraph #######################
# Essay Writer
#########################################################
def run_essay_writer(connectionId, requestId, query):
    langMode = isKorean(query)
        
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
        if langMode:
            system = (
                "당신은당신은 에세이의 개요를 작성하고 있는 전문 작가입니다. "
                "사용자가 제공한 주제에 대해 다음과 같은 개요를 작성하세요. "
                "에세이의 개요와 함께 각 섹션에 대한 관련 메모나 지시사항을 제공하세요."
                "각 세션에 필요한 모든 정보가 포함되어 있는지 확인하세요."
            )
        else:
            system = (
                "You are an expert writer tasked with writing a high level outline of an essay."
                "Write such an outline for the user provided topic." 
                "Give an outline of the essay along with any relevant notes or instructions for the sections."
                "Make sure that each session has all the information needed."
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
        
    def plan(state: State, config):
        print("###### plan ######")
        print('task: ', state["task"])
        
        update_state_message("planning...", config)
            
        task = [HumanMessage(content=state["task"])]

        planner = get_planner()
        response = planner.invoke({"messages": task})
        print('response.content: ', response.content)
            
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Plan, include_raw=True)
            info = structured_llm.invoke(response.content)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('parsed_info: ', parsed_info)        
                print('steps: ', parsed_info.steps)
                    
                return {
                    "task": state["task"],
                    "plan": parsed_info.steps
                }
        
        print('parsing_error: ', info['parsing_error'])                
        return {"plan": []}  
    
    class Queries(BaseModel):
        """List of queries as a json format"""
        queries: List[str] = Field(
            description="queries for retrieve"
        )
    
    def research_plan(state: State):
        print("###### research_plan ######")
        task = state['task']
        print('task: ', task)
        
        update_state_message("researching...", config)
        
        if langMode:
            system = (
                "당신은 다음 에세이를 작성할 때 사용할 수 있는 정보를 제공하는 연구원입니다."
                "관련 정보를 수집할 수 있는 검색 쿼리 목록을 생성하세요. 최대 3개의 쿼리만 생성하세요."
            )
        else:
            system = (
                "You are a researcher charged with providing information that can be used when writing the following essay." 
                "Generate a list of search queries that will gather any relevant information. "
                "Only generate 3 queries max."
            )
            
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
        
        queries = None 
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Queries, include_raw=True)
            info = structured_llm.invoke(response.content)
            print(f'attempt: {attempt}, info: {info}')

            if not info['parsed'] == None:
                queries = info['parsed']
                print('queries: ', queries.queries)
                break
            
        content = state["content"] if state.get("content") is not None else []
        
        if not queries == None:
            if useParrelWebSearch:
                content += web_search_using_parallel_processing(queries.queries)
                
            else:        
                search = TavilySearchResults(max_results=2)
                for q in queries.queries:
                    response = search.invoke(q)     
                    # print('response: ', response)        
                    for r in response:
                        if 'content' in r:
                            content.append(r['content'])
                        
        return {        
            "task": state['task'],
            "plan": state['plan'],
            "content": content,
        }
        
    def generation(state: State, config):    
        print("###### generation ######")
        print('content: ', state['content'])
        print('task: ', state['task'])
        print('plan: ', state['plan'])
        
        update_state_message("generating...", config)
                            
        content = "\n\n".join(state['content'] or [])
        
    #    system = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
    #Generate the best essay possible for the user's request and the initial outline. \
    #If the user provides critique, respond with a revised version of your previous attempts. \
    #Utilize all the information below as needed: """
        if langMode:
            system = """당신은 5문단의 에세이 작성을 돕는 작가입니다. \
사용자의 요청에 대해 최고의 에세이를 작성하세요. \
사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요. \
최종 답변에는 완성된 에세이 전체 내용을 반드시 포함합니다.
<content>
{content}
</content>
"""
        else:
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

    def reflection(state: State, config):    
        print("###### reflection ######")
        
        update_state_message("reflecting...", config)
        
        if langMode:
            system = (
                "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
                "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
                "특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다."
                "각 문단의 길이는 최소 200자 이상이 되도록 관련된 예제를 충분히 포함합니다.,"
            )
        else: 
            system = (
                "You are a teacher grading an essay submission."
                "Generate critique and recommendations for the user's submission."
                "Provide detailed recommendations, including requests for length, depth, style, etc."
            )

        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
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
    
    def research_critique(state: State, config):
        print("###### research_critique ######")
        
        update_state_message("research critiquing...", config)
        
        if langMode:
            system = (
                "당신은 요청된 수정 사항을 만들 때 사용할 수 있는 정보를 제공하는 연구원입니다."
                "관련 정보를 수집할 수 있는 검색 쿼리 목록을 생성하세요. 최대 3개의 쿼리만 생성하세요."
            )
        else:
            system = (
                "You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below)."
                "Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."
            )
        
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
        
        content = ""
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Queries, include_raw=True)
            info = structured_llm.invoke(response.content)
            print(f'attempt: {attempt}, info: {info}')
            
            if not info['parsed'] == None:
                queries = info['parsed']
                print('queries: ', queries.queries)
                
                content = state["content"] if state.get("content") is not None else []
                
                if useParrelWebSearch:
                    c = web_search_using_parallel_processing(queries.queries)
                    print('content: ', c)            
                    content.extend(c)
                else:
                    search = TavilySearchResults(max_results=2)
                    for q in queries.queries:
                        response = search.invoke(q)     
                        # print('response: ', response)        
                        for r in response:
                            if 'content' in r:
                                content.append(r['content'])
                break
            
        return {
            "content": content,
            "revision_number": int(state['revision_number'])
        }
    
    MAX_REVISIONS = 2
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    def should_continue(state, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"
    
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
                "continue": "reflection"}
        )

        workflow.add_edge("planner", "research_plan")
        workflow.add_edge("research_plan", "generation")

        workflow.add_edge("reflection", "research_critique")
        workflow.add_edge("research_critique", "generation")
        # graph = builder.compile(checkpointer=memory)

        return workflow.compile()
    
    app = buildEasyWriter()    
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"task": query}
    config = {
        "recursion_limit": 50,
        "max_revisions": 2,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["essay"].content)
    
    return value["essay"].content


####################### LangGraph #######################
# Knowledge Guru
#########################################################
def run_knowledge_guru(connectionId, requestId, query):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        print("###### generate ######")
        print('state: ', state["messages"])
        print('task: ', state['messages'][0].content)
        
        update_state_message("generating...", config)
        
        draft = enhanced_search(state['messages'][0].content, config)  
        print('draft: ', draft)
        
        return {
            "messages": [AIMessage(content=draft)]
        }
    
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
    
    def reflect(state: State, config):
        print("###### reflect ######")
        print('state: ', state["messages"])    
        print('draft: ', state["messages"][-1].content)
        
        update_state_message("reflecting...", config)
    
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
            "messages": state["messages"],
            "reflection": reflection,
            "search_queries": search_queries
        }

    def revise_answer(state: State, config):   
        print("###### revise_answer ######")
        
        update_state_message("revising...", config)
        
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
            
        content = []        
        if useEnhancedSearch:
            for q in state["search_queries"]:
                response = enhanced_search(q, config)
                print(f'q: {q}, response: {response}')
                content.append(response)                   
        else:
            search = TavilySearchResults(max_results=2)
            for q in state["search_queries"]:
                response = search.invoke(q)     
                for r in response:
                    if 'content' in r:
                        content.append(r['content'])     

        chat = get_chat()
        reflect = reflection_prompt | chat
            
        messages = state["messages"]
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        print('translated: ', translated)     
           
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
            "messages": [response], 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

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
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    app = buildKnowledgeGuru()
        
    isTyping(connectionId, requestId, "")    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
    return value["messages"][-1].content

####################### LangGraph #######################
# Multi Agent
#########################################################
def run_multi_agent_tool(connectionId, requestId, query):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        sender: str
    
    tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch, grade_answer_for_tool]  
    tool_node = ToolNode(tools)
    
    def agent_node(state: State, agent, name):
        print(f"###### agent_node:{name} ######")        
        print('state: ', state)
        
        response = agent.invoke(state["messages"])
        print('response: ', response)
        if isinstance(response, ToolMessage):
            pass
        else:
            response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [response],
            "sender": name
        }
    
    def create_agent(chat, tools, system_message: str):
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)
                            
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "최종 답변에는 조사한 내용을 반드시 포함합니다."
            "You are a helpful AI assistant, collaborating with other assistants."
            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            #"If you or any of the other assistants have the final answer or deliverable,"
            #"prefix your response with FINAL ANSWER so the team knows to stop."
            "You have access to the following tools: {tool_names}."
            "{system_message}"
        )
    
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | chat.bind_tools(tools)
        
    chat = get_chat()
    retrieve_agent = create_agent(
        chat,
        [search_by_tavily, search_by_opensearch],
        system_message="You should provide accurate data for the chart_generator to use.",
    )
    retrieval_node = functools.partial(agent_node, agent=retrieve_agent, name="retrieve")
    
    #chat = get_chat()
    #verification_agent = create_agent(
    #    chat,
    #    [grade_answer_for_tool],
    #    system_message="You should verify the generated data is useful for the question.",
    #)
    #verification_node = functools.partial(agent_node, agent=verification_agent, name="verify")
    def verification_node(state: State):
        tools = [grade_answer_for_tool]
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)
                            
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "최종 답변에는 조사한 내용을 반드시 포함합니다."
            "You are a helpful AI assistant, collaborating with other assistants."
            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            #"If you or any of the other assistants have the final answer or deliverable,"
            #"prefix your response with FINAL ANSWER so the team knows to stop."
            "You have access to the following tools: {tool_names}."
            "{system_message}"
        )
    
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
                
        system_message="You should verify the generated data is useful for the question."
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=tool_names)
        
        chain = prompt | chat.bind_tools(tools)
        
        message = state["messages"][-1]
        
        # [state["messages"][0]] + [AIMessage(content=message.content)]
        question = state["messages"][0].content
        question_answer = f"question: {question}, answer:{message.content}"
        print('question_answer: ', question_answer)
        
        response = chain.invoke({"messages": [HumanMessage(content=question_answer)]})              
        print('response: ', response)
        
        name = "verify"
        if isinstance(response, ToolMessage):
            pass
        else:
            response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)
        
        return {
            "messages": [response],
            "sender": name
        }        
        # return {"messages": [res]}
    
    def router1(state) -> Literal["call_tool", "end", "continue"]:
        print(f"###### router1 ######")   
        print('state: ', state["messages"])
        
        last_message = state["messages"][-1]
        print("last_message: ", last_message)
        
        if not last_message.tool_calls:            
            if "FINAL ANSWER" or "This answer was varified" in last_message.content:
                return "end"
            return "continue"
        else: 
            return "call_tool"        
    
    def router2(state) -> Literal["call_tool", "end", "continue"]:
        print(f"###### router2 ######")   
        print('state: ', state["messages"])
        
        last_message = state["messages"][-1]
        print("last_message: ", last_message)
        
        if not last_message.tool_calls:            
            if "FINAL ANSWER" or "This answer was varified" in last_message.content:
                return "end"
            return "continue"
        else: 
            return "call_tool"        
        
    def router3(state):
        print(f"###### router3 ######")   
        print("state: ", state["messages"])
        sender = state["sender"]
        print("sender: ", sender)
            
        return sender
        
    def buildMultiAgent():    
        workflow = StateGraph(State)

        workflow.add_node("retrieve", retrieval_node)
        workflow.add_node("verify", verification_node)
        workflow.add_node("call_tool", tool_node)
        workflow.set_entry_point("retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            router1,
            {
                "continue": "verify", 
                "call_tool": "call_tool", 
                "end": END
            },
        )

        workflow.add_conditional_edges(
            "verify",
            router2,
            {
                "continue": "retrieve", 
                "call_tool": "call_tool", 
                "end": END
            },
        )        
        
        workflow.add_conditional_edges(
            "call_tool",
            router3,
            {
                "retrieve": "retrieve",
                "verify": "verify",
            },
        )
        app = workflow.compile()
        
        return app
    
    app = buildMultiAgent()
        
    isTyping(connectionId, requestId, "")    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    value = ""
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
    # return value["messages"][-1].content[value["messages"][-1].content.find('<result>')+8:len(value["messages"][-1].content)-9]
    return value["messages"][-1].content
    
####################### LangGraph #######################
# Writing Agent
#########################################################
def run_writing_agent(connectionId, requestId, query):
    class State(TypedDict):
        initial_prompt : str
        plan : str
        num_steps : int
        final_doc : str
        write_steps : List[str]
        word_count : int
    
    def planning_node(state: State, config):
        """take the initial prompt and write a plan to make a long doc"""
        print("---PLANNING THE WRITING---")
        
        update_state_message("planning...", config)
        
        initial_prompt = state['initial_prompt']
        num_steps = int(state['num_steps'])
        num_steps += 1
        
        human = """I need you to help me break down the following long-form writing instruction into multiple subtasks. \
Write a 5000 word piece. \
Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph. \
The writing instruction is as follows:

<instruction>
{intructions}
</instruction>

Please break it down in the following format, with each subtask taking up one line: \
1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 400 words]
2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1000 words].
...

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction. \
Do not split the subtasks too finely; each subtask's paragraph should be no less than 200 words and no more than 1000 words. \
Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."""
                    
        plan_prompt = ChatPromptTemplate.from_messages([("human", human)])
                
        chat = get_chat()
        plan_chain = plan_prompt | chat

        plan = plan_chain.invoke({"intructions": initial_prompt})
        print('plan: ', plan.content)

        return {"plan": plan.content, "num_steps":num_steps}
    
    def count_words(text):
        words = text.split()
        return len(words)
    
    def writing_node(state: State, config):
        """take the initial prompt and write a plan to make a long doc"""
        print("---WRITING THE DOC---")
        initial_instruction = state['initial_prompt']
        plan = state['plan']
        num_steps = int(state['num_steps'])
        num_steps += 1
        
        update_state_message("writing...", config)
        
        plan = plan.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        print('planning_steps: ', planning_steps)
            
        write_template = """You are an excellent writing assistant. I will give you an original writing instruction and my planned writing steps. \
I will also provide you with the text I have already written. \
Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text.

Writing instruction:
<instruction>
{intructions}
</instruction>

Writing steps:
<plan>
{plan}
</plan>

Already written text:
<text>
{text}
</text>

Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {STEP}. \
If needed, you can add a small subtitle at the beginning. \
Remember to only output the paragraph you write, without repeating the already written text.
"""                
        write_prompt = ChatPromptTemplate([
            ('human', write_template)
        ])
        
        text = ""
        responses = []
        if len(planning_steps) > 50:
            print("plan is too long")
            # print(plan)
            return
        
        for idx, step in enumerate(planning_steps):
            update_state_message(f"writing... (step: {idx+1}/{len(planning_steps)})", config)
            
            # Invoke the write_chain
            chat = get_chat()
            write_chain = write_prompt | chat
        
            result = write_chain.invoke({
                "intructions": initial_instruction,
                "plan": plan,
                "text": text,
                "STEP": step
            })
            print('result: ', result.content)
            
            print(f"----------------------------{idx}----------------------------")
            print(step)
            print("----------------------------\n\n")
            responses.append(result.content)
            text += result.content + '\n\n'

        final_doc = '\n\n'.join(responses)

        # Count words in the final document
        word_count = count_words(final_doc)
        print(f"Total word count: {word_count}")

        return {"final_doc": final_doc, "word_count": word_count, "num_steps":num_steps}            
    
    def saving_node(state: State, config):
        """take the finished long doc and save it to local disk as a .md file   """
        print("---SAVING THE DOC---")

        plan = state['plan']
        final_doc = state['final_doc']
        word_count = state['word_count']
        num_steps = int(state['num_steps'])
        num_steps += 1
        
        update_state_message("saving...", config)

        final_doc += f"\n\nTotal word count: {word_count}"

        print('plan: ', plan)
        print('final_doc: ', final_doc)
        print('word_count: ', word_count)
        
        # To-Do: save the result in S3 as a md file
        
        return {"num_steps":num_steps}
    
    def buildWriteAgent():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("planning_node", planning_node)
        workflow.add_node("writing_node", writing_node)
        workflow.add_node("saving_node", saving_node)

        # Set entry point
        workflow.set_entry_point("planning_node")

        # Add edges
        workflow.add_edge("planning_node", "writing_node")
        workflow.add_edge("writing_node", "saving_node")
        workflow.add_edge("saving_node", END)

        return workflow.compile()
    
    app = buildWriteAgent()
    
    #instruction = "Write a 5000 word piece on the HBO TV show WestWorld and its plot, characters, and themes. \
    #Make sure to cover the tropes that relate to AI, robots, and consciousness. \
    #Finally tackle where you think the show was going in future seasons had it not been cancelled."

    # Run the workflow
    isTyping(connectionId, requestId, "")    
    
    inputs = {
        "initial_prompt": query,
        "num_steps": 0
    }    
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
    
    return output['final_doc']
    
####################### LangGraph #######################
# Long form Writing Agent
#########################################################
def run_long_form_writing_agent(connectionId, requestId, query):
    # Workflow - Reflection
    class ReflectionState(TypedDict):
        draft : str
        reflection : List[str]
        search_queries : List[str]
        revised_draft: str
        revision_number: int
        reference: List[str]
        
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better writing")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the writing."""

        reflection: Reflection = Field(description="Your reflection on the initial writing.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current writing."
        )

    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="현재 글과 관련된 3개 이내의 검색어"
        )    
        
    def reflect_node(state: ReflectionState, config):
        print("###### reflect ######")
        draft = state['draft']
        print('draft: ', draft)
        
        idx = config.get("configurable", {}).get("idx")
        print('reflect_node idx: ', idx)
        update_state_message(f"reflecting... (search_queries-{idx})", config)
    
        reflection = []
        search_queries = []
        for attempt in range(5):
            chat = get_chat()
            if isKorean(draft):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
                
            info = structured_llm.invoke(draft)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('reflection: ', parsed_info.reflection)                
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                print('reflection: ', parsed_info.reflection)            
                print('search_queries: ', search_queries)     
        
                if isKorean(draft):
                    translated_search = []
                    for q in search_queries:
                        chat = get_chat()
                        if isKorean(q):
                            search = traslation(chat, q, "Korean", "English")
                        else:
                            search = traslation(chat, q, "English", "Korean")
                        translated_search.append(search)
                        
                    print('translated_search: ', translated_search)
                    search_queries += translated_search

                print('search_queries (mixed): ', search_queries)
                break
        
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }

    def retrieve_for_writing(conn, q, config):
        top_k = numberOfDocs
        
        relevant_docs = []
        # RAG - knowledge base
        #if rag_state=='enable':
        #    update_state_message(f"reflecting... (RAG_retriever-{idx})", config)
        #    docs = retrieve_from_knowledge_base(q, top_k)
        #    print(f'q: {q}, RAG: {docs}')
                            
        #    if len(docs):
        #        update_state_message(f"reflecting... (grader-{idx})", config)        
        #        fitered_docs = grade_documents(q, docs)
                
        #        print(f'retrieve {idx}: len(RAG_relevant_docs)=', len(relevant_docs))
        #        relevant_docs += fitered_docs
            
        # web search
        idx = config.get("configurable", {}).get("idx") 
        update_state_message(f"reflecting... (WEB_retriever-{idx})", config)    
        docs = tavily_search(q, top_k)
        print(f'q: {q}, WEB: {docs}')
                
        if len(docs):
            update_state_message(f"reflecting... (grader-{idx})", config)        
            fitered_docs = grade_documents(q, docs)
            
            print(f'retrieve {idx}: len(WEB_relevant_docs)=', len(relevant_docs))
            relevant_docs += fitered_docs
                    
        conn.send(relevant_docs)
        conn.close()

    def parallel_retriever(search_queries, config):
        relevant_documents = []    
        
        processes = []
        parent_connections = []
        for q in search_queries:
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retrieve_for_writing, args=(child_conn, q, config))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            rel_docs = parent_conn.recv()

            if(len(rel_docs)>=1):
                for doc in rel_docs:
                    relevant_documents.append(doc)    

        for process in processes:
            process.join()
        
        #print('relevant_docs: ', relevant_docs)
        return relevant_documents

    def retrieve_docs(search_queries, config):
        relevant_docs = []
        top_k = numberOfDocs
        
        idx = config.get("configurable", {}).get("idx")
        
        if multi_region == 'enable':
            relevant_docs = parallel_retriever(search_queries, config)        
        else:
            for q in search_queries:        
                # RAG - knowledge base
                #if rag_state=='enable':
                #    update_state_message(f"reflecting... (RAG_retriever-{idx})", config)
                #    docs = retrieve_from_knowledge_base(q, top_k)
                #    print(f'q: {q}, RAG: {docs}')
                            
                #    if len(docs):
                #        update_state_message(f"reflecting... (grader-{idx})", config)        
                #        relevant_docs += grade_documents(q, docs)
            
                # web search
                update_state_message(f"reflecting... (WEB_retriever-{idx})", config)
                docs = tavily_search(q, top_k)
                print(f'q: {q}, WEB: {docs}')
                
                if len(docs):
                    update_state_message(f"reflecting... (grader-{idx})", config)        
                    relevant_docs += grade_documents(q, docs)
                    
        for i, doc in enumerate(relevant_docs):
            print(f"#### {i}: {doc.page_content[:100]}")
        
        return relevant_docs
        
    def revise_draft(state: ReflectionState, config):   
        print("###### revise_answer ######")
        
        draft = state['draft']
        search_queries = state['search_queries']
        reflection = state['reflection']
        print('draft: ', draft)
        print('search_queries: ', search_queries)
        print('reflection: ', reflection)
                            
        # web search
        idx = config.get("configurable", {}).get("idx")
        print('revise_draft idx: ', idx)
        update_state_message(f"revising... (retrieve-{idx})", config)
        
        filtered_docs = retrieve_docs(search_queries, config)        
        print('filtered_docs: ', filtered_docs)
        
        if 'reference' in state:
            reference = state['reference']
            reference += filtered_docs
        else:
            reference = filtered_docs
        print('len(reference): ', reference)
        
        content = []   
        if len(filtered_docs):
            for d in filtered_docs:
                content.append(d.page_content)            
        print('content: ', content)
        
        update_state_message(f"revising... (generate-{idx})", config)
        
        if isKorean(draft):
            revise_template = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft을 critique과 information 사용하여 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
                            
                "<draft>"
                "{draft}"
                "</draft>"
                            
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
        else:    
            revise_template = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
                            
                "<draft>"
                "{draft}"
                "</draft>"
                            
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
                    
        revise_prompt = ChatPromptTemplate([
            ('human', revise_template)
        ])

        chat = get_chat()
        reflect = revise_prompt | chat
           
        res = reflect.invoke(
            {
                "draft": draft,
                "reflection": reflection,
                "content": content
            }
        )
        output = res.content
        # print('output: ', output)
        
        revised_draft = output[output.find('<result>')+8:len(output)-9]
        # print('revised_draft: ', revised_draft) 
            
        if revised_draft.find('#')!=-1 and revised_draft.find('#')!=0:
            revised_draft = revised_draft[revised_draft.find('#'):]

        print('--> draft: ', draft)
        print('--> reflection: ', reflection)
        print('--> revised_draft: ', revised_draft)
        
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
        return {
            "revised_draft": revised_draft,            
            "revision_number": revision_number,
            "reference": reference
        }
        
    MAX_REVISIONS = 1
    def should_continue(state: ReflectionState, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"        
    
    def buildReflection():
        workflow = StateGraph(ReflectionState)

        # Add nodes
        workflow.add_node("reflect_node", reflect_node)
        workflow.add_node("revise_draft", revise_draft)

        # Set entry point
        workflow.set_entry_point("reflect_node")
        
        workflow.add_conditional_edges(
            "revise_draft", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect_node"
            }
        )

        # Add edges
        workflow.add_edge("reflect_node", "revise_draft")
        
        return workflow.compile()
    
    # Workflow - Long Writing
    class State(TypedDict):
        instruction : str
        planning_steps : List[str]
        drafts : List[str]
        # num_steps : int
        final_doc : str
        word_count : int
            
    def plan_node(state: State, config):
        print("###### plan ######")
        instruction = state["instruction"]
        print('subject: ', instruction)
        
        update_state_message("planning...", config)
        
        if isKorean(instruction):
            planner_template = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "당신은 글쓰기 지시 사항을 여러 개의 하위 작업으로 나눌 것입니다."
                "글쓰기 계획은 5단계 이하로 작성합니다."
                "각 하위 작업은 에세이의 한 단락 작성을 안내할 것이며, 해당 단락의 주요 내용과 단어 수 요구 사항을 포함해야 합니다."

                "글쓰기 지시 사항:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "다음 형식으로 나누어 주시기 바랍니다. 각 하위 작업은 한 줄을 차지합니다:"
                "1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]."
                "..."
                
                "각 하위 작업이 명확하고 구체적인지, 그리고 모든 하위 작업이 작문 지시 사항의 전체 내용을 다루고 있는지 확인하세요."
                "과제를 너무 세분화하지 마세요. 각 하위 과제의 문단은 500단어 이상 3000단어 이하여야 합니다."
                "다른 내용은 출력하지 마십시오. 이것은 진행 중인 작업이므로 열린 결론이나 다른 수사학적 표현을 생략하십시오."                
            )
        else:
            planner_template = (
                "You are a helpful assistant highly skilled in long-form writing."
                "You will break down the writing instruction into multiple subtasks."
                "Writing plans are created in five steps or less."
                "Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph."

                "The writing instruction is as follows:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "Please break it down in the following format, with each subtask taking up one line:"
                "1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1500 words]."
                "..."
                
                "Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction."
                "Do not split the subtasks too finely; each subtask's paragraph should be no less than 500 words and no more than 3000 words."
                "Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."                
            )
        
        planner_prompt = ChatPromptTemplate([
            ('human', planner_template) 
        ])
                
        chat = get_chat()
        
        planner = planner_prompt | chat
    
        response = planner.invoke({"instruction": instruction})
        print('response: ', response.content)
    
        plan = response.content.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        print('planning_steps: ', planning_steps)
            
        return {
            "instruction": instruction,
            "planning_steps": planning_steps
        }
        
    def execute_node(state: State, config):
        print("###### write (execute) ######")        
        instruction = state["instruction"]
        planning_steps = state["planning_steps"]
        print('instruction: ', instruction)
        print('planning_steps: ', planning_steps)
        
        update_state_message("executing...", config)
        
        if isKorean(instruction):
            write_template = (
                "당신은 훌륭한 글쓰기 도우미입니다." 
                "아래와 같이 원본 글쓰기 지시사항과 계획한 글쓰기 단계를 제공하겠습니다."
                "또한 제가 이미 작성한 텍스트를 제공합니다."

                "글쓰기 지시사항:"
                "<instruction>"
                "{intructions}"
                "</instruction>"

                "글쓰기 단계:"
                "<plan>"
                "{plan}"
                "</plan>"

                "이미 작성한 텍스트:"
                "<text>"
                "{text}"
                "</text>"

                "글쓰기 지시 사항, 글쓰기 단계, 이미 작성된 텍스트를 참조하여 다음 단계을 계속 작성합니다."
                "다음 단계:"
                "<step>"
                "{STEP}"
                "</step>"
                
                "글이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
                "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
                "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
                "Markdown 포맷으로 서식을 작성하세요."
                "최종 결과에 <result> tag를 붙여주세요."
            )
        else:    
            write_template = (
                "You are an excellent writing assistant." 
                "I will give you an original writing instruction and my planned writing steps."
                "I will also provide you with the text I have already written."
                "Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text."

                "Writing instruction:"
                "<instruction>"
                "{intructions}"
                "</instruction>"

                "Writing steps:"
                "<plan>"
                "{plan}"
                "</plan>"

                "Already written text:"
                "<text>"
                "{text}"
                "</text>"

                "Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {STEP}."
                "If needed, you can add a small subtitle at the beginning."
                "Remember to only output the paragraph you write, without repeating the already written text."
                
                "Use markdown syntax to format your output:"
                "- Headings: # for main, ## for sections, ### for subsections, etc."
                "- Lists: * or - for bulleted, 1. 2. 3. for numbered"
                "- Do not repeat yourself"
                "Provide the final answer with <result> tag."
                #"Provide the final answer using Korean with <result> tag."
            )

        write_prompt = ChatPromptTemplate([
            ('human', write_template)
        ])
        
        text = ""
        drafts = []
        if len(planning_steps) > 50:
            print("plan is too long")
            # print(plan)
            return
        
        for idx, step in enumerate(planning_steps):
            update_state_message(f"executing... (step: {idx+1}/{len(planning_steps)})", config)
            
            # Invoke the write_chain
            chat = get_chat()
            write_chain = write_prompt | chat
            
            result = write_chain.invoke({
                "intructions": instruction,
                "plan": planning_steps,
                "text": text,
                "STEP": step
            })            
            output = result.content
            # print('output: ', output)
            
            draft = output[output.find('<result>')+8:len(output)-9]
            # print('draft: ', draft) 
                       
            if draft.find('#')!=-1 and draft.find('#')!=0:
                draft = draft[draft.find('#'):]
            
            print(f"--> step:{step}")
            print(f"--> {draft}")
                
            drafts.append(draft)
            text += draft + '\n\n'

        return {
            "instruction": instruction,
            "drafts": drafts
        }

    def reflect_draft(conn, reflection_app, idx, config, draft):     
        inputs = {
            "draft": draft
        }            
        output = reflection_app.invoke(inputs, config)
        
        print('idx: ', idx)
        
        result = {
            "revised_draft": output['revised_draft'],
            "idx": idx,
            "reference": output['reference']
        }
            
        conn.send(result)    
        conn.close()
        
    def reflect_drafts_using_parallel_processing(drafts, config):
        revised_drafts = drafts
        
        processes = []
        parent_connections = []
        references = []
        
        reflection_app = buildReflection()
        
        requestId = config.get("configurable", {}).get("requestId", "")
        print('requestId: ', requestId)
        connectionId = config.get("configurable", {}).get("connectionId", "")
        print('connectionId: ', connectionId)
        
        for idx, draft in enumerate(drafts):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            
            print(f"idx:{idx} --> draft:{draft}")
            
            app_config = {
                "recursion_limit": 50,
                "max_revisions": MAX_REVISIONS,
                "requestId":requestId,
                "connectionId": connectionId,
                "idx": idx
            }
            process = Process(target=reflect_draft, args=(child_conn, reflection_app, idx, app_config, draft))
            processes.append(process)
            
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            result = parent_conn.recv()

            if result is not None:
                print('result: ', result)
                revised_drafts[result['idx']] = result['revised_draft']
                references += result['reference']

        for process in processes:
            process.join()
                
        final_doc = ""   
        for revised_draft in revised_drafts:
            final_doc += revised_draft + '\n\n'
        
        return final_doc, references

    def get_subject(query):
        system = (
            "Extract the subject of the question in 6 words or fewer."
        )
        
        human = "<question>{question}</question>"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
        
        chat = get_chat()
        chain = prompt | chat    
        try: 
            result = chain.invoke(
                {
                    "question": query
                }
            )        
            subject = result.content
            # print('the subject of query: ', subject)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")        
        return subject
    
    def markdown_to_html(body, reference):
        body = body + f"\n\n### 참고자료\n\n\n"
        
        html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <md-block>
        </md-block>
        <script type="module" src="https://md-block.verou.me/md-block.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.css" integrity="sha512-n5zPz6LZB0QV1eraRj4OOxRbsV7a12eAGfFcrJ4bBFxxAwwYDp542z5M0w24tKPEhKk2QzjjIpR5hpOjJtGGoA==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
    </head>
    <body>
        <div class="markdown-body">
            <md-block>{body}
            </md-block>
        </div>
        {reference}
    </body>
    </html>"""        
        return html

    def get_references_for_html(docs):
        reference = ""
        nameList = []
        cnt = 1
        for i, doc in enumerate(docs):
            print(f"reference {i}: doc")
            page = ""
            if "page" in doc.metadata:
                page = doc.metadata['page']
                #print('page: ', page)            
            url = ""
            if "url" in doc.metadata:
                url = doc.metadata['url']
                #print('url: ', url)                
            name = ""
            if "name" in doc.metadata:
                name = doc.metadata['name']
                #print('name: ', name)     
            pos = name.rfind('/')
            name = name[pos+1:]
            print(f"name: {name}")
            
            excerpt = ""+doc.page_content

            excerpt = re.sub('"', '', excerpt)
            print('length: ', len(excerpt))
            
            if name in nameList:
                print('duplicated!')
            else:
                reference = reference + f"{cnt}. <a href={url} target=_blank>{name}</a><br>"
                nameList.append(name)
                cnt = cnt+1
                
        return reference

    def revise_answer(state: State, config):
        print("###### revise ######")
        drafts = state["drafts"]        
        print('drafts: ', drafts)
        
        update_state_message("revising...", config)
        
        # reflection
        if multi_region == 'enable':  # parallel processing
            final_doc, references = reflect_drafts_using_parallel_processing(drafts, config)
        else:
            reflection_app = buildReflection()
                
            final_doc = ""   
            references = []
            
            requestId = config.get("configurable", {}).get("requestId", "")
            print('requestId: ', requestId)
            connectionId = config.get("configurable", {}).get("connectionId", "")
            print('connectionId: ', connectionId)
            
            for idx, draft in enumerate(drafts):
                inputs = {
                    "draft": draft
                }                    
                app_config = {
                    "recursion_limit": 50,
                    "max_revisions": MAX_REVISIONS,
                    "requestId":requestId,
                    "connectionId": connectionId,
                    "idx": idx
                }
                output = reflection_app.invoke(inputs, config=app_config)
                final_doc += output['revised_draft'] + '\n\n'
                references += output['reference']

        subject = get_subject(state['instruction'])
        subject = subject.replace(" ","_")
        subject = subject.replace("?","")
        subject = subject.replace("!","")
        subject = subject.replace(".","")
        subject = subject.replace(":","")
        
        print('len(references): ', len(references))
        
        # markdown file
        markdown_key = 'markdown/'+f"{subject}.md"
        # print('markdown_key: ', markdown_key)
        
        markdown_body = f"## {state['instruction']}\n\n"+final_doc
                
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=markdown_key,
            ContentType='text/markdown',
            Body=markdown_body.encode('utf-8')
        )
        # print('response: ', response)
        
        markdown_url = f"{path}{markdown_key}"
        print('markdown_url: ', markdown_url)
        
        # html file
        html_key = 'markdown/'+f"{subject}.html"
        
        html_reference = ""
        print('references: ', references)
        if references:
            html_reference = get_references_for_html(references)
            
            global reference_docs
            reference_docs += references
            
        html_body = markdown_to_html(markdown_body, html_reference)
        print('html_body: ', html_body)
        
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=html_key,
            ContentType='text/html',
            Body=html_body
        )
        # print('response: ', response)
        
        html_url = f"{path}{html_key}"
        print('html_url: ', html_url)
        
        return {
            "final_doc": final_doc+f"\n<a href={html_url} target=_blank>[미리보기 링크]</a>\n<a href={markdown_url} download=\"{subject}.md\">[다운로드 링크]</a>"
        }
        
    def buildLongformWriting():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("planning_node", plan_node)
        workflow.add_node("execute_node", execute_node)
        workflow.add_node("revising_node", revise_answer)

        # Set entry point
        workflow.set_entry_point("planning_node")

        # Add edges
        workflow.add_edge("planning_node", "execute_node")
        workflow.add_edge("execute_node", "revising_node")
        workflow.add_edge("revising_node", END)
        
        return workflow.compile()
    
    app = buildLongformWriting()
    
    # Run the workflow
    isTyping(connectionId, requestId, "")        
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
    
    return output['final_doc']
            
####################### Knowledge Base #######################
# Knowledge Base
##############################################################
def get_reference_of_knoweledge_base(docs, path, doc_prefix):
    reference = "\n\nFrom\n"
    #print('path: ', path)
    #print('doc_prefix: ', doc_prefix)
    #print('prefix: ', f"/{doc_prefix}")
    
    for i, document in enumerate(docs):
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]
        print('score:', score)
        doc_prefix = "knowledge-base"
        
        link = ""
        if "s3Location" in document.metadata["location"]:
            link = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
            
            print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in document.metadata["location"]:
            link = document.metadata["location"]["webLocation"]["url"] if document.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "WWW"

        print('link:', link)
                    
        reference = reference + f"{i+1}. <a href={link} target=_blank>{name}</a>, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
                    
    return reference

knowledge_base_id = None
def get_answer_using_knowledge_base(chat, text, connectionId, requestId):    
    revised_question = text # use original question for test
 
    global knowledge_base_id
    if not knowledge_base_id:        
        client = boto3.client('bedrock-agent')         
        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('response: ', response)
                
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
                    break
    
    msg = reference = ""
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
        )
        
        relevant_docs = retriever.invoke(revised_question)
        print(relevant_docs)
        
        #selected_relevant_docs = []
        #if len(relevant_docs)>=1:
        #    print('start priority search')
        #    selected_relevant_docs = priority_search(revised_question, relevant_docs, minDocSimilarity)
        #    print('selected_relevant_docs: ', json.dumps(selected_relevant_docs))
        
    isTyping(connectionId, requestId, "generating...")  
    msg = generate_answer_with_stream(connectionId, requestId, chat, relevant_docs, revised_question)
    
    if len(relevant_docs):
        reference = get_reference_of_knoweledge_base(relevant_docs, path, doc_prefix)  
        
    return msg, reference
    
####################### Prompt Flow #######################
# Prompt Flow
###########################################################  
    
flow_arn = None
flow_alias_identifier = None
def run_prompt_flow(text, connectionId, requestId):    
    print('prompt_flow_name: ', prompt_flow_name)
    
    client = boto3.client(service_name='bedrock-agent')   
    
    global flow_arn, flow_alias_identifier
    
    if not flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        print('response: ', response)
        
        for flow in response["flowSummaries"]:
            print('flow: ', flow)
            if flow["name"] == prompt_flow_name:
                flow_arn = flow["arn"]
                print('flow_arn: ', flow_arn)
                break

    msg = ""
    if flow_arn:
        if not flow_alias_identifier:
            # get flow alias arn
            response_flow_aliases = client.list_flow_aliases(
                flowIdentifier=flow_arn
            )
            print('response_flow_aliases: ', response_flow_aliases)
            
            flowAlias = response_flow_aliases["flowAliasSummaries"]
            for alias in flowAlias:
                print('alias: ', alias)
                if alias['name'] == "latest_version":  # the name of prompt flow alias
                    flow_alias_identifier = alias['arn']
                    print('flowAliasIdentifier: ', flow_alias_identifier)
                    break
        
        # invoke_flow
        isTyping(connectionId, requestId, "")  
        
        client_runtime = boto3.client('bedrock-agent-runtime')
        response = client_runtime.invoke_flow(
            flowIdentifier=flow_arn,
            flowAliasIdentifier=flow_alias_identifier,
            inputs=[
                {
                    "content": {
                        "document": text,
                    },
                    "nodeName": "FlowInputNode",
                    "nodeOutputName": "document"
                }
            ]
        )
        print('response of invoke_flow(): ', response)
        
        response_stream = response['responseStream']
        try:
            result = {}
            for event in response_stream:
                print('event: ', event)
                result.update(event)
            print('result: ', result)

            if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
                print("Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
                # msg = result['flowOutputEvent']['content']['document']
                
                msg = readStreamMsg(connectionId, requestId, result['flowOutputEvent']['content']['document'])
                print('msg: ', msg)
            else:
                print("The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
        except Exception as e:
            raise Exception("unexpected event.",e)

    return msg

rag_flow_arn = None
rag_flow_alias_identifier = None
def run_RAG_prompt_flow(text, connectionId, requestId):
    global rag_flow_arn, rag_flow_alias_identifier
    
    print('rag_prompt_flow_name: ', rag_prompt_flow_name) 
    print('rag_flow_arn: ', rag_flow_arn)
    print('rag_flow_alias_identifier: ', rag_flow_alias_identifier)
    
    client = boto3.client(service_name='bedrock-agent')       
    if not rag_flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        print('response: ', response)
         
        for flow in response["flowSummaries"]:
            if flow["name"] == rag_prompt_flow_name:
                rag_flow_arn = flow["arn"]
                print('rag_flow_arn: ', rag_flow_arn)
                break
    
    if not rag_flow_alias_identifier and rag_flow_arn:
        # get flow alias arn
        response_flow_aliases = client.list_flow_aliases(
            flowIdentifier=rag_flow_arn
        )
        print('response_flow_aliases: ', response_flow_aliases)
        rag_flow_alias_identifier = ""
        flowAlias = response_flow_aliases["flowAliasSummaries"]
        for alias in flowAlias:
            print('alias: ', alias)
            if alias['name'] == "latest_version":  # the name of prompt flow alias
                rag_flow_alias_identifier = alias['arn']
                print('flowAliasIdentifier: ', rag_flow_alias_identifier)
                break
    
    # invoke_flow
    isTyping(connectionId, requestId, "")  
    
    client_runtime = boto3.client('bedrock-agent-runtime')
    response = client_runtime.invoke_flow(
        flowIdentifier=rag_flow_arn,
        flowAliasIdentifier=rag_flow_alias_identifier,
        inputs=[
            {
                "content": {
                    "document": text,
                },
                "nodeName": "FlowInputNode",
                "nodeOutputName": "document"
            }
        ]
    )
    print('response of invoke_flow(): ', response)
    
    response_stream = response['responseStream']
    try:
        result = {}
        for event in response_stream:
            print('event: ', event)
            result.update(event)
        print('result: ', result)

        if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
            print("Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
            # msg = result['flowOutputEvent']['content']['document']
            
            msg = readStreamMsg(connectionId, requestId, result['flowOutputEvent']['content']['document'])
            print('msg: ', msg)
        else:
            print("The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
    except Exception as e:
        raise Exception("unexpected event.",e)

    return msg


####################### Bedrock Agent #######################
# Bedrock Agent
#############################################################

agent_id = agent_alias_id = None
sessionId = dict() 
def run_bedrock_agent(text, connectionId, requestId, userId, sessionState):
    global agent_id, agent_alias_id
    print('agent_id: ', agent_id)
    print('agent_alias_id: ', agent_alias_id)
    
    client = boto3.client(service_name='bedrock-agent')  
    if not agent_id:
        response_agent = client.list_agents(
            maxResults=10
        )
        print('response of list_agents(): ', response_agent)
        
        for summary in response_agent["agentSummaries"]:
            if summary["agentName"] == "tool-executor":
                agent_id = summary["agentId"]
                print('agent_id: ', agent_id)
                break
    
    if not agent_alias_id and agent_id:
        response_agent_alias = client.list_agent_aliases(
            agentId = agent_id,
            maxResults=10
        )
        print('response of list_agent_aliases(): ', response_agent_alias)   
        
        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == "latest_version":
                agent_alias_id = summary["agentAliasId"]
                print('agent_alias_id: ', agent_alias_id) 
                break
    
    global sessionId
    if not userId in sessionId:
        sessionId[userId] = str(uuid.uuid4())
        
    msg = msg_contents = ""
    isTyping(connectionId, requestId, "")  
    if agent_alias_id and agent_id:
        client_runtime = boto3.client('bedrock-agent-runtime')
        try:
            if sessionState:
                response = client_runtime.invoke_agent( 
                    agentAliasId=agent_alias_id,
                    agentId=agent_id,
                    inputText=text, 
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId,
                    sessionState=sessionState
                )
            else:
                response = client_runtime.invoke_agent( 
                    agentAliasId=agent_alias_id,
                    agentId=agent_id,
                    inputText=text, 
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId
                )
            print('response of invoke_agent(): ', response)
            
            response_stream = response['completion']
            
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    msg += chunk.get('bytes').decode()
                    print('event: ', chunk.get('bytes').decode())
                        
                    result = {
                        'request_id': requestId,
                        'msg': msg,
                        'status': 'proceeding'
                    }
                    #print('result: ', json.dumps(result))
                    sendMessage(connectionId, result)
                    
                # files generated by code interpreter
                if 'files' in event:
                    files = event['files']['files']
                    for file in files:
                        objectName = file['name']
                        print('objectName: ', objectName)
                        contentType = file['type']
                        print('contentType: ', contentType)
                        bytes_data = file['bytes']
                                                
                        pixels = BytesIO(bytes_data)
                        pixels.seek(0, 0)
                                    
                        img_key = 'agent/contents/'+objectName
                        
                        s3_client = boto3.client('s3')  
                        response = s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=img_key,
                            ContentType=contentType,
                            Body=pixels
                        )
                        print('response: ', response)
                        
                        url = path+'agent/contents/'+parse.quote(objectName)
                        print('url: ', url)
                        
                        if contentType == 'application/json':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        elif contentType == 'application/csv':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        else:
                            width = 600            
                            msg_contents = f'\n\n<img src=\"{url}\" alt=\"{objectName}\" width=\"{width}\">'
                            print('msg_contents: ', msg_contents)
                                                            
        except Exception as e:
            raise Exception("unexpected event.",e)
        
    return msg+msg_contents

####################### LangGraph #######################
# RAG with Reflection
#########################################################
def run_rag_with_reflection(connectionId, requestId, query):   
    MAX_REVISIONS = 1
    class State(TypedDict):
        query: str
        draft: str
        relevant_docs: List[str]
        filtered_docs: List[str]
        reflection : List[str]
        sub_queries : List[str]
        revision_number: int 
        
    def continue_reflection(state: State, config):
        print("###### continue_reflection ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
                
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def retrieve_node(state: State, config):
        print("###### retrieve ######")
        query = state['query']
        
        update_state_message("retrieving...", config)
        
        relevant_docs = retrieve_from_knowledge_base(query)
        
        # print(f'q: {query}, RAG: {relevant_docs}')
        print(f'--> query: {query}, length: {len(relevant_docs)}')
        
        return {
            "relevant_docs": relevant_docs
        }
        
    def parallel_grader(state: State, config):
        print("###### parallel_grader ######")
        query = state['query']
        relevant_docs = state['relevant_docs']
        
        update_state_message("grading...", config)
        
        print('length of relevant_docs: ', len(relevant_docs))
        
        global selected_chat    
        filtered_docs = []    

        processes = []
        parent_connections = []
        
        for i, doc in enumerate(relevant_docs):
            #print(f"grading doc[{i}]: {doc.page_content}")        
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=grade_document_based_on_relevance, args=(child_conn, query, doc, multi_region_models, selected_chat))
            processes.append(process)

            selected_chat = selected_chat + 1
            if selected_chat == length_of_models:
                selected_chat = 0
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            doc = parent_conn.recv()

            if doc is not None:
                filtered_docs.append(doc)

        for process in processes:
            process.join()    
        #print('filtered_docs: ', filtered_docs)
        
        print('length of filtered_docs: ', len(filtered_docs))

        global reference_docs 
        reference_docs += filtered_docs    
        print('langth of reference_docs: ', len(reference_docs))    
        
        # duplication checker
        reference_docs = check_duplication(reference_docs)
        
        return {
            "filtered_docs": filtered_docs
        }    

    def generate_node(state: State, config):
        print("###### generate ######")
        query = state["query"]
        filtered_docs = state["filtered_docs"]
        print('query: ', query)
        print('filtered_docs: ', filtered_docs)
        
        update_state_message("generating...", config)
            
        # RAG generation
        rag_chain = get_reg_chain(isKorean(query))
            
        answer = rag_chain.invoke({"context": filtered_docs, "question": query})
        print('answer: ', answer.content)
                
        return {
            "draft": answer.content
        }

    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better writing")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the question/answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        sub_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )

    class ReflectionKor(BaseModel):
        missing: str = Field(description="답변에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 답변이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="답변의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """답변을 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="답변에 대한 평가")
        sub_queries: list[str] = Field(
            description="답변과 관련된 3개 이내의 검색어"
        )
        
    def reflect_node(state: State, config):
        print("###### reflect ######")
        #print('state: ', state)    
        query = state['query']
        print('query: ', query)
        draft = state['draft']
        print('draft: ', draft)
        
        update_state_message("reflecting...", config)
            
        reflection = []
        sub_queries = []
        for attempt in range(5):
            chat = get_chat()
            
            if isKorean(draft):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
                qa = f"질문: {query}\n\n답변: {draft}"
        
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
                qa = f"Question: {query}\n\nAnswer: {draft}"
            
            info = structured_llm.invoke(qa)
            print(f'attempt: {attempt}, info: {info}')
                    
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('reflection: ', parsed_info.reflection)
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                sub_queries = parsed_info.sub_queries
                    
                print('reflection: ', parsed_info.reflection)            
            
                if isKorean(draft):
                    translated_search = []
                    for q in sub_queries:
                        chat = get_chat()
                        if isKorean(q):
                            search = traslation(chat, q, "Korean", "English")
                        else:
                            search = traslation(chat, q, "English", "Korean")
                        translated_search.append(search)
                            
                    print('translated_search: ', translated_search)
                    sub_queries += translated_search

                break
        print('sub_queries: ', sub_queries)
            
        return {
            "reflection": reflection,
            "sub_queries": sub_queries,
        }

    def retriever(conn, query):
        relevant_docs = retrieve_from_knowledge_base(query)    
        print("---RETRIEVE: RELEVANT DOCUMENT---")
        print(f'--> query: {query}, length: {len(relevant_docs)}')
        
        conn.send(relevant_docs)    
        conn.close()
        
        return relevant_docs
    
    def parallel_retriever(state: State, config):
        print("###### parallel_retriever ######")
        sub_queries = state['sub_queries']
        print('sub_queries: ', sub_queries)
        
        update_state_message("retrieving...", config)
        
        relevant_docs = []
        processes = []
        parent_connections = []
        
        for i, query in enumerate(sub_queries):
            print(f"retrieve sub_queries[{i}]: {query}")        
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retriever, args=(child_conn, query))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            docs = parent_conn.recv()
            print('docs: ', docs)
            
            for doc in docs:
                relevant_docs.append(doc)

        for process in processes:
            process.join()    
        print('relevant_docs: ', relevant_docs)
        
        # duplication checker
        relevant_docs = check_duplication(relevant_docs)

        return {
            "relevant_docs": relevant_docs
        }

    def revise_node(state: State, config):   
        print("###### revise ######")
            
        draft = state['draft']
        reflection = state['reflection']
        print('draft: ', draft)
        print('reflection: ', reflection)
        
        update_state_message("revising...", config)
        
        if isKorean(draft):
            revise_template = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft을 critique과 information 사용하여 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
                                
                "<draft>"
                "{draft}"
                "</draft>"
                                
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
        else:    
            revise_template = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
                                
                "<draft>"
                "{draft}"
                "</draft>"
                            
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
                        
        revise_prompt = ChatPromptTemplate([
            ('human', revise_template)
        ])
        
        filtered_docs = state['filtered_docs']
        print('filtered_docs: ', filtered_docs)
                
        content = []   
        if len(filtered_docs):
            for d in filtered_docs:
                content.append(d.page_content)        
        print('content: ', content)

        chat = get_chat()
        reflect = revise_prompt | chat
            
        res = reflect.invoke(
            {
                "draft": draft,
                "reflection": reflection,
                "content": content
            }
        )
        output = res.content
        # print('output: ', output)
            
        revised_draft = output[output.find('<result>')+8:len(output)-9]
        # print('revised_draft: ', revised_draft) 
                
        print('--> draft: ', draft)
        print('--> reflection: ', reflection)
        print('--> revised_draft: ', revised_draft)
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
                
        return {
            "draft": revised_draft,
            "revision_number": revision_number + 1
        }

    def buildRagWithReflection():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("retrieve_node", retrieve_node)
        workflow.add_node("parallel_grader", parallel_grader)
        workflow.add_node("generate_node", generate_node)
        
        workflow.add_node("reflect_node", reflect_node)    
        workflow.add_node("parallel_retriever", parallel_retriever)    
        workflow.add_node("parallel_grader_subqueries", parallel_grader)
        workflow.add_node("revise_node", revise_node)

        # Set entry point
        workflow.set_entry_point("retrieve_node")
        
        workflow.add_edge("retrieve_node", "parallel_grader")
        workflow.add_edge("parallel_grader", "generate_node")
        
        workflow.add_edge("generate_node", "reflect_node")
        workflow.add_edge("reflect_node", "parallel_retriever")    
        workflow.add_edge("parallel_retriever", "parallel_grader_subqueries")    
        workflow.add_edge("parallel_grader_subqueries", "revise_node")
        
        workflow.add_conditional_edges(
            "revise_node", 
            continue_reflection, 
            {
                "end": END, 
                "continue": "reflect_node"}
        )
            
        return workflow.compile()

    app = buildRagWithReflection()
    
    # Run the workflow
    isTyping(connectionId, requestId, "")        
    inputs = {
        "query": query
    }    
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    output = app.invoke(inputs, config)
    print('output (run_rag_with_reflection): ', output)
    
    return output['draft']

####################### LangGraph #######################
# RAG with query trasnformation
#########################################################

def run_rag_with_transformation(connectionId, requestId, query):    
    MAX_REVISIONS = 1
    class State(TypedDict):
        query: str
        draft: str
        relevant_docs: List[str]
        filtered_docs: List[str]
        sub_queries : List[str]

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        query = state['query']
        
        update_state_message("rewriting...", config)
        
        query_rewrite_template = (
            "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system."
            "Given the original query, rewrite it to be more specific," 
            "detailed, and likely to retrieve relevant information."
            "Put it in <result> tags."

            "Original query: {original_query}"
            "Rewritten query:"
        )
        
        rewrite_prompt = ChatPromptTemplate([
            ('human', query_rewrite_template)
        ])

        chat = get_chat()
        rewrite = rewrite_prompt | chat
            
        res = rewrite.invoke({"original_query": query})    
        revised_query = res.content
        
        revised_query = revised_query[revised_query.find('<result>')+8:len(revised_query)-9] # remove <result> tag                   
        print('revised_query: ', revised_query)
        
        return {
            "query": revised_query
        }

    def decompose_node(state: State, config):
        print("###### decompose ######")
        query = state['query']
        
        update_state_message("decomposing...", config)
        
        if isKorean(query):
            subquery_decomposition_template = (
                "당신은 복잡한 쿼리를 RAG 시스템에 더 간단한 하위 쿼리로 분해하는 AI 어시스턴트입니다. "
                "주어진 원래 쿼리를 1-3개의 더 간단한 하위 쿼리로 분해하세요. "
                "최종 결과에 <result> tag를 붙여주세요."

                "<query>"
                "{original_query}"
                "</query>"

                "다음의 예제를 참조하여 쿼리를 생성합니다. 각 쿼리는 한 줄을 차지합니다:"
                "<example>"
                "질문: 기후 변화가 환경에 미치는 영향은 무엇입니까? "

                "하위 질문:"
                "1. 기후 변화가 환경에 미치는 주요 영향은 무엇입니까?"
                "2. 기후 변화는 생태계에 어떤 영향을 미칩니까? "
                "3. 기후 변화가 환경에 미치는 부정적인 영향은 무엇입니까?"
                "</example>"
            )
        else:
            subquery_decomposition_template = (
                "You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system."
                "Given the original query, decompose it into 1-3 simpler sub-queries."
                "Provide the final answer with <result> tag."

                "<query>"
                "{original_query}"
                "</query>"

                "Create queries referring to the following example. Each query occupies one line."
                "<example>"
                "Query: What are the impacts of climate change on the environment?"

                "Sub-queries:"
                "1. What are the impacts of climate change on biodiversity?"
                "2. How does climate change affect the oceans?"
                "3. What are the effects of climate change on agriculture?"
                "</example>"
            )    
            
        decomposition_prompt = ChatPromptTemplate([
            ('human', subquery_decomposition_template)
        ])

        chat = get_chat()
        
        decompose = decomposition_prompt | chat
        
        response = decompose.invoke({"original_query": query})
        print('response: ', response.content)
        
        result = response.content[response.content.find('<result>')+8:len(response.content)-9]
        print('result: ', result)
        
        result = result.strip().replace('\n\n', '\n')
        decomposed_queries = result.split('\n')        
        print('decomposed_queries: ', decomposed_queries)

        sub_queries = []    
        if len(decomposed_queries):
            sub_queries = decomposed_queries    
        else:
            sub_queries = [query]
        
        return {
            "sub_queries": [query] + sub_queries
        }    

    def retriever(conn, query):
        relevant_docs = retrieve_from_knowledge_base(query)    
        print("---RETRIEVE: RELEVANT DOCUMENT---")
        print(f'--> query: {query}, length: {len(relevant_docs)}')
        
        conn.send(relevant_docs)    
        conn.close()
        
        return relevant_docs
    
    def parallel_retriever(state: State, config):
        print("###### parallel_retriever ######")
        sub_queries = state['sub_queries']
        print('sub_queries: ', sub_queries)
        
        update_state_message("retrieving...", config)
        
        relevant_docs = []
        processes = []
        parent_connections = []
        
        for i, query in enumerate(sub_queries):
            print(f"retrieve sub_queries[{i}]: {query}")        
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retriever, args=(child_conn, query))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            docs = parent_conn.recv()
            print('docs: ', docs)
            
            for doc in docs:
                relevant_docs.append(doc)

        for process in processes:
            process.join()    
        print('relevant_docs: ', relevant_docs)
        
        # duplication checker
        relevant_docs = check_duplication(relevant_docs)

        return {
            "relevant_docs": relevant_docs
        }

    def parallel_grader(state: State, config):
        print("###### parallel_grader ######")
        query = state['query']
        relevant_docs = state['relevant_docs']
        
        update_state_message("grading...", config)
        
        print('length of relevant_docs: ', len(relevant_docs))
        
        global selected_chat    
        filtered_docs = []    

        processes = []
        parent_connections = []
        
        for i, doc in enumerate(relevant_docs):
            #print(f"grading doc[{i}]: {doc.page_content}")        
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=grade_document_based_on_relevance, args=(child_conn, query, doc, multi_region_models, selected_chat))
            processes.append(process)

            selected_chat = selected_chat + 1
            if selected_chat == length_of_models:
                selected_chat = 0
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            doc = parent_conn.recv()

            if doc is not None:
                filtered_docs.append(doc)

        for process in processes:
            process.join()    
        #print('filtered_docs: ', filtered_docs)
        
        print('length of filtered_docs: ', len(filtered_docs))

        global reference_docs 
        reference_docs += filtered_docs    
        print('langth of reference_docs: ', len(reference_docs))    
        
        # duplication checker
        reference_docs = check_duplication(reference_docs)
        
        return {
            "filtered_docs": filtered_docs
        }    
        
    def generate_node(state: State, config):
        print("###### generate ######")
        query = state["query"]
        filtered_docs = state["filtered_docs"]
        print('query: ', query)
        print('filtered_docs: ', filtered_docs)
        
        update_state_message("generating...", config)
            
        # RAG generation
        rag_chain = get_reg_chain(isKorean(query))
            
        answer = rag_chain.invoke({"context": filtered_docs, "question": query})
        print('answer: ', answer.content)
                
        return {
            "draft": answer.content
        }
        
    def buildRagWithTransformation():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("rewrite_node", rewrite_node)
        workflow.add_node("decompose_node", decompose_node)
        workflow.add_node("parallel_retriever", parallel_retriever)
        workflow.add_node("parallel_grader", parallel_grader)
        workflow.add_node("generate_node", generate_node)
        
        # Set entry point
        workflow.set_entry_point("rewrite_node")
        
        # Add edges
        workflow.add_edge("rewrite_node", "decompose_node")
        workflow.add_edge("decompose_node", "parallel_retriever")
        workflow.add_edge("parallel_retriever", "parallel_grader")
        workflow.add_edge("parallel_grader", "generate_node")    
        workflow.add_edge("generate_node", END)
                
        return workflow.compile()
    
    app = buildRagWithTransformation()
    
    # Run the workflow
    isTyping(connectionId, requestId, "")        
    inputs = {
        "query": query
    }    
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    output = app.invoke(inputs, config)
    print('output (run_rag_with_reflection): ', output)
    
    return output['draft']

####################### LangGraph #######################
# data enrichment agent
#########################################################
def run_data_enrichment_agent(connectionId, requestId, text):
    class State(TypedDict):
        messages: Annotated[List[BaseMessage],add_messages]=field(default_factory=list)
        loop_step: Annotated[int,operator.add]=field(default=0)
        topic: str
        extraction_schema: dict[str, Any]
        info: Optional[dict[str, Any]] = field(default=None)

    class OutputState:
        info: dict[str, Any]
    
    max_search_results = 10
    max_info_tool_calls = 3
    max_loops = 6
    
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
        # print('result of search: ', result)
        
        output = cast(list[dict[str, Any]], result)
        print('output of search: ', json.dumps(output))
        
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
            print('soup result: ', content)
            
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
            print('result of scrape_website: ', result)
            content = str(result.content)
        else:
            content = "Failed to retrieve the webpage. Status code: " + str(response.status_code)
            print(content)
        
        return content

    tools = [search, scrape_website]
    tool_node = ToolNode(tools)
    
    def agent_node(state: State) -> Dict[str, Any]:
        print("###### agent_node ######")
        
        info_tool = {
            "name": "Info",
            "description": "Call this when you have gathered all the relevant info",
            "parameters": state["extraction_schema"],
        }
        #print('topic: ', state["topic"])
        #print('schema: ', json.dumps(state["extraction_schema"]))
        
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
        print('messages: ', messages)

        chat = get_chat() 
        tools = [scrape_website, search, info_tool]
        model = chat.bind_tools(tools, tool_choice="any")
        result = model.invoke(messages)
        # print('result of call_model: ', result)
        
        response = cast(AIMessage, result)
        print('response of call_model: ', response)

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
            print('response.tool_calls: ', response.tool_calls)

        response_messages: List[BaseMessage] = [response]
        if not response.tool_calls:  
            response_messages.append(
                HumanMessage(content="Please respond by calling one of the provided tools.")
            )
        
        return {
            "messages": response_messages,
            "info": info,
            # Add 1 to the step count
            "loop_step": 1,
        }
        
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
        print('last_message: ', last_message)
        
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"{reflect_node.__name__} expects the last message in the state to be an AI message with tool calls."
                f" Got: {type(last_message)}"
            )
            
        # p = MAIN_PROMPT.format(
        #     info=json.dumps(state['extraction_schema'], indent=2), topic=state["topic"]
        # )
        # messages = [HumanMessage(content=p)] + state["messages"][:-1]
        # print('messages: ', messages)
        
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
        # print('result of checker_prompt: ', result)
        output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
        print('output of checker_prompt: ', output)
        
        response = ""
        reason = []
        is_satisfactory = False
        improvement_instructions = ""
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(InfoIsSatisfactory, include_raw=True)
            
            info = structured_llm.invoke(output)
            print(f'attempt: {attempt}, info: {info}')
        
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                print('parsed_info: ', parsed_info)
                
                reason = parsed_info.reason.values
                print('reason: ', reason)
                is_satisfactory = parsed_info.is_satisfactory
                print('is_satisfactory: ', is_satisfactory)
                improvement_instructions = parsed_info.improvement_instructions                
                print('improvement_instructions: ', improvement_instructions)
                
                response = cast(InfoIsSatisfactory, info)
                print('response of InfoIsSatisfactory: ', response)                
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
       
    def markdown_output(query, result):
        markdown_text = f"# {query}\n\n"

        for company in result["companies"]:
            markdown_text += f"""
    ## {company['name']}

    **Key Technologies:** {company['technologies']}

    **Market Share:** {company['market_share']}

    **Key Powers:** {company.get('key_powers', 'Not specified')}

    **Future Outlook:** {company['future_outlook']}

    ---
    """
        return markdown_text
    
    def text_output(result):
        text = ""

        if isKorean(result)==True:
            for i, company in enumerate(result["companies"]):
                text += f"""
{i+1}. {company['name']}

- 주요 기술: {company['technologies']}

- 시장 점유율: {company['market_share']}

- 핵심 경쟁력: {company.get('key_powers', 'Not specified')}

- 미래 전망: {company['future_outlook']}"""
        else:
            for i, company in enumerate(result["companies"]):
                text += f"""
{i+1}. {company['name']}

- Key Technologies: {company['technologies']}

- Market Share: {company['market_share']}

- Key Powers: {company.get('key_powers', 'Not specified')}

- Future Outlook: {company['future_outlook']}"""

        return text

    app = build_data_enrichment_agent()
    
    isTyping(connectionId, requestId, "")
    
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
    
    # message = ""
    # for event in app.stream(inputs, config, stream_mode="values"):
    #     print('event: ', event)
        
        #if "messages" in event:
        #    if len(event["messages"]) > 1:
        #        msg = readStreamMsg(connectionId, requestId, event["messages"][-1].content)
        #        message += msg
        #message = event["messages"][-1]
        #print('message: ', message)

    #msg = readStreamMsg(connectionId, requestId, message.content)
    #print('output: ', output)
    result = app.invoke(inputs, config)
    print('result: ', result)
    
    # final = markdown_output(text, result["info"])
    final = text_output(result["info"])
    
    if isKorean(text)==True:
         chat = get_chat()
         final = traslation(chat, final, "English", "Korean")
        
    return final
        
#########################################################
def generate_answer_with_stream(connectionId, requestId, chat, relevant_docs, question):        
    relevant_context = ""
    msg = ""    
    for i, document in enumerate(relevant_docs):
        # print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
    # print('relevant_context: ', relevant_context)
                
    if isKorean(question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                   
    chain = prompt | chat    
    try: 
        stream = chain.invoke(
            {
                "context": relevant_context,
                "input": question,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        print('msg: ', msg)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    return msg

def generate_answer(chat, relevant_docs, question):        
    relevant_context = ""
    msg = ""    
    for i, document in enumerate(relevant_docs):
        # print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
    # print('relevant_context: ', relevant_context)
                
    if isKorean(question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                   
    chain = prompt | chat    
    response = chain.invoke({
        "context": relevant_context,
        "input": question,
    })
    print('response.content: ', response.content)

    return msg

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
    # print('prompt: ', prompt)
    
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
    print("###### revise_question ######")
    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
    
    isTyping(connectionId, requestId, "revising...")
        
    if isKorean(query)==True :      
        human = (
            "이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요." 
            "새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다." 
            "결과는 <result> tag를 붙여주세요."
        
            "<question>"
            "{question}"
            "</question>"
        )
        
    else: 
        human = (
            "Rephrase the follow up <question> to be a standalone question." 
            "Put it in <result> tags."

            "<question>"
            "{question}"
            "</question>"
        )
            
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)]
    )
    # print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    # print('memory_chain: ', history)
                
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
        # print('revised_question: ', revised_question)
        
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

def isTyping(connectionId, requestId, msg):    
    if not msg:
        msg = "typing a message..."
    msg_proceeding = {
        'request_id': requestId,
        'msg': msg,
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
    # print('prompt: ', prompt)
    
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
            "Here is pieces of article, contained in <article> tags." 
            "Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
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

def extract_text(img_base64):    
    multimodal = get_multimodal()
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
        result = multimodal.invoke(messages)
        
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
    
    global multi_region    
    if "multi_region" in jsonBody:
        multi_region = jsonBody['multi_region']
    print('multi_region: ', multi_region)
    
    global grade_state
    if "grade" in jsonBody:
        grade_state = jsonBody['grade']
    else:
        grade_state = 'LLM'
    print('grade_state: ', grade_state)
        
    print('initiate....')
    global reference_docs
    reference_docs = []

    global map_chain, memory_chain
    
    # Multi-LLM
    global selected_chat, length_of_models
    if multi_region == 'enable':
        length_of_models = len(multi_region_models)
        if selected_chat >= length_of_models:
            selected_chat = 0
        profile = multi_region_models[selected_chat]
        
    else:
        length_of_models = len(LLM_for_chat)
        if selected_chat >= length_of_models:
            selected_chat = 0
        profile = LLM_for_chat[selected_chat]    
        
    print('length_of_models: ', length_of_models)
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    
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
        sendResultMessage(connectionId, requestId, msg)
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
                sendResultMessage(connectionId, requestId, msg)
            
            elif type == 'text' and body[:21] == 'reflesh current index':
                # reflesh index
                isTyping(connectionId, requestId, "")
                reflesh_opensearch_index()
                msg = "The index was refleshed in OpenSearch."
                sendResultMessage(connectionId, requestId, msg)
                
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)
                    
                elif convType == 'rag-opensearch':   # RAG - Vector
                    msg = get_answer_using_opensearch(connectionId, requestId, chat, text)                    

                elif convType == 'rag-opensearch-chat':   # RAG - Vector
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)
                    msg = get_answer_using_opensearch(connectionId, requestId, chat, text)      

                elif convType == 'agent-executor':
                    msg = run_agent_executor(connectionId, requestId, text)
                
                elif convType == 'agent-executor2':
                    msg = run_agent_executor2(connectionId, requestId, text)
                        
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
                
                elif convType == 'agent-plan-and-execute':  # plan and execute
                    msg = run_plan_and_exeucute(connectionId, requestId, text)        
                                                
                elif convType == 'agent-essay-writer':  # essay writer
                    msg = run_essay_writer(connectionId, requestId, text)      
                    
                elif convType == 'agent-knowledge-guru':  # knowledge guru
                    msg = run_knowledge_guru(connectionId, requestId, text)      
                
                elif convType == 'multi-agent-tool':  # multi-agent
                    msg = run_multi_agent_tool(connectionId, requestId, text)      
                    
                elif convType == 'writing-agent':  # writing agent
                    msg = run_writing_agent(connectionId, requestId, text)
                
                elif convType == 'long-form-writing-agent':  # long writing
                    msg = run_long_form_writing_agent(connectionId, requestId, text)
                    
                elif convType == "rag-knowledge-base":
                    msg, reference = get_answer_using_knowledge_base(chat, text, connectionId, requestId)                
                elif convType == "rag-knowledge-base":
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)      
                    msg, reference = get_answer_using_knowledge_base(chat, revised_question, connectionId, requestId)
                
                elif convType == "prompt-flow":
                    msg = run_prompt_flow(text, connectionId, requestId)
                elif convType == "prompt-flow-chat":
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)                    
                    msg = run_prompt_flow(revised_question, connectionId, requestId)
                
                elif convType == "rag-prompt-flow":
                    msg = run_RAG_prompt_flow(text, connectionId, requestId)
                
                elif convType == "bedrock-agent":
                    msg = run_bedrock_agent(text, connectionId, requestId, userId, "")
                
                elif convType == 'rag-with-reflection':  # rag-with-reflection
                    msg = run_rag_with_reflection(connectionId, requestId, text)

                elif convType == 'rag-with-transformation':  # rag-with-transformation
                    msg = run_rag_with_transformation(connectionId, requestId, text)
                
                elif convType == 'data-enrichment-agent': 
                    msg = run_data_enrichment_agent(connectionId, requestId, text)
                    
                elif convType == "translation":
                    msg = translate_text(chat, text) 
                
                elif convType == "grammar":
                    msg = check_grammer(chat, text)  
                
                else:
                    msg = general_conversation(connectionId, requestId, chat, text)  
                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                
                if reference_docs:
                    reference = get_references(reference_docs)
                                        
        elif type == 'document':
            isTyping(connectionId, requestId, "")
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                if not convType == "bedrock-agent":
                    docs = load_csv_document(object)
                    contexts = []
                    for doc in docs:
                        contexts.append(doc.page_content)
                    print('contexts: ', contexts)
                
                    msg = get_summary(chat, contexts)
                else: # agent
                    text = body                    
                    
                    #if text and isKorean(text)==False:
                    #    text += f"\n\nEnsure that the graph is clearly labeled and easy to read. \
#After generating the graph, provide a brief interpretation of the results, highlighting \
#which category has the highest total spend and any other notable observations."
                    #else:
                    text += f"그래프에 명확한 레이블을 지정하고 읽기 쉽도록 하세요. \
그래프를 생성한 후에는 결과를 간략하게 해석하여 값이 가장 높거나 낮은 범주와 다른 주목할 만한 관찰 사항을 강조하세요."
                    
                    print('text: ', text)
                    
                    s3Location = f"s3://{s3_bucket}/{s3_prefix}/{object}"
                    print('s3Location: ', s3Location)
                    
                    sessionState = {
                        "files": [
                            {
                                "name": object,
                                "source": {
                                    "s3Location": {
                                        "uri": s3Location
                                    },
                                    "sourceType": 'S3'
                                },
                                "useCase": "CODE_INTERPRETER"
                            }
                        ]
                    }
                    msg = run_bedrock_agent(text, connectionId, requestId, userId, sessionState)

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
                                'url': path+doc_prefix+parse.quote(object)
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
                text = extract_text(img_base64)
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
            # raise Exception ("Not able to write into dynamodb")         
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
