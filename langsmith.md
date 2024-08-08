# LangSmith

[LangSmith](https://www.langchain.com/langsmith)에 접속하여 가입후 API key를 생성합니다. [LangSmith 상세](https://python.langchain.com/v0.1/docs/langsmith/)에는 좀더 상세한 정보가 있습니다.

Lambda등에서 로그를 보고 싶은 경우에 아래와 같이 설정합니다. 

## 설정

아래와 같이 LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT을 설정하고 [LangSmith](https://www.langchain.com/langsmith)에 접속하면 동작현황을 확인할 수 있습니다.

```python
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<project-name>
```

#### Lambda에서 LangSmith를 사용하기 위한 설정

[Secrets manager console](https://ap-northeast-2.console.aws.amazon.com/secretsmanager/listsecrets?region=ap-northeast-2)에서 langsmithapikey을 선택한 후에, [Secret value]에서 [Retrieve secret value]을 선택한 후에 langsmith_api_key에 적절한 key를 입력합니다. LangSmith의 api key는 [LangSmith](https://www.langchain.com/langsmith)에서 발급 받습니다.
