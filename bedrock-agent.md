# Bedrock Agent

## CDK Deployment

Llambda

```python
// create the lambda for the agent - this is the lambda that determines
// what the prompt looks like with regards to mapping to the schema
const actionGroupAgentLambda: nodeLambda.NodejsFunction =
  new nodeLambda.NodejsFunction(this, 'AgentLambda', {
    functionName: 'action-group-executor',
    runtime: lambda.Runtime.NODEJS_20_X,
    entry: path.join(
      __dirname,
      './src/adapters/primary/action-group-executor/action-group-executor.adapter.ts'
    ),
    memorySize: 1024,
    handler: 'handler',
    timeout: cdk.Duration.minutes(5),
    description: 'action group lambda function',
    architecture: lambda.Architecture.ARM_64,
    tracing: lambda.Tracing.ACTIVE,
    bundling: {
      minify: true,
    },
    environment: {
      ...lambdaConfig,
    },
  });
```

```typescript
// create the bedrock agent
const agent = new bedrock.Agent(this, 'BedrockAgent', {
  name: 'Agent',
  description: 'The agent for hotels, Spa and golf bookings.',
  foundationModel: bedrock.BedrockFoundationModel.ANTHROPIC_CLAUDE_V2,
  instruction:
    'Please help our customers to book hotel rooms, spa sessions and golf bookings; whilst providing them with any special offers depending on the day and booking type, make them aware of any opening times or prices before they complete the booking, and also take into consideration our hotel policies.',
  idleSessionTTL: cdk.Duration.minutes(10),
  knowledgeBases: [kb],
  shouldPrepareAgent: true,
  aliasName: 'Agent',
});
```

Action Group을 정의합니다. 여기서 Open API schema로 [api-schema.json](https://github.com/kyopark2014/llm-agent/blob/main/schema/api-schema.json)을 정의합니다.

```typescirpt
// add the action group for making bookings
new bedrock.AgentActionGroup(this, 'AgentActionGroup', {
  actionGroupName: 'agent-action-group',
  description: 'The action group for making a booking',
  agent: agent,
  apiSchema: bedrock.S3ApiSchema.fromAsset(
    path.join(__dirname, './schema/api-schema.json')
  ),
  actionGroupState: 'ENABLED',
  actionGroupExecutor: actionGroupAgentLambda,
  shouldPrepareAgent: true,
});
```
