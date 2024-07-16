const aws = require('aws-sdk');

var dynamo = new aws.DynamoDB();
let tableName = process.env.tableName;
let indexName = process.env.indexName;

//tableName = 'db-call-log-for-bedrock-with-simple';
//indexName = 'index-type-for-bedrock-with-simple';

console.log('tableName: ', tableName);
console.log('indexName: ', indexName);

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));

    let requestId = event['request_id'];
    console.log('requestId: ', requestId);    
    
    let msg = "";
    let isCompleted = false;
    let queryParams = {
        TableName: tableName,
        IndexName: indexName, 
        KeyConditionExpression: "request_id = :requestId",
        ExpressionAttributeValues: {
            ":requestId": {'S': requestId}
        }
    };
    
    let response;
    try {
        let result = await dynamo.query(queryParams).promise();    
        console.log('result: ', JSON.stringify(result));

        isCompleted = true;
        if(result['Items'][0])
            msg = result['Items'][0]['msg']['S'];

        console.log('msg: ', msg);   
        response = {
            statusCode: 200,
            msg: msg
        };
    } catch (error) {
        console.log(error);
        response = {
            statusCode: 500,
            msg: error
        };
    }     
    
    function wait(){
        return new Promise((resolve, reject) => {
            if(!isCompleted) {
                setTimeout(() => resolve("wait..."), 1000);
            }
            else {
                setTimeout(() => resolve("done..."), 0);
            }
        });
    }
    console.log(await wait());
    console.log(await wait());
    console.log(await wait());
    console.log(await wait());
    console.log(await wait());
    
    return response;
};

