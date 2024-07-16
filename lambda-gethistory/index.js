const aws = require('aws-sdk');

var dynamo = new aws.DynamoDB();
const tableName = process.env.tableName;

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));

    const userId = event['userId'];
    const allowTime = event['allowTime'];

    console.log('userId: ', userId)
    console.log('allowTime: ', allowTime)

    let queryParams = {
        TableName: tableName,
        KeyConditionExpression: "user_id = :userId and request_time > :allowTime",
        ExpressionAttributeValues: {
            ":userId": {'S': userId},
            ":allowTime": {'S': allowTime}
        }
    };
    
    try {
        let result = await dynamo.query(queryParams).promise();
    
        console.log('History: ', JSON.stringify(result));    

        let history = [];
        for(let item of result['Items']) {
            console.log('item: ', item);
            let request_time = item['request_time']['S'];
            let request_id = item['request_id']['S'];
            let body = item['body']['S'];
            let msg = item['msg']['S'];
            let type = item['type']['S'];

            history.push({
                'request_time': request_time,
                'request_id': request_id,
                'type': type,
                'body': body,
                'msg': msg,
            });
        }

        console.log('Json History: ', history);
        const response = {
            statusCode: 200,
            msg: JSON.stringify(history)
        };
        return response;  
          
    } catch (error) {
        console.log(error);

        const response = {
            statusCode: 500,
            msg: error
        };
        return response;  
    } 
};