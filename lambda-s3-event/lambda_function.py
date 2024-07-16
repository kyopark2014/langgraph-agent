import json
import boto3
import os
import datetime
import uuid

sqs_client = boto3.client('sqs')
sqsUrl = os.environ.get('queueS3event')

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print('bucket: ', bucket)
        print('key: ', key)
        
        eventId = str(uuid.uuid1())
        print('eventId: ', eventId)
        
        s3EventInfo = {
            'event_id': eventId,
            'event_timestamp': record['eventTime'],
            'bucket': bucket,
            'key': key,
            'type': record['eventName']
        }
        
        # push to SQS
        try:            
            sqs_client.send_message(  # standard 
                DelaySeconds=0,
                MessageAttributes={},
                MessageBody=json.dumps(s3EventInfo),
                QueueUrl=sqsUrl
            )
            
            #sqs_client.send_message(  # fifo
            #    QueueUrl=sqsUrl, 
            #    MessageAttributes={},
            #    MessageDeduplicationId=eventId,
            #    MessageGroupId="putEvent",
            #    MessageBody=json.dumps(s3EventInfo)
            #)
            print('Successfully push the queue message: ', json.dumps(s3EventInfo))

        except Exception as e:        
            print('Fail to push the queue message: ', e)
        
    return {
        'statusCode': 200
    }