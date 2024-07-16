import json
import boto3
import os
import uuid
import random

sqs_client = boto3.client('sqs')
sqsUrl = os.environ.get('sqsUrl')
sqsFifoUrl = json.loads(os.environ.get('sqsFifoUrl'))
print('sqsFifoUrl: ', json.dumps(sqsFifoUrl))

nqueue = os.environ.get('nqueue')

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for i, record in enumerate(event['Records']):
        receiptHandle = record['receiptHandle']
        print("receiptHandle: ", receiptHandle)
        
        body = record['body']
        print("body: ", json.loads(body))
        
        # idx = i % int(nqueue)
        idx = random.randrange(0,int(nqueue))
        print('idx: ', idx)
        
        eventId = str(uuid.uuid1())
        print('eventId: ', eventId)
        
        # push to SQS
        try:
            print('sqsFifoUrl: ', sqsFifoUrl[idx])            
            #sqs_client.send_message(  # standard 
            #    DelaySeconds=0,
            #    MessageAttributes={},
            #    MessageBody=body,
            #    QueueUrl=sqsFifoUrl[idx])
            #)
            
            sqs_client.send_message(  # fifo
                QueueUrl=sqsFifoUrl[idx], 
                MessageAttributes={},
                MessageDeduplicationId=eventId,
                MessageGroupId="putEvent",
                MessageBody=body
            )
            print('Successfully push the queue message: ', body)
            
            # delete queue
            try:
                sqs_client.delete_message(QueueUrl=sqsUrl, ReceiptHandle=receiptHandle)
            except Exception as e:        
                print('Fail to delete the queue message: ', e)

        except Exception as e:        
            print('Fail to push the queue message: ', e)
        
    return {
        'statusCode': 200
    }