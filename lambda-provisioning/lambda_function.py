import json
import boto3
import os

wss_url = os.environ.get('wss_url')

def lambda_handler(event, context):
    print('event: ', event)

    print('wss_url: ', wss_url)
    
    return {
        'statusCode': 200,
        'info': json.dumps({
            'wss_url': wss_url
        })
    }
