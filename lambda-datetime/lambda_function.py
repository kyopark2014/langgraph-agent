import json
from pytz import timezone
import datetime

def lambda_handler(event, context):
    print('event: ', event)
    
    format: str = f"%Y-%m-%d %H:%M:%S"
    
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    print('timestr: ', timestr)

    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps({'timestr': timestr})
    }