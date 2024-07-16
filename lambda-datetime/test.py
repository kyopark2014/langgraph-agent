import numpy as np
import time
from lambda_function import lambda_handler  

def load_event():
    json_data = {
        "user_id": "user1234",
        "request_id": "test1234",
        "request_time": "2023-10-08 18:01:45",        
        "type": "text",
        "body": "Building a website can be done in 10 simple steps.",
        "convType": "normal"
    }
    return json_data

def main():
    start = time.time()

    # load samples
    event = load_event()

    # run
    results = lambda_handler(event,"")  
    
    # results
    print(results['statusCode'])
    print(results['msg'])

    print('Elapsed time: %0.2fs' % (time.time()-start))   

if __name__ == '__main__':
    main()
