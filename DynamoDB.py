import boto3
from botocore.config import Config
from datetime import datetime
from boto3.dynamodb.conditions import Key, Attr


def table_function():
    my_config = Config(
    region_name = 'us-west-2'
    )
    
    # Get the service resource.
    
    session = boto3.Session(
        aws_access_key_id='AKIAVFPETSJYIUA66ASG',
        aws_secret_access_key='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'
    )
    
    dynamodb = session.resource('dynamodb', config=my_config)
    table = dynamodb.Table('Lab3scores')
    
    return table 

def post_score(log_table, feature_string, class_string, prob_string, label):
    now = datetime.now()
    current_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]

    response = log_table.put_item(
       Item={
            'partition_key': current_time,
            'sort_key': "abc",
            'Features': feature_string,
            'Class' : class_string,
            'Probability' : prob_string,
            'Label': label
            }
    )
    return response




