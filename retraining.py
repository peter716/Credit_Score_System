import boto3
from botocore.config import Config
from boto3.dynamodb.conditions import Key, Attr
import time
import csv
from datetime import datetime
import requests
import sys
import ast
import shutil



my_config = Config(
    region_name = 'us-west-2'
)

# Get the service resource.

session = boto3.Session(
    aws_access_key_id='AKIAVFPETSJYIUA66ASG',
    aws_secret_access_key='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'
)

dynamodb = session.resource('dynamodb', config=my_config)
update_table = dynamodb.Table('Lab3retraining')


def build_training_update():
    list_of_lists = []
    response = update_table.scan()
    items = response['Items']
    print(items)
    for item in items:
        # build the training feature set
        features_str = item['Features']
        features = ast.literal_eval(features_str)
        features.append(item['Label'])
        features.insert(0, item['partition_key'])
        print(features)
        list_of_lists.append( features )

    # copy original training data to new training_file_name.csv
    # check https://docs.python.org/3/library/shutil.html for info on how to do the file system copy!

    shutil.copyfile("credit_train.csv","new_training_file.csv")

    with open("new_training_file.csv", "a") as f:
        wr = csv.writer(f)
        wr.writerows( list_of_lists )

    return 

# use the example REST invocations in the model driver python script to then reprocess your updated training data.
# be sure to do the "context" step as well as the retraining step
# then run a set of scoring tests to check the service is still operational

def do_model_update():
    # use the pattern from model_drive.py to pre-process and retrain you model, calling the credit service using the REST API
    

    train_data = { 'training_data': 'new_training_file.csv' }
    r = requests.put("http://localhost:4000/credit/context", params=train_data)
    print(r.text)
    if ( r.status_code != 201 ):
        print("Exiting")
        sys.exit()
    
    r = requests.post("http://localhost:4000/credit/model")
    print(r.text)

    
    train_type = {"type":"whole"}
    r = requests.put("http://localhost:4000/credit/model", params= train_type)
    print(r.text)

    return

build_training_update()
do_model_update()