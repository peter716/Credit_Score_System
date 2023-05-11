import csv
import requests
import sys

train_data = { 'training_data': 'credit_train.csv' }
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

with open('credit_score_clean.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count=0
    for row in csv_reader:
        if line_count == 0:
            # print(row)
            heads = row
            line_count+=1
        else:
            req_data = {heads[i]: row[i] for i in range(1,len(row))}
            req_data["mode"] = "post"
            print(req_data)
            r = requests.get("http://localhost:4000/credit/model", params=req_data)
            print(r.text)
            line_count+=1
