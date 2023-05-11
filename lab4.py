from flask import Flask
from flask import request
from flask import Response
from flask import abort
from lab3 import setup, build, train, score
from DynamoDB import post_score, table_function
from datetime import datetime

app = Flask(__name__)

has_setup = False
has_build = False
has_train = False
has_score = False
setup_date_time = 0
build_date_time = 0
train_date_time = 0
score_date_time = 0
score_pred = 0
pred_class = 0


# the minimal Flask application
@app.route('/')
def index():
    return Response( '<h1>Hello, World!</h1>', status=201 )


# bind multiple URL for one view function
@app.route('/hi')
@app.route('/hello')
def say_hello():
    return Response('<h1>Hello, Flask!</h1>', status=201)


# dynamic route, URL variable default
@app.route('/greet', defaults={'name': 'Programmer'})
@app.route('/greet/<name>')
def greet(name):
    ret = 'Hello ' + name
    print(ret)
    return Response(ret, status=201)

@app.route('/my_resource', methods=['POST', 'PUT'])
def post_example():
    inputs = request.args.get('res_name')
    print(inputs)
    if ( inputs != 'credit'):
        abort(404)
        return("quitting")
    else:
        return Response('You passed in res_name = ' + inputs, status=200 )

@app.route('/credit/context', methods=['POST', 'PUT'])
def do_setup():
    global has_setup, setup_date_time
    file = request.args.get("data")
    setup(file)
    setup_date_time = datetime.now()
    has_setup = True
    
    return Response("Setup done", status = 200)

@app.route('/credit/model', methods=['POST'])
def do_build():
    global has_build, build_date_time
    if has_setup:
        
        build()
        has_build = True
        build_date_time = datetime.now()
        
        return Response("Build done", status = 200)
    
    else:
        return Response("Build failed. You need to setup with the file first", status=403)
        

@app.route('/credit/model', methods=['PUT'])
#@app.route('/model_training/<training_type>', methods = ["POST", "PUT"])
def do_training():
    
    traintype = request.args.get("type")
    global has_train, train_date_time 
    
    #To check if the setup and build are already completed
    if has_setup and has_build:
        if traintype == "whole":
            train("whole")
            has_train = True
            train_date_time = datetime.now()
            return Response("Trained successfully. Predicted using the whole dataset", status = 200)
        elif traintype == "male":
            train("male")
            has_train = True
            train_date_time = datetime.now()
            return Response("Trained successfully. Predicted on only the male dataset", status = 200)
        elif traintype == "female":
            train("female")
            has_train = True
            train_date_time = datetime.now()
            
            return Response("Trained successfully. Predicted on only the female dataset", status = 200)
        else:
            return Response("Training failed. You need to pass the right type of prediction you want(i.e whole, male or female)", status=400)
            
        
        
    
    else:
        return Response("Training failed. Setup or Build, or both not yet completed", status =403)

@app.route('/credit/model', methods=['GET'])
def do_score():
    global score_date_time, score_pred, pred_class
    
    mode = request.args.get("mode")
    
    if mode == "No post":        
        prediction_list = ["Good", "Bad"]
        if has_setup and has_build and has_train:
            #print("seen")
            prediction, raw_pred =  score()
            print("The prediction is", prediction)
            score_date_time = datetime.now()
            print("Scoring end")
            score_pred = prediction
            pred_class = prediction_list[prediction]
            return Response('''<h3>
                Raw Prediction is {}. Class prediction is {}
                </h3>'''.format(prediction,prediction_list[prediction]), status = 200)
                
        else:
            return Response("Scoring failed. Setup or Build or Train  or either 2 or all of them not yet completed", status= 403)
        
    elif mode == "post":
        if has_setup and has_build and has_train:
            age = request.args.get('Age')
            sex = request.args.get('Sex')
            job = request.args.get('Job')
            housing = request.args.get('Housing')
            savings = request.args.get('Saving acounts')
            checking = request.args.get('Checking account')
            amount =  request.args.get('Credit amount')
            duration = request.args.get('Duration')
            purpose = request.args.get('Purpose')
            label = request.args.get("Risk")
            
            prediction_list = ["Good", "Bad"]
            
            prediction, raw_pred =  score()
            print("prediction", prediction)
            table = table_function()
            features = str([ int(age), sex, int(job), housing, savings, checking, int(amount), int(duration), purpose  ])
            response = post_score(table, features, str(prediction), str(raw_pred), label)
            
            score_date_time = datetime.now()
        
            score_pred = prediction
            pred_class = prediction_list[prediction]
            return Response('''<h3>
                Raw Prediction is {}. Class prediction is {},and at {}, and the prediction has been sent to DynamoDB
                </h3>'''.format(prediction,prediction_list[prediction],score_date_time), status = 200)
                
        else:
            return Response("Scoring failed. Setup or Build or Train  or either 2 or all of them not yet completed", status= 403)
        
    else:
        return Response("Scoring failed. mode has to be 'post' or 'No post'", status= 403)
        

@app.route('/logs', methods=['GET'])
def log():
    global setup_date_time,build_date_time,train_date_time,score_date_time
    return Response('''<h3>The last logs are:</br> setup at {},</br>build at {},</br> 
         train at {},</br> score at {},</br>
         with a score of {} </br>
         and the prediction is {}</h3>'''.format(setup_date_time,build_date_time,train_date_time,score_date_time,score_pred, pred_class), status = 200)
    
    
    
        

    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
