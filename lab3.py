import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout
from pandas.api.types import is_string_dtype
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import copy
from flask import Flask
from flask import request
from flask import Response
from flask import abort

import boto3
from botocore.config import Config
import datetime



min_values = {}
max_values = {}
score_template = {}
data = 0
data_train = 0
data_valid = 0
x_train = 0
y_train = 0
x_train_male = 0
y_train_male = 0
x_val_male = 0
y_val_male = 0
x_val = 0
y_val = 0
model = 0

print('starting up credit score app')


def normalize(col, df, training):
    global max_values, min_values

    result = df.copy()
    if training == True:
        max_values[col] = df.max()
        min_values[col] = df.min()
    result = (df - min_values[col]) / (max_values[col] - min_values[col])
    return result

def fixup(data, training):
    labels = data.columns
    # lets go through column 2 column
    for col in labels:
        if is_string_dtype(data[col]):
            if col == 'Risk':
                # we want 'Risk' to be a binary variable
                data[col] = pd.factorize(data[col])[0]
                continue
            # the other categorical columns should be one-hot encoded
            data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
            data.drop(col, axis=1, inplace=True)
        else:
            data[col] = normalize(col, data[col], training)
    return data


def setup(training_data):
    global x_train, y_train, x_val, y_val, score_template, data, data_train, data_valid, min_values, max_values
    global x_train_male, y_train_male, x_val_male, y_val_male, x_train_female, y_train_female, x_val_female, y_val_female
    print('Setup credit score data')

    min_values = {}
    max_values = {}
    score_template = {}
    data = 0
    data_train = 0
    data_valid = 0
    x_train = 0
    y_train = 0
    x_val = 0
    y_val = 0
    x_train_male = 0
    y_train_male = 0
    x_val_male = 0
    y_val_male = 0
    
    data = pd.read_csv(training_data,index_col=0,sep=',')
    data = fixup(data, True)
    print(data["Sex_male"].head())
    print(data["Sex_female"].head())

    # move 'Risk' back to the end of the df
    data = data[[c for c in data if c not in ['Risk']] + ['Risk']]

    for col in data.columns:
        if col != 'Risk':
            score_template[col] = 0

    data_train = data.iloc[int(0.2*(len(data))):]
    data_valid = data.iloc[:int(0.2*(len(data)))]
    x_train = data_train.iloc[:,:-1]
    y_train = data_train.iloc[:,-1]
    x_train_male = x_train.loc[x_train["Sex_male"]== 1]
    y_train_male = y_train.loc[x_train["Sex_male"]== 1]
    x_train_female = x_train.loc[x_train["Sex_female"]==1]
    y_train_female = y_train.loc[x_train["Sex_female"] == 1]
    #print(x_train.head())
    x_val = data_valid.iloc[:,:-1]
    y_val = data_valid.iloc[:,-1]
    x_val_male = x_val.loc[x_val["Sex_male"] == 1]
    y_val_male = y_val.loc[x_val["Sex_male"] == 1]
    x_val_female = x_val.loc[x_val["Sex_female"] == 1]
    y_val_female = y_val.loc[x_val["Sex_female"] == 1]


    return "Setup done"

def build():
    global model

    sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)

    model = Sequential()
    model.add(Dense(units=50, activation='tanh', input_dim=24, kernel_initializer='glorot_normal', bias_initializer='zeros'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.35))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

def train(predict_type):
    model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values), epochs=30, batch_size=128)
    #model.fit(x_train_male.values, y_train_male.values, validation_data=(x_val_male.values, y_val_male.values), epochs=30, batch_size=128)
    #model.fit(x_train_female.values, y_train_female.values, validation_data=(x_val_female.values, y_val_female.values), epochs=30, batch_size=128)
    
    if predict_type == "whole":

        y_pred = (model.predict( x_val.values ) > 0.5).astype("int32")
        #y_pred = (model.predict( x_val_male.values ) > 0.5).astype("int32")
        #y_pred = (model.predict( x_val_female.values ) > 0.5).astype("int32")
    
        sns.heatmap(confusion_matrix(y_val,y_pred),annot=True,fmt='.5g')
        print('Confusion matrix on validation data is {}'.format(confusion_matrix(y_val, y_pred)))
        #print("Accuracy score",accuracy_score(y_val,y_pred))
        #print('Confusion matrix on validation data is {}'.format(confusion_matrix(y_val_male, y_pred)))
        #print('Confusion matrix on validation data is {}'.format(confusion_matrix(y_val_female, y_pred)))
        #print('Precision Score on validation data is {}'.format(precision_score(y_val_male, y_pred, average='weighted')))
        #print('Precision Score on validation data is {}'.format(precision_score(y_val_female, y_pred, average='weighted')))
        print('Precision Score on validation data is {}'.format(precision_score(y_val, y_pred, average= None)))
        
    if predict_type == "male":
        
        y_pred = (model.predict( x_val_male.values ) > 0.5).astype("int32")
        sns.heatmap(confusion_matrix(y_val_male,y_pred),annot=True,fmt='.5g')
        print('Confusion matrix on validation data for male is {}'.format(confusion_matrix(y_val_male, y_pred)))
        print('Precision Score on validation data is for male {}'.format(precision_score(y_val_male, y_pred, average=None)))
        
    if predict_type == "female":
        
        y_pred = (model.predict( x_val_female.values ) > 0.5).astype("int32")
        sns.heatmap(confusion_matrix(y_val_female,y_pred),annot=True,fmt='.5g')
        print('Confusion matrix on validation data for female is {}'.format(confusion_matrix(y_val_female, y_pred)))
        print('Precision Score on validation data for females is {}'.format(precision_score(y_val_female, y_pred, average= None )))
        
    
        

    return "Train done"


def score():
    age = request.args.get('Age')
    sex = request.args.get('Sex')
    job = request.args.get('Job')
    housing = request.args.get('Housing')
    savings = request.args.get('Saving accounts')
    checking = request.args.get('Checking account')
    amount =  request.args.get('Credit amount')
    duration = request.args.get('Duration')
    purpose = request.args.get('Purpose')

    x_score_array = [[ int(age), sex, int(job), housing, savings, checking, int(amount), int(duration), purpose  ]]
    x_score_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    df = pd.DataFrame( x_score_array, columns = x_score_cols )
    x_score = fixup( df, False )

    score_instance = score_template.copy()
    for col in x_score.columns:
        score_instance[col] = x_score.get(col)[0]
    print( score_instance )

    x_score_list = list(score_instance.values())
    x_score_args = [ x_score_list ]
    print(x_score_args)

    raw_pred = model.predict( x_score_args )
    y_pred = ( raw_pred > 0.5).astype("int32")

    print( " prediction is ", y_pred )
    print("Score done")

    return y_pred[0][0], raw_pred[0][0]





