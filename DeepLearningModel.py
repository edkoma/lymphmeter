#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:41:49 2017

@author: sahana
"""

from flask import Flask
from flask_cors import CORS
from flask import request
import pandas as pd
from keras.models import load_model
import numpy as np


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
	
    # X = np.empty((len(request.args), 1), dtype=int)
    X = []
    for key in request.args:
        # print(key, request.args[key])
        X.append(int(request.args[key]))

    X = np.array(X)
    X = np.transpose(X)
    X = np.reshape(X,(1,31))   
    
    model = load_model('model.h5')
    prediction = model.predict(X)
    
    # Grab just the first element of the first prediction (since we only have one)
    prediction = prediction[0][0]

    # Re-scale the data from the 0-to-1 range back to dollars
    # These constants are from when the data was originally scaled down to the 0-to-1 range
    # prediction = round(prediction)

    return(str(prediction))
    # return("Lymphmeter Prediction for Proposed data : ")


if __name__ == '__main__':
    app.run(debug=True)

