
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import pickle
import os
app=Flask(__name__)
fertilizer_predict_model_path='models/fertilizer.pkl'
crop_predict_path="models/cropv2.pkl"


diseases_model_path='models/diseasemodel.pth'


folder = os.path.join('static','cropimages')
app.config['UPLOAD_FOLDER'] = folder


modelfertilizer = pickle.load(open(fertilizer_predict_model_path, 'rb'))
modelcrop=pickle.load(open(crop_predict_path,'rb'))

soil_types = np.array(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
crop_types = np.array(['Maize', 'Sugarcane' ,'Cotton' ,'Tobacco' ,'Paddy' ,'Barley', 'Wheat', 'Millets',
 'Oil seeds' ,'Pulses', 'Ground Nuts'])
@app.route("/")
def home():
    return render_template('index.html')
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html')
@app.route('/crop')
def crop_recommendation():
    return render_template('crop.html')

@app.route("/fertilizer_predict",methods=['POST'])
def fertilizer_predict():
    if request.method == 'POST':
        temperature = int(request.form['Temparature'])
        humidity = int(request.form['Humidity'])
        moisture = int(request.form['Moisture'])
        soil = request.form['Soil']
        soil_index = np.where(soil_types == soil)[0][0]
        crop = request.form['cropname']
        crop_index = np.where(crop_types == crop)[0][0]
        nitrogen = int(request.form['nitrogen'])
        phosphorous = int(request.form['phosphorous'])
        potassium = int(request.form['pottasium'])
        
        data = np.array([[temperature, humidity, moisture,soil_index,crop_index,nitrogen, phosphorous, potassium]])
        ans = modelfertilizer.predict(data)
        if ans[0] == 0:
            prediction="10-26-26"
        elif ans[0] == 1:
            prediction="14-35-14"

        elif ans[0] == 2:
            prediction="17-17-17"

        elif ans[0] == 3:
            prediction="20-20"
        
        elif ans[0] == 4:
            prediction="28-28"

        elif ans[0] == 5:
            prediction="DAP"

        else:
            prediction="Urea"
    return render_template('fertilizer-result.html', prediction=prediction) 

@app.route("/crop_predict",methods=['POST'])
def crop_predict():
    if request.method == 'POST':
        nitrogen = int(request.form['nitrogen'])
        phosphorous = int(request.form['phosphorous'])
        potassium = int(request.form['pottasium'])
        temperature = float(request.form['Temparature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['ph'])
       
        rainfall = float(request.form['Rainfall'])
        data = np.array([[nitrogen,phosphorous,potassium,temperature,humidity,ph,rainfall]])
        ans=modelcrop.predict(data)
        imagepath = os.path.join(app.config['UPLOAD_FOLDER'], ans[0]+'.jpg')
        return render_template('crop-result.html',prediction=ans[0],image=imagepath)
if __name__ == "__main__":
    app.run(debug=True)
