
from flask import Flask, render_template, request, Markup,redirect
import numpy as np
import pandas as pd
import pickle
import os
import io
import torch
from torchvision import transforms
from PIL import Image
from models.model import ResNet9
from utils.disease import diseases

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

app=Flask(__name__)
fertilizer_predict_model_path='models/classifier.pkl'
crop_predict_path="models/cropv2.pkl"
#####################################################################
disease_model_path = 'models/plant_disease_model.pth'


disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction
#####################################################################

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
@app.route('/disease')
def disease_recommendation():
    return render_template('disease.html')
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
    
@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html')
        
        img = file.read()

         

        prediction = predict_image(img)
        prediction = Markup(str(diseases[prediction]))
        

        return render_template('disease-result.html', prediction=prediction)
        
    


if __name__ == "__main__":
    app.run(debug=True)
