# SMARTAGRO 🌱🌾
SMARTAGRO is an innovative website that leverages the power of Machine Learning (ML) and Deep Learning (DL) to provide recommendations for crop selection, fertilizer usage, and disease identification in crops.


## Introduction

The agricultural sector plays a vital role in the economic growth of a country. In countries like India, where a significant portion of the population depends on agriculture for their livelihood, it is crucial to enhance farming practices and maximize yield. With advancements in technology, such as ML and DL, we have the opportunity to revolutionize the agricultural industry.

Harvestify aims to provide a user-friendly website that offers the following applications:

- Crop Recommendation: By inputting soil data, users can receive recommendations on the most suitable crop to grow based on the provided information.

- Fertilizer Recommendation: Users can input soil data and specify the type of crop they are growing. The application will analyze the soil's nutrient content and suggest improvements or specific fertilizers to optimize crop growth.

- Plant Disease Prediction: Users can upload an image of a diseased plant leaf. The application will predict the disease affecting the plant and provide information about the disease along with preventive measures or treatment options.

## Data Source

The data used in this project includes the following datasets:

- Crop Recommendation dataset 
- Fertilizer Suggestion dataset 
- Disease Detection dataset

## Notebooks

Corresponding code for this project is available on Kaggle Notebooks:

- [Crop Recommendation](link_to_crop_recommendation_notebook)
- [Disease Detection](link_to_disease_detection_notebook)

## Technologies Used

This project incorporates several modern technologies to deliver its functionalities:

- **Python**: The core programming language used for building the project.
- **Flask**: A lightweight and flexible web framework in Python for creating the website and handling HTTP requests.
- **HTML/CSS**: The standard markup language and styling sheet used for designing and structuring the web pages.
- **TensorFlow**: An open-source machine learning framework used for developing and training deep learning models.
- **Scikit-learn**: A popular machine learning library in Python that provides a wide range of tools for data preprocessing, modeling, and evaluation.



## How to Use

The SMARTARGO website provides the following functionalities:

### Crop Recommendation System

1. Enter the corresponding nutrient values of your soil, state, and city. Please note that the N-P-K (Nitrogen-Phosphorus-Potassium) values should be entered as ratios. 
2. When entering the city name, use common city names as remote cities/towns may not be available in the Weather API used for fetching humidity and temperature data.

### Fertilizer Suggestion System

1. Enter the nutrient contents of your soil and specify the crop you want to grow.
2. The algorithm will analyze the soil's nutrient composition and provide suggestions for buying fertilizers based on nutrient deficiencies or excesses.

### Disease Detection System

1. Upload an image of a plant leaf.
2. The algorithm will identify the crop type and determine whether the leaf is diseased or healthy.
3. If the leaf is diseased, the algorithm will provide information about the disease and suggest preventive measures or treatment options.

Please note that the system currently supports a limited number of crops.


## How to Run Locally

Before following the steps below, ensure that you have Git, Anaconda, or Miniconda installed on your system.

1. Clone the complete project using the command: `git clone https://github.com/Gladiator07/Harvestify.git` or download and unzip the code.

2. To download the updated code used for deployment, clone the deploy branch with the following command: `git clone -b deploy https://github.com/Gladiator07/Harvestify.git`.
   - The deploy branch contains only the code required for deploying the app. For the code used for training the models and data preparation, access the master branch.
   - It is recommended to clone the deploy branch to run the project locally. The following steps assume you have the deploy branch cloned.

3. Once the project is cloned, open Anaconda Prompt in the directory where the project is located and run the following commands:
   ```python
   cd app
   conda create -n "You-can-write-anything-here" python=3.6.12
   conda activate "You-can-write-anything-here"
   pip install -r requirement.txt
   python app.py

     

5. Open the provided localhost URL after running `app.py` and use the project locally in your web browser.

## Demo

### Crop Recommendation System

[Demo](link_to_crop_recommendation_demo)

### Fertilizer Suggestion System

[Demo](link_to_fertilizer_suggestion_demo)

### Disease Detection System

[Demo](link_to_disease_detection_demo)

Feel free to explore the functionalities and enjoy using the SMARTARGO website!


