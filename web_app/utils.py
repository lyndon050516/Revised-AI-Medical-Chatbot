import pandas as pd 
import json
from io import BytesIO
import os 
import requests 
from PIL import Image
import base64

urls = {
    "predict_data": "http://18.216.177.95/predict_data", 
    "generate_mask": "http://18.216.177.95/generate_mask",
    "ask_chatbot": "http://18.216.177.95/ask_chatbot"
}

def send_data_and_get_response(data, url = urls['predict_data']):
    """takes in a dict, returns the response as a dict"""
    response = requests.post(url, json=data)
 
    return response.json()

def send_image_and_get_response(image, url = urls['generate_mask']): 
    """takes in an image, returns the prediction & overlayed image & original image"""
    response = requests.post(url, files={'file': image})
    response.raise_for_status()  # Ensure the request was successful
    response_data = response.json()
    
    # Extract and decode base64 images
    prediction = response_data['prediction']
    overlayed_image_base64 = response_data['overlayed_image']
    original_image_base64 = response_data['original_image']

    overlayed_image = Image.open(BytesIO(base64.b64decode(overlayed_image_base64)))
    original_image = Image.open(BytesIO(base64.b64decode(original_image_base64)))
    
    return prediction, overlayed_image, original_image


def ask_chatbot_and_get_response(question, url = urls['ask_chatbot']):
    """takes in a question and history, returns the response as a string"""
    response = requests.post(url, json={"question": question})
    response.raise_for_status()  # Ensure the request was successful
    response_data = response.json()
    return response_data['response']
    


def convert_to_json(file):
    # df = pd.read_csv(file)
    # file.seek(0)  # Reset file pointer after reading
    json_data = file.to_json(orient='records')
    return json_data

def process_image(): 
    pass 


def json_to_csv(json_string, output_folder, output_filename): 
    """used to generate example data for the app"""
    data = json.loads(json_string)
    df = pd.DataFrame([data])  # Treat the single JSON object as a list of one item

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save DataFrame to CSV file
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path, index=False)
    print(f"CSV file saved to {output_path}")
    