# page for turning ml models into api
import io 
import pickle 
import numpy as np 
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import base64

import tensorflow as tf 
from keras.layers import Layer, Conv2D, Dropout, BatchNormalization, ReLU
from keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow.image as tfi 

import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# from chatbot import ask_question
from typing import Union


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


# define class items structure for quantitative 
class Features(BaseModel): # only included highly correlated features 
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float
    


# load in quantitative model 
with open('app/models/naive_bayes_model.pkl', 'rb') as f: 
    quantitative_model = pickle.load(f)


# load in segmentation model 
class ConvBlock(Layer):
    def __init__(self, filters=256, kernel_size=3, use_bias=False, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate=dilation_rate, use_bias=use_bias, kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(), 
            Dropout(0.2)
        ])
    def call(self, X): 
        return self.net(X)        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "kernel_size":self.kernel_size,
            "use_bias":self.use_bias,
            "dilation_rate":self.dilation_rate
    }
    
segmentation_model = tf.keras.models.load_model(
    'app/models/DeepLabV3-Plus.h5', 
    custom_objects={'ConvBlock': ConvBlock}, 
    compile=False
)



# load in image classification model 
classification_model_mask = torch.load('app/models/Resnet152_fineTuning_with_mask.pth', map_location=torch.device('cpu'), weights_only=False)
classification_model_mask.eval()



# predict data: quantitative model 
@app.post('/predict_data')
async def predict_data(item:Features): 
    class_names = ['benign', 'malignant']
    df = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    relevant_features = ['radius_mean', 'texture_mean', 'perimeter_mean',
                        'area_mean', 'smoothness_mean', 'compactness_mean',
                        'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                        'radius_se', 'perimeter_se', 'area_se',
                        'compactness_se', 'concavity_se', 'concave_points_se',
                        'radius_worst', 'texture_worst', 'perimeter_worst',
                        'area_worst', 'smoothness_worst', 'compactness_worst',
                        'concavity_worst', 'concave_points_worst', 'symmetry_worst',
                        'fractal_dimension_worst'
                        ]
    feature_mapping = {
        'concave_points_mean': 'concave points_mean',
        'concave_points_se': 'concave points_se',
        'concave_points_worst': 'concave points_worst'
    }
    X = df[relevant_features]
    X = X.rename(columns=feature_mapping)
    prediction = quantitative_model.predict(X) 
    confidence = quantitative_model.predict_proba(X)
    return {"prediction": class_names[int(prediction)], 
            "confidence": "{:.2f}".format(confidence[0][int(prediction)])}



# generate mask: segmentation model  
@app.post('/generate_mask')
async def generate_mask(file: UploadFile = File(...)): 
    """generate mask and return overlayed image & original image"""
    # loads image
    original_image = await file.read()
    original_image = Image.open(io.BytesIO(original_image)).convert('RGB')
    IMAGE_SIZE = 256
    original_image = original_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    image = img_to_array(original_image)
    image = tfi.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = image/255.
    # generate mask 
    pred_mask = segmentation_model.predict(image[np.newaxis, ...])[0] # adds batch dimension then remove it
    # overlay the mask on image 
    alpha = 0.5 
    overlayed_image = (1-alpha)*image + alpha * pred_mask
    predicted_class = predict_image(overlayed_image) 
    
    # convert to base64 
    buffered_original = io.BytesIO()
    original_image.save(buffered_original, format="PNG") 
    original_image_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
    
    buffered_overlayed = io.BytesIO()
    overlayed_image_pil = Image.fromarray((overlayed_image.numpy() * 255).astype(np.uint8))
    overlayed_image_pil.save(buffered_overlayed, format="PNG")
    overlayed_image_base64 = base64.b64encode(buffered_overlayed.getvalue()).decode('utf-8')
    
    return {
        "prediction": predicted_class, 
        "overlayed_image": overlayed_image_base64, 
        "original_image": original_image_base64
    }



# predict image: image classification model  
def predict_image(overlayed_image): 
    """given an overlayed image, return the predicted class"""
    
    class_names = ['benign', 'malignant'] 
    overlayed_image = tf_to_torch(overlayed_image) 
    # Run the PyTorch classification model
    with torch.no_grad():
        overlayed_image = overlayed_image.unsqueeze(0)  # Add batch dimension
        output = classification_model_mask(overlayed_image)
        _, pred = torch.max(output, 1)
        prediction = class_names[pred.item()]
    return prediction 


def tf_to_torch(tensor):
    """convert data type from tensorflow to pytorch compatible"""
    if not tf.is_tensor(tensor):
        raise ValueError("Input must be a TensorFlow tensor.")
    # Convert the TensorFlow tensor to a NumPy array
    np_array = tensor.numpy()
    # Convert the NumPy array to a PyTorch tensor
    torch_tensor = torch.from_numpy(np_array).permute(2, 0, 1)  # Change the order of dimensions to (C, H, W)
    return torch_tensor





 
 
 

# define chatbot 
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate  
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
# from langchain_together import Together  
import os
 

load_dotenv()
# api_key = os.getenv("TOGETHER_API_KEY")
# os.environ["TOGETHER_API_KEY"] = api_key

## Prompt Template
prompt_template = PromptTemplate(
    input_variables=['history', 'input'],
    template="""
    You are an AI Medical Assistant specializing in breast cancer support. Your goal is to provide empathetic, accurate, and supportive guidance to individuals seeking information about breast cancer. With integrated machine learning models, you can offer preliminary analysis of breast cancer-related data, but always emphasize the importance of consulting with healthcare professionals for a confirmed diagnosis.

    **Key Responsibilities:**
    - **Support and Empathy:** Recognize and validate users' emotions. Use comforting language to reassure them during difficult times.
    - **Clear Guidance:** Provide concise, direct information without medical jargon, making sure users understand the steps they need to take.
    - **Actionable Advice:** Offer practical, personalized recommendations and next steps tailored to the user's needs.
    - **Engagement:** Encourage open-ended questions to foster a two-way dialogue, making users feel heard and supported.
    - **Resource Provision:** Direct users to relevant resources and support groups where they can find additional help and community support.

    **Instructions for Users:**
    - If you wish to determine if your breast cancer is malignant or benign, please upload an ultrasound image or a data file containing real-valued features of the breast mass using the sidebar. I will analyze the data and provide a preliminary assessment.
    - Please remember that this analysis is not a substitute for professional medical advice. Consult with your healthcare provider for accurate diagnosis and treatment plans.

    **Example Interactions:**

    **Conversation History:**
    
    Human: What are the common symptoms of breast cancer?

    AI Medical Assistant: I understand that this can be a worrying time. Common symptoms of breast cancer include a lump or mass in the breast, changes in breast size or shape, skin dimpling, unusual nipple discharge, and persistent breast pain. It’s important to consult with a healthcare provider if you notice any of these symptoms. Is there anything specific you'd like to know more about?

    **Conversation History:**
    
    Human: How can I check if my breast cancer is malignant or benign?

    AI Medical Assistant: To determine whether your breast cancer is malignant or benign, please upload an ultrasound image or a data file containing specific features of the breast mass using the sidebar. I'll provide a preliminary assessment based on the data. Remember, this is just an initial analysis—your healthcare provider is essential for an accurate diagnosis and treatment plan. If you have any questions about the upload process, feel free to ask.

    **Conversation History:**
    
    Human: What are the treatments for breast cancer?

    AI Medical Assistant: There are several treatment options available, including surgery, chemotherapy, radiation therapy, hormone therapy, and targeted therapy. Each treatment plan is personalized based on the type and stage of cancer, as well as your individual health profile. I recommend discussing these options with your healthcare provider to determine the best approach for you. Would you like more details on any specific treatment or advice on managing side effects?

    **Conversation History:**
    
    Human: I have uploaded the image. What are the results?

    AI Medical Assistant: Thank you for uploading the image. Based on the analysis, there’s a high probability that your tumor is benign, which is hopeful news. However, it’s crucial to confirm this with your healthcare provider to determine the next steps. Regular follow-ups are vital in your health journey. If you have more questions or need further information, I’m here to support you.

    **Conversation History:**
    {history}
    
    Human: {input}
    
    AI Medical Assistant: 
    """
)

# ollama LLAma3 LLm 
# llm = Ollama(model="llama3", temperature = 0.7)
# llm = Together(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#     temperature=0.7,
#     top_p=0.7, 
#     top_k=50,
#     repetition_penalty=1, 
#     max_tokens=1000
# )

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

output_parser=StrOutputParser()

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    prompt=prompt_template, 
    output_parser=output_parser, 
    memory=memory
)


def ask_question(question): 
    return conversation.predict(input=question)



# api for chatbot

# Define a data model for the request body
class Question(BaseModel):
    question: str 
    
    
# Define the endpoint for asking questions
@app.post("/ask_chatbot")
async def ask_chatbot(question: Question):
    response = ask_question(question.question)
    return {"response": response} 


