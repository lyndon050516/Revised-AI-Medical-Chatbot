import pandas as pd 
import streamlit as st 
from utils import * 

# utility functions
def respond_to_image(prediction):     
    response = ask_chatbot_and_get_response(f"""Based on the user's uploaded ultrasound image, the machine learning classification algorithm detects the tumor 
                             to be {prediction} with high probability based on the provided ultrasound image. 
                             Inform the user/patient of this prediction. Be supportive for the user.""")
    st.session_state.messages.append({'role': 'assistant', 'content': response}) 

def respond_to_data(prediction, confidence): 
    response = ask_chatbot_and_get_response(f"""Based on teh user's uploaded data file, the machine learning classification algorithm
                             classifies the tumor to be {prediction} with confidence level of {confidence}. 
                             Inform the user/patient of this prediction. Be supportive for the user.""")
    st.session_state.messages.append({'role': 'assistant', 'content': response}) 
    
    
def display_hisotry(): 
    # display all historical messages in real time
    for message in st.session_state.messages: 
        if message['role'] == 'image':
            col1, col2, col3 = st.columns(3)
            # cwd = os.getcwd() # current working directory 
            # image_path = os.path.join("static/right-arrow.png")
            with col1:
                st.image(message['original_image'], caption='Original Image', use_column_width=True, width=200)
            with col2:
                st.image(Image.open("static/right-arrow.png"), caption='Image Segmentation', use_column_width=True)
            with col3:
                st.image(message['overlayed_image'], caption='Segmented Image', use_column_width=True, width=200)
        elif message['role'] == 'data':
            st.write("Uploaded Data:")
            st.dataframe(message['data'])
        else: 
            st.chat_message(message['role']).markdown(message['content'])

def display_image(original_image, overlayed_image):
    st.session_state.messages.append({
        'role': 'image',
        'original_image': original_image,
        'overlayed_image': overlayed_image
}) 
    
def display_data(data_frame): 
    st.session_state.messages.append({
        'role': 'data',
        'data': data_frame
    })

# app title 
st.set_page_config(page_title='AI Medical Assistant', page_icon='❤️')

st.title('Ask Your Medical Assistant 🧑‍⚕️')
st.sidebar.title('Upload Your Medical Data Below:')


# set up side bar  
uploaded_image = st.sidebar.file_uploader('Upload your medical image data:', type=['jpg', 'jpeg', 'png'])
uploaded_data = st.sidebar.file_uploader('Upload your medical tabular data:', type=['csv', 'xlsx', 'json'])

# setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state: 
    st.session_state.messages = []

if "uploaded_image" not in st.session_state: 
    st.session_state.uploaded_image = None

if "uploaded_data" not in st.session_state: 
    st.session_state.uploaded_data = None
    
    
# prompt input template to display the prompts 
input_text = st.chat_input("Ask your questions here?") 


# handle the user input
if input_text:
    # store user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': input_text})
    # get llm response
    response = ask_chatbot_and_get_response(input_text)
    # store the llm response in state 
    st.session_state.messages.append({'role': 'assistant', 'content': response}) 
 

if uploaded_image:
    # when a new image is uploaded
    if st.session_state.uploaded_image is None or st.session_state.uploaded_image != uploaded_image: 
        # api calls & retrieve response 
        prediction, overlayed_image, original_image = send_image_and_get_response(uploaded_image)
        # display response and add to state 
        display_image(original_image, overlayed_image)
        st.session_state.uploaded_image = uploaded_image
            
        # make chatbot respond 
        respond_to_image(prediction)
        
    
if uploaded_data: 
    if st.session_state.uploaded_data is None or st.session_state.uploaded_data != uploaded_data: 
        if uploaded_data.name.endswith('.csv'):
            data_frame = pd.read_csv(uploaded_data)
        elif uploaded_data.name.endswith('.xlsx'):
            data_frame = pd.read_excel(uploaded_data)
        elif uploaded_data.name.endswith('.json'):
            data_frame = pd.read_json(uploaded_data)
        else:
            st.error("Unsupported file type")
         
        st.session_state.uploaded_data = uploaded_data
        display_data(data_frame)
        
        # send data & get response
        json_result = json.loads(convert_to_json(data_frame))[0]  
        response = send_data_and_get_response(json_result) 
        prediction, confidence = response["prediction"], response["confidence"]  
        
        # Make chatbot respond
        respond_to_data(prediction, confidence)
        
display_hisotry()