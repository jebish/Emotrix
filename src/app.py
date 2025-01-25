import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import requests

API_URL="https://public_url.ngrok-free.app/predict/"

#Initialize a session state to store messages
if 'messages' not in st.session_state:
    st.session_state.messages=[]

#Set page title
st.set_page_config(page_title='Emotrix - An LLM-powered Depression Detection app')


#this function clears the session state and also clears conversation history in api
def reset_conversation():
    st.session_state.messages.clear()
    st.session_state.past.clear()
    response=requests.post(API_URL,json={'task':'reset','prompt':'classify'})

#Create sidebar for some info
with st.sidebar:
    st.title('🔥 Ellama App')
    st.markdown('''
    ## About
                This app in an LLM-powered chatbot
                built using:
                - [Streamlit]
                - [Unsloth]
                - [LLaMA 3.2 3B]
                - [DAIC WoZ]
''')
    add_vertical_space(5)
    st.write('Made with 💖 by Jebish')

#Reset button to trigger reset_conversation()
    if st.button("Reset"):
        reset_conversation()
        st.rerun()


col2, col3 = st.columns([20,4])

input_container=st.container()

#A horizontal line to divide input and output
st.markdown("""<hr style="height:5px;border:none;color:#342;background-color:#008080;" /> """, unsafe_allow_html=True)
output_container=st.container(height=600,border=None)


#Function to get response/question from backend
def get_response(user_input):
    response=requests.post(API_URL,json={'task':'chat','prompt':user_input})
    if response.status_code==200:
        return response.json().get('response')
    else:
        return "Error in response"

#Function to get classification response from backend
def classify_conversation():
    response=requests.post(API_URL,json={'task':'classify','prompt':'classify'})
    if response.status_code==200:
        return response.json().get('response')
    else:
        return "Error in response"

#input chatbox for user with classification button
with input_container:
    with col2:
        prompt=st.chat_input('Hello there')
    with col3:
        class_button=st.button('Classify',use_container_width=True)


#Output display section
with output_container:
    for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        
        st.session_state.messages.append({'role':'user','content':prompt})
        response=get_response(prompt)

        with st.chat_message('assistant'):
            st.markdown(response)

        st.session_state.messages.append({'role':'assistant','content':response})
    
    if class_button:
        response=classify_conversation()
        with st.chat_message('assistant'):
            st.markdown(response)
