import time
import streamlit as st
import requests
import json
import boto3
from pydantic import BaseModel
from typing import List, Dict, Union, Any
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from components.layout import render_sidebar
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from botocore.exceptions import NoCredentialsError

# Set page configuration
st.set_page_config(page_title="Tender Evaluation Bot", page_icon="🤖")

# API Gateway URL
LAMBDA_API_URL = "https://9d859kfrp7.execute-api.us-east-1.amazonaws.com/dev/ask"

# Data model for citations
class Citation(BaseModel):
    page_content: str
    metadata: Dict[str, Union[str, float, Dict[str, Any], None]]

# Helper function to extract citations from the response
def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})) for doc in response]

# S3 presigned URL function
def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
            Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=expiration)
    except NoCredentialsError:
        st.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    parts = uri.replace("s3://", "").split("/")
    return parts[0], "/".join(parts[1:])

# Function to call Lambda API
def call_lambda(question, history):
    payload = {"question": question, "history": [{"role": "user", "content": question}] + history}
    print("Payload",payload)
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(LAMBDA_API_URL, json=payload, headers=headers, timeout=500)
        response.raise_for_status()
        response_data = response.json()
        if "body" in response_data:
            return json.loads(response_data["body"])
        return response_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Lambda: {e}")
        return None

# Function to handle conversation
def handle_conversation(question, history):
    return call_lambda(question, history)

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Clear Chat History function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your assistant for your Tender Evaluation. How can I help you?"}]
    st.session_state.conversation_started = False
    history.clear()

# Function to simulate streaming response (optional)
def simulate_streaming_response(full_response, placeholder):
    words = full_response.split(" ")
    current_text = ""
    for word in words:
        current_text += word + " "
        placeholder.markdown(current_text)
        time.sleep(0.05)

# Sidebar: Streaming toggle and History Logs
with st.sidebar:
    render_sidebar()
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.write("### History Logs")
    if "messages" in st.session_state:
        st.json(st.session_state.messages)
    else:
        st.write("No history available.")

# Initialize session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your assistant for your Tender Evaluation. How can I help you?"}]
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Display previous messages in chat window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display citations if available
def display_citations(context_data):
    if context_data:
        citations = extract_citations(context_data)
        with st.expander("Show source details >"):
            for citation in citations:
                st.write("Page Content:", citation.page_content)
                s3_uri = citation.metadata.get('location', {}).get('s3Location', {}).get('uri', "")
                if s3_uri:
                    bucket, key = parse_s3_uri(s3_uri)
                    presigned_url = create_presigned_url(bucket, key)
                    st.markdown(f"Source: [{s3_uri}]({presigned_url})" if presigned_url else f"Source: {s3_uri} (Presigned URL generation failed)")
                st.write("Score:", citation.metadata.get('score', 'N/A'))

# Evaluate button and response handling
if not st.session_state.conversation_started:
    if st.button('🔎 Evaluate and Summarise Tenderer Documents', help='Click to summarise the tenderer documents'):
        user_message = "Generate a review and evaluation report of the Tenderer's proposal."#"Generate a review and evaluation report of the Tenderer's proposal. Provide recommendation with justification."
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

        history_payload = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[:-1]]
        with st.spinner("Waiting for response..."):
            response = handle_conversation(user_message, history_payload)

        if response:
            full_response = response.get("response", "No response")
            context_data = response.get("context", [])
            with st.chat_message("assistant"):
                placeholder = st.empty()
                simulate_streaming_response(full_response, placeholder)
            display_citations(context_data)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.conversation_started = True

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history_payload = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[:-1]]
    with st.spinner("Waiting for response..."):
        response = handle_conversation(prompt, history_payload)

    if response:
        full_response = response.get("response", "No response")
        context_data = response.get("context", [])
        with st.chat_message("assistant"):
            placeholder = st.empty()
            simulate_streaming_response(full_response, placeholder)
        display_citations(context_data)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_started = True
