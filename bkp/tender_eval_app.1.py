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
from components.layout import render_sidebar

# Set page configuration to change the title and favicon of the app
st.set_page_config(page_title="Tender Evaluation Bot", page_icon="🤖")

# API Gateway URL for your Lambda function
LAMBDA_API_URL = "https://9d859kfrp7.execute-api.us-east-1.amazonaws.com/dev/ask"
# Load evaluation criteria and prompt file path from the S3 bucket using the sidebar
#render_sidebar()
#with st.expander("Evaluation Documents "):
render_sidebar()
# ------------------------------------------------------
# Pydantic data model for Citations
class Citation(BaseModel):
    page_content: str
    metadata: Dict[str, Union[str, float, Dict[str, Any], None]]  # Flexible handling of metadata

# Helper function to extract citations from the response
def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})) for doc in response]

# ------------------------------------------------------
# S3 Presigned URL function
def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Generate a presigned URL to share an S3 object"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
            Params={'Bucket': bucket_name,
                    'Key': object_name},
            ExpiresIn=expiration)
    except NoCredentialsError:
        st.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI to extract bucket and key"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

# ------------------------------------------------------
# Function to call Lambda API
def call_lambda(question, history):
    # Prepare the payload with the question and entire conversation history
    payload = {
        "question": question,
        "history": [{"role": "user", "content": question}] + history  # Add the current user message first
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make POST request to Lambda API
        response = requests.post(LAMBDA_API_URL, json=payload, headers=headers)
        response.raise_for_status()

        # Print the raw response for debugging
        print("Raw Lambda API response:", response.text)

        # Process Lambda response
        response_data = response.json()
        if "body" in response_data:
            return json.loads(response_data["body"])  # Decode the JSON string in the 'body' field

        return response_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Lambda: {e}")
        return None

# ------------------------------------------------------
# Function to handle conversation
def handle_conversation(question, history):
    # Call the Lambda function and return the result
    return call_lambda(question, history)

# ------------------------------------------------------
# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# ------------------------------------------------------
# Streamlit

# Clear Chat History function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
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
    streaming_on = st.checkbox('Streaming')
    st.button('Clear Chat History', on_click=clear_chat_history)
    
    # Display the conversation history logs as structured JSON-like objects
    st.write("### History Logs")
    if "messages" in st.session_state:
        st.json(st.session_state.messages)  # Display the JSON-like format
    else:
        st.write("No history available.")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display previous messages in chat window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt
if prompt := st.chat_input():
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare the chat history for the Lambda payload, excluding the current prompt
    history_payload = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[:-1]]

    # Add a spinner while waiting for the Lambda response
    with st.spinner("Waiting for response..."):
        response = handle_conversation(prompt, history_payload)

    if response:
        full_response = response.get("response", "No response")
        context_data = response.get("context", [])

        # Handle streaming mode
        # if streaming_on:
        #     with st.chat_message("assistant"):
        #         placeholder = st.empty()
        #         chunks = full_response.split(". ")  # Split by sentences instead of words
        #         for chunk in chunks:
        #             time.sleep(0.2)  # Delay for streaming effect
        #             placeholder.write(chunk)
        if streaming_on:
            print("Streaming mode enabled")
            with st.chat_message("assistant"):
                placeholder = st.empty()
                simulate_streaming_response(full_response, placeholder)
        else:
            # Non-streaming mode: Display the full response at once
            with st.chat_message("assistant"):
                st.write(full_response)

        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Citations (if any) with S3 pre-signed URL
        if context_data:
            citations = extract_citations(context_data)
            with st.expander("Show source details >"):
                for citation in citations:
                    st.write("Page Content:", citation.page_content)
                    s3_uri = citation.metadata.get('location', {}).get('s3Location', {}).get('uri', "")
                    if s3_uri:
                        bucket, key = parse_s3_uri(s3_uri)
                        presigned_url = create_presigned_url(bucket, key)
                        if presigned_url:
                            st.markdown(f"Source: [{s3_uri}]({presigned_url})")
                        else:
                            st.write(f"Source: {s3_uri} (Presigned URL generation failed)")
                    st.write("Score:", citation.metadata.get('score', 'N/A'))
    else:
        st.error("Failed to retrieve response from Lambda.")
