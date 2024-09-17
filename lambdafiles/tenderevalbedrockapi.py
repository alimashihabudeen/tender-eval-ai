import json
import boto3
from typing import List, Dict
from operator import itemgetter  # Import itemgetter for extracting dictionary keys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock, AmazonKnowledgeBasesRetriever

# Amazon Bedrock client setup
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")
s3_client = boto3.client('s3')  # S3 client to fetch context from S3 bucket

# Define the S3 bucket and object key for the evaluation criteria file
bucket_name = 'tender-eval-bucket'
object_key = 'prompt-files/evaluation_criteria.txt'

# Function to read context from S3 bucket
def read_s3_file(bucket_name, object_key):
    """
    Reads the file content from an S3 bucket and returns the content as a string.
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode('utf-8')  # Read and decode the content
        return content
    except Exception as e:
        raise Exception(f"Error reading S3 file: {str(e)}")
# Fetch the evaluation criteria from S3
evaluation_criteria = read_s3_file(bucket_name, object_key)
template = "'''"+evaluation_criteria+"'''"  

# Define Bedrock model and configuration
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.9,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# LangChain - Define the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the question based only on the following context:\n {context}"+template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="FYNKYVWUPB",  # Your KnowledgeBase ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

# Bedrock Chat Model
model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# Combine the retriever and model into a LangChain execution chain using itemgetter
chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,  # Use itemgetter for extracting 'question' to pass into retriever
        "question": itemgetter("question"),  # Extract 'question' for prompt
        "history": itemgetter("history"),  # Extract 'history' if available
    })
    .assign(response=prompt | model | StrOutputParser())  # Generate response from the model
)

# Function to invoke the chain and handle Document objects
def query_bedrock(question, history):
    inputs = {"question": question, "history": history}
    
    # Ensure that the question is a string before passing it through
    if isinstance(question, dict):
        # Convert the dictionary to a string
        question = json.dumps(question)
    
    # Run the LangChain pipeline
    output = chain.invoke(inputs)
    
    # Process the response and context
    response = output['response']
    
    # Convert context to JSON-serializable format by extracting page_content and metadata
    context_data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in output['context']
    ]
    
    return response, context_data

# Lambda Handler
def lambda_handler(event, context):
    try:
        # Extract the question and history from the request payload
        question = event.get('question', 'No question provided')
        history = event.get('history', [])

        # Ensure that the question is a string
        if not isinstance(question, str):
            question = json.dumps(question)

        # Invoke Bedrock and LangChain
        response, context_data = query_bedrock(question, history)

        # Return the response and context
        return {
            'statusCode': 200,
            'body': json.dumps({
                "response": response,
                "context": context_data
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }