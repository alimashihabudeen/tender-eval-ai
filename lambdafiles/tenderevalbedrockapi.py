import json
import boto3
import os
from typing import List, Dict, Any
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock, AmazonKnowledgeBasesRetriever
from botocore.config import Config

# Amazon Bedrock client setup
config = Config(
     connect_timeout=5000,
     read_timeout=5000,
)
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name="us-east-1", config=config)
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

# Fetch evaluation criteria from S3 to use as a guiding context for the prompt
evaluation_criteria = read_s3_file(bucket_name, object_key)

# Define the ChatPromptTemplate using from_template with evaluation criteria as the core context
prompt = ChatPromptTemplate.from_template(
    template="""

Evaluation Criteria:
{evaluation_criteria}

Context from Knowledge Base:
{context}

Previous Conversation:
{history}

Question:
{question}
"""
)
# Define the ChatPromptTemplate using from_message with evaluation criteria as the core context
# prompt = ChatPromptTemplate.from_messages(
#     messages=[
#         ("system", "You are a helpful assistant. Answer the question based only on the following context:\n{context}\nEvaluation Criteria:\n{evaluation_criteria}"),
#         MessagesPlaceholder(variable_name="history"),  # Placeholder for chat history
#         ("human", "{question}")
#     ]
# )
# Amazon Bedrock - KnowledgeBase Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id= os.environ['KNOWLEDGEBASEID'],#"9FGP6FQHXN",  # Your KnowledgeBase ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 12}},
)

# Bedrock model configuration
#model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_kwargs = {
    "max_tokens": 4096,
    "temperature": 0,
    "top_k": 250,
    "top_p": 0,
    "stop_sequences": ["\n\nHuman"],
}

# Bedrock Chat Model
model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    beta_use_converse_api=True,
)

# Combine the retriever and model into a LangChain execution chain
chain = RunnableParallel({
    "context": itemgetter("question") | retriever,  # Retrieve Knowledge Base context based on the question
    "evaluation_criteria": lambda _: evaluation_criteria,  # Pass evaluation criteria as core guiding context
    "question": itemgetter("question"),
    "history": itemgetter("history")  # Include conversation history
}).assign(response=prompt | model | StrOutputParser())  # Generate response based on evaluation criteria, history, and Knowledge Base

# Function to invoke the chain and handle Document objects
def query_bedrock(question, history):
    inputs = {
        "question": question,
        "history": history,  # Pass in chat history to maintain context
        #"context": "",  # Will be dynamically retrieved based on the question
        "evaluation_criteria": evaluation_criteria  # Pass evaluation criteria as the primary guiding context
    }
    
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
        for doc in output.get('context', [])
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

        # Invoke Bedrock and LangChain with the question and history
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