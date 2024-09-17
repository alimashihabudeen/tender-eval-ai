import time
import streamlit as st
import boto3
import os
from botocore.exceptions import ClientError

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-agent', region_name='us-east-1')
lambda_client = boto3.client('lambda', region_name='us-east-1')

def trigger_bedrock_sync():
    """Trigger Bedrock sync using a Lambda function."""
    try:
        with st.spinner('Syncing with Bedrock...'):
            # Call the Lambda function to start Bedrock ingestion job
            response = lambda_client.invoke(
                FunctionName='tendereval-upload-doc-autosync',
                InvocationType='RequestResponse'
            )
            response_payload = response['Payload'].read().decode('utf-8')

        # Check if the response indicates success
        if 'Success' in response_payload:  # Check for success based on your payload response
            st.success('Knowledge Base Synced Successfully ✅')
        else:
            st.error(f"Bedrock Sync Error: {response_payload}")

    except ClientError as e:
        st.error(f"Error starting Bedrock sync: {e.response['Error']['Message']}")
    except Exception as e:
        st.error(f"An unexpected error occurred during Bedrock sync: {str(e)}")

def render_sidebar():
    """Render the sidebar with file upload, selection, and deletion."""
    s3_client = boto3.client('s3')
    bucket_name = 'tender-eval-bucket'
    tender_eval_folder = 'eval-doc-files/'  # Folder for Tender Evaluation Documents

    def list_s3_files(folder):
        """List files in a specified S3 folder, filtering out the root folder."""
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
            files = [item['Key'] for item in response.get('Contents', []) if item['Key'] != folder]  # Filter out root folder
            return files
        except ClientError as e:
            st.error(f"Error fetching files: {e.response['Error']['Message']}")
            return []

    def delete_s3_file(file_key):
        """Delete the selected file from the S3 bucket, ensure not deleting the folder."""
        try:
            # Ensure it's not trying to delete the folder itself
            if file_key.endswith('/'):
                st.sidebar.error("Cannot delete a folder, only files.")
                return
            
            s3_client.delete_object(Bucket=bucket_name, Key=file_key)
            st.sidebar.success(f"File '{file_key}' deleted successfully!")
            st.rerun()  # Rerun the app to reflect the changes
        except ClientError as e:
            st.sidebar.error(f"Failed to delete the file: {e.response['Error']['Message']}")
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred: {str(e)}")
    
    def upload_file(document, folder_name="eval-doc-files/"):
        """Upload the file to the S3 bucket, replacing the existing one."""
        s3_file_path = os.path.join(folder_name, document.name)  # Use original file name
        try:
            s3_client.upload_fileobj(document, bucket_name, s3_file_path)
            st.sidebar.success(f"Successfully uploaded the file to `{s3_file_path}`!")
            # Trigger Bedrock sync after a successful upload
            #trigger_bedrock_sync()
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

    # Sidebar UI
    st.sidebar.title("Tender Evaluation GenAI POC 📊✍⚖️📝🔍") 
    st.sidebar.header("Upload Tenderer Evaluation Document")

    # File uploader for Tenderer Evaluation Document
    document = st.sidebar.file_uploader("Upload Document", type=None, key="tenderer_eval_doc")

    if document is not None:
        # Directly upload the file to S3, replacing any existing file
        upload_file(document, folder_name=tender_eval_folder)

    # List and select Tender Evaluation files
    tender_eval_files = list_s3_files(tender_eval_folder)

    # Creating columns to place dropdown and delete button next to each other
    col1, col2 = st.sidebar.columns([8, 1])

    with col1:
        selected_eval_file = st.selectbox("Select an Evaluation Document", tender_eval_files, key="selected_eval_file")

    with col2:
        if selected_eval_file and st.button("❌", key="delete_eval_file", help="Delete this evaluation document"):
            delete_s3_file(selected_eval_file)
            st.write("No content available for the selected evaluation document.")
            st.rerun()

# Main app logic
def main():
    render_sidebar()

if __name__ == "__main__":
    main()
