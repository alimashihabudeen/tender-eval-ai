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
    """Render file upload, file listing, and delete button in a container section."""
    s3_client = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'tender-eval-bucket'
    tender_eval_folder = 'eval-doc-files/'  # Folder for Tender Evaluation Documents
    prompt_folder_name = 'prompt-files/'
    st.sidebar.title("Tender Evaluation GenAI POC 📊✍⚖️📝🔍") 
    #st.sidebar.header("Upload Evaluation Documents")

    def list_s3_files(folder):
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
            files = [item['Key'] for item in response.get('Contents', []) if item['Key'] != folder]  # Filter out root folder
            return files
        except ClientError as e:
            st.error(f"Error fetching files: {e.response['Error']['Message']}")
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            return []

    # def delete_s3_file(file_key):
    #     """Delete the selected file from the S3 bucket, ensure not deleting the folder."""
    #     try:
    #         if file_key.endswith('/'):
    #             st.error("Cannot delete a folder, only files.")
    #             return
            
    #         s3_client.delete_object(Bucket=bucket_name, Key=file_key)
    #         st.success(f"File '{file_key}' deleted successfully!")
    #         st.rerun()  # Rerun the app to reflect the changes
    #     except ClientError as e:
    #         st.error(f"Failed to delete the file: {e.response['Error']['Message']}")
    #     except Exception as e:
    #         st.error(f"An unexpected error occurred: {str(e)}")
    def delete_s3_file(file_key):
        """Permanently delete all versions of the file from the S3 bucket."""
        try:
            if file_key.endswith('/'):
                st.error("Cannot delete a folder, only files.")
                return
            
            # List object versions
            response = s3_client.list_object_versions(Bucket=bucket_name, Prefix=file_key)
            
            if 'Versions' not in response and 'DeleteMarkers' not in response:
                st.warning(f"No versions or delete markers found for '{file_key}'")
                return
            
            # Delete all versions and delete markers
            delete_list = []
            
            # Add all object versions
            if 'Versions' in response:
                for version in response['Versions']:
                    if version['Key'] == file_key:
                        delete_list.append({'Key': file_key, 'VersionId': version['VersionId']})
            
            # Add delete markers
            if 'DeleteMarkers' in response:
                for marker in response['DeleteMarkers']:
                    if marker['Key'] == file_key:
                        delete_list.append({'Key': file_key, 'VersionId': marker['VersionId']})
            
            # Perform the delete
            if delete_list:
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': delete_list}
                )
                st.success(f"All versions of '{file_key}' and delete markers have been permanently deleted!")
            else:
                st.warning(f"No valid versions or delete markers found for '{file_key}'")
            
            st.rerun()  # Rerun the app to reflect the changes
        
        except ClientError as e:
            st.error(f"Failed to delete the file: {e.response['Error']['Message']}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
        
    def upload_file(document, folder_name="eval-doc-files/"):
        """Upload the file to the S3 bucket."""
        s3_file_path = os.path.join(folder_name, document.name)
        try:
            s3_client.upload_fileobj(document, bucket_name, s3_file_path)
            st.success(f"Successfully uploaded the file to `{s3_file_path}`!")
            st.rerun()  # Rerun the app to reflect the changes
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    def list_s3_files_with_metadata(folder):
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
            
            # Debug: Print each item in Contents to inspect structure
            contents = response.get('Contents', [])
            #st.write("Debug: Contents of S3 Response:")
            #for item in contents:
            #    st.write(item)  # This will display each item to inspect its structure
    
            files = []
            for item in contents:
                # Ensure each entry has both 'Key' and 'LastModified'
                if isinstance(item, dict) and 'Key' in item and 'LastModified' in item and item['Key'] != folder:
                    files.append({
                        'Key': item['Key'],
                        'LastModified': item['LastModified']
                    })
            return files
        except ClientError as e:
            st.error(f"Error fetching files: {e.response['Error']['Message']}")
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            return []  
    # Create a container to group all elements together
    # st.markdown(
    #     """
    #     <style>
    #     .custom-container {
    #         background-color: #ff000050;
    #     }
    #     </style>
    #     """, 
    #     unsafe_allow_html=True
    #     )
    # container = st.container(border=True,height=800)
    # container.markdown(
    #     """
    #     <style>
    #     .custom-container {
    #         background-color: #ff000050;
    #     }
    #     </style>
    #     """, 
    #     unsafe_allow_html=True
    #     )
    #container.markdown('<div class="custom-container">', unsafe_allow_html=True)
    # Add custom CSS to make the expander background white
    st.markdown(
        """
        <style>
        .streamlit-expanderHeader {
            background-color: white;
            color: black; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: white;
            color: black; # Expander content color
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # container = st.expander("Step 1: Upload Evaluation Documents",expanded=False)
    #      # Sidebar UI
    # #container.write("This is inside the container")
    # #     # File uploader for Tenderer Evaluation Document
    # document = container.file_uploader(" ", type=None, key="tenderer_eval_doc")

    # if document is not None:
    #     upload_file(document, folder_name=tender_eval_folder)

    # # List Tender Evaluation files with delete buttons
    # tender_eval_files = list_s3_files(tender_eval_folder)

        #st.subheader("Tender Evaluation Files")
    # if tender_eval_files:
    #     for file_info in tender_eval_files:
    #         file_key = file_info['Key']
    #         file_name = file_key.split('/')[-1]  # Display only the file name
    #         last_modified = file_info['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
    
    #         # Use columns to display the file name with the delete button
    #         col1, col2 = container.columns([8, 1])
    #         with col1:
    #             container.write(f"{file_name} (Last Modified: {last_modified})")
    #         with col2:
    #             if container.button('🗑️', key=file_key, help="Delete this file"):
    #                 delete_s3_file(file_key)
    # else:
    #     container.write("No files available.")
    # Create the expander for Evaluation Criteria
    with st.expander("Step 1:Upload Evaluation Documents", expanded=False):
        #st.write("Note: Previous version of the evaluation criteria will be replaced.")
        uploaded_file = st.file_uploader("Note: Previous version of the evaluation criteria will be replaced.", type=None, key="tender_eval_file_uploader_unique_key")

        # If a file is uploaded, save it in the prompt folder
        if uploaded_file:
            upload_file(uploaded_file, folder_name=tender_eval_folder)

        # List the evaluation criteria file in the prompt folder
        tender_eval_files = list_s3_files_with_metadata(tender_eval_folder)
        if tender_eval_files:
            for file_info in tender_eval_files:
                file_name = file_info['Key'].split('/')[-1]
                file_key = file_info['Key']
                #if file_name == 'evaluation_criteria.txt':
                last_modified = file_info['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                #st.write(f"{file_name} (Last Modified: {last_modified})")
           #         # Use columns to display the file name with the delete button
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(f"{file_name} (Last Modified: {last_modified})")
                with col2:
                    if st.button('🗑️', key=file_key, help="Delete this file"):
                        delete_s3_file(file_key)         
        else:
            st.write("No evaluation criteria file available.")


    # Create the expander for Evaluation Criteria
    with st.expander("Step 2: Upload Evaluation Criteria", expanded=False):
        #st.write("Note: Previous version of the evaluation criteria will be replaced.")
        uploaded_file = st.file_uploader("Note: Previous version of the evaluation criteria will be replaced.", type="txt", key="file_uploader_unique_key")

        # If a file is uploaded, save it in the prompt folder
        if uploaded_file:
            upload_file(uploaded_file, folder_name=prompt_folder_name)

        # List the evaluation criteria file in the prompt folder
        prompt_eval_files = list_s3_files_with_metadata(prompt_folder_name)
        if prompt_eval_files:
            for file_info in prompt_eval_files:
                file_name = file_info['Key'].split('/')[-1]
                if file_name == 'evaluation_criteria.txt':
                    last_modified = file_info['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    st.write(f"{file_name} (Last Modified: {last_modified})")
        else:
            st.write("No evaluation criteria file available.")
# Main app logic
#def main():
#    render_file_upload_section()

#if __name__ == "__main__":
#    main()
