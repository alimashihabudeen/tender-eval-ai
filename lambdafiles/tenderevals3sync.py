import os
import json
import boto3

# Initialize Bedrock client
bedrockClient = boto3.client('bedrock-agent')

def lambda_handler(event, context):
    print('Inside Lambda Handler')
    print('Event: ', json.dumps(event, indent=2))

    dataSourceId = os.environ['DATASOURCEID']
    knowledgeBaseId = os.environ['KNOWLEDGEBASEID']
    
    # Iterate through the records in the event
    for record in event.get('Records', []):
        s3_event_name = record['eventName']
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        
        print(f'Event Name: {s3_event_name}, Bucket: {bucket_name}, Key: {object_key}')
        
        # Exclude delete markers in case of object removal events
        if 'ObjectRemoved' in s3_event_name:
            # Check for delete marker flag (handled for versioned S3 buckets)
            if record['s3']['object'].get('deleteMarker', False):
                print(f"Skipping delete marker for {object_key}")
                continue  # Skip delete marker objects
        
        # Trigger Bedrock ingestion job
        print('Starting Bedrock Ingestion Job...')
        response = bedrockClient.start_ingestion_job(
            knowledgeBaseId=knowledgeBaseId,
            dataSourceId=dataSourceId
        )
        
        print('Ingestion Job Response: ', response)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Ingestion Job Triggered Successfully')
    }