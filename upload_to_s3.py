# upload_to_s3.py

import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

# Constants
BUCKET_NAME = 's3-bucket-umairrr'
S3_FOLDER = 'feature-selection-multipipeline/'
LOCAL_FILE_PATH = os.path.join('data', 'feature_selection.csv')
S3_FILE_KEY = S3_FOLDER + 'feature_selection.csv'

def upload_to_s3():
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(LOCAL_FILE_PATH, BUCKET_NAME, S3_FILE_KEY)
        print(f"✅ Successfully uploaded {LOCAL_FILE_PATH} to s3://{BUCKET_NAME}/{S3_FILE_KEY}")
    except FileNotFoundError:
        print(f"❌ File not found: {LOCAL_FILE_PATH}")
    except NoCredentialsError:
        print("❌ AWS credentials not found.")
    except ClientError as e:
        print(f"❌ Client error: {e}")

if __name__ == "__main__":
    upload_to_s3()
