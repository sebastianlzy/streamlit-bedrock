import io

import boto3
import streamlit as st
import json
import pydash
import time

s3_client = boto3.client("s3")
lambda_client = boto3.client("lambda")

bucket_name = st.secrets["EXTERNAL_FILES"]["S3_BUCKET_NAME"]
text_detection_function_name = st.secrets["EXTERNAL_FILES"]["TEXT_DETECTION_LAMBDA_FN_NAME"]
sns_publishing_role_arn = st.secrets["EXTERNAL_FILES"]["SNS_PUBLISHING_ROLE_ARN"]
sns_topic_arn = st.secrets["EXTERNAL_FILES"]["SNS_TOPIC_ARN"]


def is_file_a_pdf(filename):
    file_extension_name = filename.split('.')[-1]

    if file_extension_name == "pdf":
        return True

    return False


def invoke_lambda_to_start_text_detection(file_name):
    response = lambda_client.invoke(
        FunctionName=text_detection_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "S3_FILE_NAME": file_name,
            "S3_BUCKET_NAME": bucket_name,
            "SNS_PUBLISHING_ROLE_ARN": sns_publishing_role_arn,
            "SNS_TOPIC_ARN": sns_topic_arn
        })
    )

    lambda_response_payload = json.loads(response['Payload'].read())
    job_id = lambda_response_payload.get('JobId')

    return job_id


def get_result_from_text_detection(job_id):
    is_textract_job_completed = False
    response_text = ""

    while not is_textract_job_completed:
        time.sleep(10)
        print(job_id)
        response = lambda_client.invoke(
            FunctionName=text_detection_function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                "JOB_ID": job_id
            })
        )
        lambda_response_payload = json.loads(response['Payload'].read())
        lambda_payload_body = pydash.get(lambda_response_payload, 'body')
        print("lambda_payload_body: ", lambda_payload_body)

        if lambda_payload_body is None:
            is_textract_job_completed = True
            response_text = lambda_response_payload
            continue

        job_status = pydash.get(lambda_payload_body, 'JobStatus')
        print("job_status: ", job_status)

        if job_status == "FAILED":
            is_textract_job_completed = True
            continue

    return response_text


def upload_file_to_s3(bytes_data, filename):
    print("upload file to s3")
    s3_client.upload_fileobj(
        io.BytesIO(bytes_data),
        bucket_name,
        filename
    )


def extract_text_from_pdf(bytes_data, file_name):
    upload_file_to_s3(bytes_data, file_name)
    job_id = invoke_lambda_to_start_text_detection(file_name)
    text = get_result_from_text_detection(job_id)
    return text
