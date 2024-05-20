import base64
import time

import boto3
import multiprocess as mp
import pandas as pd
import pydash
import streamlit as st
import streamlit.components.v1 as components
import streamlit_analytics2 as streamlit_analytics

from model_configurations import fm_models
from process_pdf_to_text import is_file_a_pdf, extract_text_from_pdf
from prompt_generator import generate_better_prompt_via_meta_prompting
from prompts import *

bedrock_client = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500


def list_foundational_models():
    response = bedrock_client.list_foundation_models()
    models = response['modelSummaries']
    table = []

    for model in models:
        table.append([model.get('modelName'), model.get('modelId'), model.get('responseStreamingSupported')])
    return pd.DataFrame(table, columns=['ModelName', 'modelId', 'isStreamingSupported'])


def measure_time_taken(cb):
    start_time = time.time()
    response_body, tokens_consumed = cb()
    time_in_seconds = time.time() - start_time
    return response_body, tokens_consumed, time_in_seconds


def execute_thread(input_prompt, input_image, fm_model, model_outputs):
    model_name = fm_model["model_name"]
    print(f"executing {model_name} model")
    response_body, tokens_consumed, response_in_seconds = measure_time_taken(
        lambda: fm_model["invoke_model_runtime"](
            {"prompt": input_prompt, "image": input_image},
            fm_model["model_id"])
    )

    model_output = {
        "model_name": model_name,
        "runtime_response": response_body,
        "runtime_response_in_text": fm_model["output_formatter"](response_body),
        "time_taken_in_seconds": response_in_seconds,
        "total_cost": fm_model["calculate_cost"](tokens_consumed),
    }
    print(f"{model_name} executed")
    model_outputs.append(model_output)
    return model_output


def invoke_models_in_parallel(input_prompt, input_image, fm_models):
    jobs = []
    manager = mp.Manager()
    model_outputs = manager.list()

    for fm_model in fm_models:
        process = mp.Process(target=execute_thread, args=(input_prompt, input_image, fm_model, model_outputs))
        process.start()
        jobs.append(process)

    for job in jobs:
        job.join()

    return model_outputs._getvalue()


def is_selected(arr, predicate):
    return pydash.find_index(arr, lambda x: x == predicate) > - 1


def main():
    with open("analytics.html", "r") as f:
        html_code = f.read()
    components.html(html_code, height=10)

    st.title(f'Prompt')
    prompt = st.text_area("Prompt", custom_prompt, label_visibility="hidden", key='prompt_text_area')
    is_generate_prompt_selected = st.checkbox("Do you want me to submit an alternative prompt for you?")
    if is_generate_prompt_selected:
        prompt = generate_better_prompt_via_meta_prompting(prompt)
        with st.expander("Generated alternative prompt"):
            st.caption(prompt)

    analytics_password = st.secrets["ANALYTICS"]["DASHBOARD_PASSWORD"]
    streamlit_analytics.start_tracking()

    uploaded_file = st.file_uploader(
        "This feature utilises Textract for PDF. For images, it only works with Claude Sonnet")
    encoded_image = None

    user_email = st.experimental_user["email"]
    st.text_input(label="user_email", value=user_email, label_visibility="hidden", disabled=True)

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        file_name = uploaded_file.name
        encoded_image = base64.b64encode(bytes_data).decode('utf-8')

        if is_file_a_pdf(file_name):
            text = extract_text_from_pdf(bytes_data, file_name)
            prompt = f'### PDF Text ### \n {text} \n ### Instruction ### {prompt}'
            encoded_image = None
            print(prompt)

    st.divider()

    fm_model_names = pydash.map_(fm_models, "model_name")
    default_models = ["jurassic_ultra", "cohere_command", "claude_3_sonnet"]

    selected_fm_model_names = st.multiselect(
        'Select models',
        fm_model_names,
        default_models
    )

    enabled_fm_models = pydash.filter_(
        fm_models,
        lambda x: x['isEnabled'] and is_selected(selected_fm_model_names, x["model_name"])
    )

    streamlit_analytics.stop_tracking(
        unsafe_password=analytics_password,
        # streamlit_secrets_firestore_key="firebase",
        # firestore_collection_name="streamlit-analytics2",
        # firestore_project_name="streamlit-bedrock"
    )

    model_outputs = invoke_models_in_parallel(prompt, encoded_image, enabled_fm_models)

    for model_output in model_outputs:
        with st.expander(
                f'{model_output["model_name"].capitalize()} took {model_output["time_taken_in_seconds"]} sec',
                expanded=True):
            st.write(model_output["runtime_response_in_text"])
            if model_output["total_cost"] > 0:
                st.caption(f'${model_output["total_cost"]}')

    with st.expander(f'#List of models'):
        st.data_editor(list_foundational_models(), use_container_width=True)


if __name__ == "__main__":
    main()
