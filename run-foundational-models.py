import base64

import multiprocess as mp

import boto3
import pydash
from pydash import get, map_
import time
from prompts import *
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from model_runtimes import invoke_jurrasic_ultra_runtime, invoke_claude_2_runtime, invoke_cohere_command_runtime, \
    invoke_llama_13b_runtime, invoke_llama_70b_runtime, invoke_titan_text_g1_runtime, invoke_mixtral_8x7b_runtime, \
    invoke_claude_3_sonnet_runtime, invoke_claude_3_haiku_runtime

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500

fm_models = [
    {
        "model_name": "jurassic_ultra",
        "model_id": "ai21.j2-ultra",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completions.0.data.text'),
        "invoke_model_runtime": lambda input, _model_id: invoke_jurrasic_ultra_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.0188 / 1000 + _tokens.get(
            "output_tokens") * 0.0188 / 1000,
    },
    {
        "model_name": "claude_2",
        "model_id": "anthropic.claude-v2:1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completion'),
        "invoke_model_runtime": lambda input, _model_id: invoke_claude_2_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00800 / 1000 + _tokens.get(
            "output_tokens") * 0.02400 / 1000,
    },
    {
        "model_name": "cohere_command",
        "model_id": "cohere.command-text-v14",
        "isEnabled": True,
        "output_formatter": lambda _response: " ".join(map_(get(_response, 'generations'), "text")),
        "invoke_model_runtime": lambda input, _model_id: invoke_cohere_command_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.0015 / 1000 + _tokens.get(
            "output_tokens") * 0.0020 / 1000,
    },
    {
        "model_name": "llama_13b",
        "model_id": "meta.llama2-13b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda input, _model_id: invoke_llama_13b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00075 / 1000 + _tokens.get(
            "output_tokens") * 0.00100 / 1000,
    },
    {
        "model_name": "llama_70b",
        "model_id": "meta.llama2-70b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda input, _model_id: invoke_llama_70b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00195 / 1000 + _tokens.get(
            "output_tokens") * 0.00256 / 1000,
    },
    {
        "model_name": "titan_text_lite",
        "model_id": "amazon.titan-text-lite-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'results.0.outputText'),
        "invoke_model_runtime": lambda input, _model_id: invoke_titan_text_g1_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.0003 / 1000 + _tokens.get(
            "output_tokens") * 0.0004 / 1000,
    },
    {
        "model_name": "mixtral_8x7b",
        "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'outputs.0.text'),
        "invoke_model_runtime": lambda input, _model_id: invoke_mixtral_8x7b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00045 / 1000 + _tokens.get(
            "output_tokens") * 0.0007 / 1000,
    },
    {
        "model_name": "claude_3_sonnet",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'content.0.text'),
        "invoke_model_runtime": lambda input, _model_id: invoke_claude_3_sonnet_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00300 / 1000 + _tokens.get(
            "output_tokens") * 0.01500 / 1000,
    },
    {
        "model_name": "claude_3_haiku",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'content.0.text'),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_claude_3_haiku_runtime(
            _input_prompt,
            _model_id
        ),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00025 / 1000 + _tokens.get(
            "output_tokens") * 0.00125 / 1000,
    },
]


def list_foundational_models():
    response = bedrock.list_foundation_models()
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


def main(input_prompt, input_image, fm_models):
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


if __name__ == "__main__":
    with open("analytics.html", "r") as f:
        html_code = f.read()
    components.html(html_code, height=0)

    st.title("Prompt")
    prompt = st.text_area("Prompt", custom_prompt, label_visibility="hidden")
    uploaded_file = st.file_uploader("Choose a file (only works with Claude Sonnet)")
    encoded_image = None

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        encoded_image = base64.b64encode(bytes_data).decode('utf-8')

    st.divider()

    fm_model_names = pydash.map_(fm_models, "model_name")
    selected_fm_model_names = st.multiselect(
        'Select models',
        fm_model_names,
        ["jurassic_ultra", "cohere_command", "claude_3_sonnet"]
    )
    print(selected_fm_model_names)

    enabled_fm_models = pydash.filter_(
        fm_models,
        lambda x: x['isEnabled'] and is_selected(selected_fm_model_names, x["model_name"])
    )
    model_outputs = main(prompt, encoded_image, enabled_fm_models)

    for model_output in model_outputs:
        with st.expander(f'{model_output["model_name"].capitalize()} took {model_output["time_taken_in_seconds"]} sec',
                         expanded=True):
            st.write(model_output["runtime_response_in_text"])
            if model_output["total_cost"] > 0:
                st.caption(f'${model_output["total_cost"]}')

    with st.expander(f'#List of models'):
        st.data_editor(list_foundational_models(), use_container_width=True)


