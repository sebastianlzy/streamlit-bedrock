import multiprocess as mp

import boto3
import pydash
from pydash import get, map_
import time
from prompts import *
import pandas as pd
import streamlit as st

from model_runtimes import invoke_jurrasic_runtime, invoke_claude_2_runtime, invoke_cohere_runtime, \
    invoke_llama_13b_runtime, invoke_llama_70b_runtime

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500

fm_models = [
    {
        "model_name": "jurassic",
        "model_id": "ai21.j2-ultra",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completions.0.data.text'),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_jurrasic_runtime(_input_prompt, _model_id)
    },
    {
        "model_name": "claude_2",
        "model_id": "anthropic.claude-v2:1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completion'),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_claude_2_runtime(_input_prompt, _model_id)
    },
    {
        "model_name": "cohere",
        "model_id": "cohere.command-text-v14",
        "isEnabled": True,
        "output_formatter": lambda _response: " ".join(map_(get(_response, 'generations'), "text")),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_cohere_runtime(_input_prompt, _model_id)
    },
    {
        "model_name": "llama_13b",
        "model_id": "meta.llama2-13b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_llama_13b_runtime(_input_prompt, _model_id)
    },
    {
        "model_name": "llama_70b",
        "model_id": "meta.llama2-70b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda _input_prompt, _model_id: invoke_llama_70b_runtime(_input_prompt, _model_id)
    },
]


def list_foundational_models():
    response = bedrock.list_foundation_models()
    models = response['modelSummaries']
    table = []

    for model in models:
        table.append([model.get('model_name'), model.get('model_id'), model.get('responseStreamingSupported')])
    return pd.DataFrame(table, columns=['ModelName', 'modelId', 'isStreamingSupported'])


def measure_time_taken(cb):
    start_time = time.time()
    response = cb()
    time_in_seconds = time.time() - start_time
    return response, time_in_seconds


def execute_thread(input_prompt, fm_model, model_outputs):
    model_name = fm_model["model_name"]
    print(f"executing {model_name} model")
    response, response_in_seconds = measure_time_taken(
        lambda: fm_model["invoke_model_runtime"](input_prompt, fm_model["model_id"])
    )
    model_output = {
        "model_name": model_name,
        "runtime_response": response,
        "runtime_response_in_text": fm_model["output_formatter"](response),
        "time_taken_in_seconds": response_in_seconds
    }
    print(f"{model_name} executed")
    model_outputs.append(model_output)
    return model_output


def main(input_prompt, fm_models):
    jobs = []
    manager = mp.Manager()
    model_outputs = manager.list()

    for fm_model in fm_models:
        process = mp.Process(target=execute_thread, args=(input_prompt, fm_model, model_outputs))
        process.start()
        jobs.append(process)

    for job in jobs:
        job.join()

    return model_outputs._getvalue()


def is_selected(arr, predicate):
    return pydash.find_index(arr, lambda x: x == predicate) > - 1


if __name__ == "__main__":
    st.title("Prompt")
    prompt = st.text_area("Prompt", custom_prompt, label_visibility="hidden")
    st.divider()

    fm_model_names = pydash.map_(fm_models, "model_name")
    selected_fm_model_names = st.multiselect(
        'Select models',
        fm_model_names,
        ["jurassic", "cohere", "claude_2", "llama_70b"]
    )
    print(selected_fm_model_names)

    enabled_fm_models = pydash.filter_(
        fm_models,
        lambda x: x['isEnabled'] and is_selected(selected_fm_model_names, x["model_name"])
    )
    model_outputs = main(prompt, enabled_fm_models)

    for model_output in model_outputs:
        with st.expander(f'#{model_output["model_name"].capitalize()}', expanded=True):
            st.write(model_output["runtime_response_in_text"])
            st.caption(f'{model_output["time_taken_in_seconds"]} sec')

    with st.expander(f'#List of models'):
        st.data_editor(list_foundational_models(), use_container_width=True)
