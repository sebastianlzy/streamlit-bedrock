import multiprocess as mp


import boto3
import json

import pydash
from pydash import get, map_
import os
import time
from prompts import *
import pandas as pd
import streamlit as st

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500

fm_models = [
    {
        "model_name": "jurassic",
        "modelId": "ai21.j2-ultra",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completions.0.data.text'),
        "invoke_model_runtime": lambda _input_prompt: invoke_jurrasic_runtime(_input_prompt)
    },
    {
        "model_name": "claude_2",
        "modelId": "anthropic.claude-v2:1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'completion'),
        "invoke_model_runtime": lambda _input_prompt: invoke_claude_runtime(_input_prompt)
    },
    {
        "model_name": "cohere",
        "modelId": "cohere.command-text-v14",
        "isEnabled": True,
        "output_formatter": lambda _response: " ".join(map_(get(_response, 'generations'), "text")),
        "invoke_model_runtime": lambda _input_prompt: invoke_cohere_runtime(_input_prompt)
    },
    {
        "model_name": "llama_13b",
        "modelId": "meta.llama2-13b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda _input_prompt: invoke_llama_13b_runtime(_input_prompt)
    },
    {
        "model_name": "llama_70b",
        "modelId": "meta.llama2-70b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda _input_prompt: invoke_llama_70b_runtime(_input_prompt)
    },
]


def list_foundational_models():
    response = bedrock.list_foundation_models()
    models = response['modelSummaries']
    table = []

    for model in models:
        table.append([model.get('model_name'), model.get('modelId'), model.get('responseStreamingSupported')])
    return pd.DataFrame(table, columns=['ModelName', 'modelId', 'isStreamingSupported'])


def invoke_runtime_model(model_id, runtime_input, accept='application/json'):
    contentType = 'application/json'

    body = json.dumps(runtime_input)
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body


def invoke_jurrasic_runtime(prompt):
    model_id = os.environ.get("JURASSIC_MODEL_ID")
    input_for_model_runtime = {
        'prompt': prompt,
        'maxTokens': 1024,
        'temperature': temperature,
        'topP': top_p,
        'stopSequences': [],
        'countPenalty': {'scale': 0},
        'presencePenalty': {'scale': 0},
        'frequencyPenalty': {'scale': 0}
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_amazon_titan_runtime(prompt):
    model_id = os.environ.get("AMAZON_TITAN_MODEL_ID")
    input_for_model_runtime = {
        'inputText': prompt,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': temperature,
            'topP': top_p
        }
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_cohere_runtime(prompt):
    model_id = os.environ.get("COHERE_MODEL_ID")
    input_for_model_runtime = {
        'prompt': prompt,
        'max_tokens': 1024,
        'temperature': temperature,
        'k': top_k,
        'p': top_p,
        'stop_sequences': [],
        'return_likelihoods': 'NONE'
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_claude_runtime(prompt):
    model_id = os.environ.get("CLAUDE_2_MODEL_ID")
    input_for_model_runtime = {
        "prompt": f'\n\nHuman: {prompt} \n\nAssistant:',
        "max_tokens_to_sample": 300,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": [
            "\n\nHuman:"
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_llama_13b_runtime(prompt):
    model_id = os.environ.get("LLAMA_2_13B")
    input_for_model_runtime = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_llama_70b_runtime(prompt):
    model_id = os.environ.get("LLAMA_2_70B")
    input_for_model_runtime = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def measure_time_taken(cb):
    start_time = time.time()
    response = cb()
    time_in_seconds = time.time() - start_time
    return response, time_in_seconds


def execute_thread(input_prompt, fm_model, model_outputs):
    model_name = fm_model["model_name"]
    print(f"executing {model_name} model")
    response, response_in_seconds = measure_time_taken(
        lambda: fm_model["invoke_model_runtime"](input_prompt)
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


if __name__ == "__main__":
    st.title("Prompt")
    prompt = st.text_area("Prompt", custom_prompt, label_visibility="hidden")
    st.divider()

    enabled_fm_models = pydash.filter_(fm_models, lambda x: x['isEnabled'])
    model_outputs = main(prompt, enabled_fm_models)
    
    for model_output in model_outputs:
        with st.expander(f'#{model_output["model_name"].capitalize()}', expanded=True):
            st.write(model_output["runtime_response_in_text"])
            st.caption(f'{model_output["time_taken_in_seconds"]} sec')

    with st.expander(f'#List of models'):
        st.data_editor(list_foundational_models(), use_container_width=True)
