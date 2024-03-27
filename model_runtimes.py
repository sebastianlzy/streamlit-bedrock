import boto3
import json
import os

bedrock = boto3.client('bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500


def invoke_runtime_model(model_id, runtime_input, accept='application/json'):
    contentType = 'application/json'
    try:
        body = json.dumps(runtime_input)
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        return response_body
    except Exception as e:
        print(e)
        return {}


def invoke_jurrasic_runtime(prompt, model_id):
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


def invoke_titan_text_g1_runtime(prompt, model_id):
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


def invoke_cohere_runtime(prompt, model_id):
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


def invoke_claude_2_runtime(prompt, model_id):
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


def invoke_llama_13b_runtime(prompt, model_id):
    input_for_model_runtime = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_gen_len": 512
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_llama_70b_runtime(prompt, model_id):
    input_for_model_runtime = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_gen_len": 512
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")
