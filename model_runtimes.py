import boto3
import json

import pydash
from pydash import get

from prompt_generator import generate_better_prompt_via_meta_prompting

bedrock_runtime = boto3.client(service_name='bedrock-runtime')
top_p = 1
temperature = 0
top_k = 500


def invoke_runtime_model(model_id, runtime_input,
                         get_token_consumption=lambda _: {"input_tokens": 0, "output_tokens": 0},
                         accept='application/json'):
    content_type = 'application/json'
    try:
        body = json.dumps(runtime_input)

        response = bedrock_runtime.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)

        response_body = json.loads(pydash.get(response, 'body').read())
        return response_body, get_token_consumption(response_body)
    except Exception as e:
        print(e)
        return {}


def invoke_jurrasic_ultra_runtime(input, model_id):
    input_for_model_runtime = {
        'prompt': input["prompt"],
        'maxTokens': 1024,
        'temperature': temperature,
        'topP': top_p,
        'stopSequences': [],
        'countPenalty': {'scale': 0},
        'presencePenalty': {'scale': 0},
        'frequencyPenalty': {'scale': 0}
    }

    def get_token_consumption(response_body):
        return {
            "input_tokens": len(get(response_body, 'prompt.tokens')),
            "output_tokens": len(get(response_body, 'completions.0.data.tokens')),
        }

    return invoke_runtime_model(model_id, input_for_model_runtime, get_token_consumption)


def invoke_titan_text_g1_runtime(input, model_id):
    input_for_model_runtime = {
        'inputText': input["prompt"],
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': temperature,
            'topP': top_p
        }
    }
    return invoke_runtime_model(model_id, input_for_model_runtime)


def invoke_cohere_command_runtime(input, model_id):
    input_for_model_runtime = {
        'prompt': input["prompt"],
        'max_tokens': 1024,
        'temperature': temperature,
        'k': top_k,
        'p': top_p,
        'stop_sequences': [],
        'return_likelihoods': 'ALL'
    }

    def get_token_consumption(response_body):
        tokens = get(response_body, 'generations.0.token_likelihoods')
        input_tokens = 0
        for token in tokens:
            if get(token, "token") == '<EOP_TOKEN>':
                break
            input_tokens = input_tokens + 1

        return {
            "input_tokens": input_tokens,
            "output_tokens": len(tokens) - input_tokens,
        }

    return invoke_runtime_model(model_id, input_for_model_runtime, get_token_consumption)


def invoke_claude_2_runtime(input, model_id):
    input_for_model_runtime = {
        "prompt": f'\n\nHuman: {input["prompt"]} \n\nAssistant:',
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


def invoke_llama_2_13b_runtime(input, model_id):
    input_for_model_runtime = {
        "prompt": input["prompt"],
        "temperature": temperature,
        "top_p": top_p
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_llama_2_70b_runtime(input, model_id):
    input_for_model_runtime = {
        "prompt": input["prompt"],
        "temperature": temperature,
        "top_p": top_p
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_llama_3_70b_runtime(input, model_id):
    prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {input["prompt"]}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
    """

    input_for_model_runtime = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_mixtral_8x7b_runtime(input, model_id):
    input_for_model_runtime = {
        "prompt": input["prompt"],
        "temperature": temperature,
    }
    return invoke_runtime_model(model_id, input_for_model_runtime, accept="*/*")


def invoke_claude_3_sonnet_runtime(input, model_id):
    prompt = input["prompt"]
    content = [{"type": "text", "text": prompt}]
    print(input["image"])

    if input["image"] is not None:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": input["image"],
            },
        })

    input_for_model_runtime = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    def get_token_consumption(response_body):
        return {
            "input_tokens": get(response_body, 'usage.input_tokens'),
            "output_tokens": get(response_body, 'usage.output_tokens'),
        }

    return invoke_runtime_model(model_id, input_for_model_runtime, get_token_consumption)


def invoke_claude_3_haiku_runtime(input, model_id):
    input_for_model_runtime = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": input["prompt"]}],
            }
        ]
    }

    def get_token_consumption(response_body):
        return {
            "input_tokens": get(response_body, 'usage.input_tokens'),
            "output_tokens": get(response_body, 'usage.output_tokens'),
        }

    return invoke_runtime_model(model_id, input_for_model_runtime, get_token_consumption)
