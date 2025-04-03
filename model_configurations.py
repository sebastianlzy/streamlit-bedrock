from pydash import get, map_

from model_runtimes import invoke_claude_2_runtime, invoke_cohere_command_runtime, \
    invoke_llama_2_13b_runtime, invoke_llama_2_70b_runtime, invoke_llama_3_70b_runtime, invoke_titan_text_g1_runtime, \
    invoke_mixtral_8x7b_runtime, \
    invoke_claude_3_sonnet_runtime, invoke_claude_3_haiku_runtime, invoke_jurassic_jamba_runtime, \
    invoke_deepseek_r1_runtime

fm_models = [
    {
        "model_name": 'deepseek_r1',
        "model_id": "us.deepseek.r1-v1:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'choices.0.text'),
        "invoke_model_runtime": lambda input, _model_id: invoke_deepseek_r1_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00135 / 1000 + _tokens.get(
            "output_tokens") * 0.0054 / 1000,
    },
    {
        "model_name": "jurassic_jamba_1_5_large",
        "model_id": "ai21.jamba-1-5-large-v1:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'choices.0.message.content'),
        "invoke_model_runtime": lambda input, _model_id: invoke_jurassic_jamba_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.002 / 1000 + _tokens.get(
            "output_tokens") * 0.008 / 1000,
    },
    {
        "model_name": "claude_3_5_sonnet_v2",
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'content.0.text'),
        "invoke_model_runtime": lambda input, _model_id: invoke_claude_3_sonnet_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00800 / 1000 + _tokens.get(
            "output_tokens") * 0.02400 / 1000,
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
        "model_name": "llama_2_13b",
        "model_id": "meta.llama2-13b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda input, _model_id: invoke_llama_2_13b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00075 / 1000 + _tokens.get(
            "output_tokens") * 0.00100 / 1000,
    },
    {
        "model_name": "llama_2_70b",
        "model_id": "meta.llama2-70b-chat-v1",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda input, _model_id: invoke_llama_2_70b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00195 / 1000 + _tokens.get(
            "output_tokens") * 0.00256 / 1000,
    },
    {
        "model_name": "llama_3_70b",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "isEnabled": True,
        "output_formatter": lambda _response: get(_response, 'generation'),
        "invoke_model_runtime": lambda input, _model_id: invoke_llama_3_70b_runtime(input, _model_id),
        "calculate_cost": lambda _tokens: _tokens.get("input_tokens") * 0.00265 / 1000 + _tokens.get(
            "output_tokens") * 0.0035 / 1000,
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