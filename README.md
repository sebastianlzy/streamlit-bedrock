# Objective

1. To explore the results from different model

# Features

1. Multi-processing
2. Results from multiple models (cohere, jurassic, claude2, llama13b, llama70b)
3. Response time in seconds
4. Configuration to select which model

# To be added

1. Cost calculation per request for each model
2. Additional model(mistral, sonnet, haiku)

# Getting started

```
> pip install -r requirements.txt
> cp .env.sample .env # Update values inside
> source .env
> streamlit run your_script.py run-foundational-models.py

```


# Screenshots

![say hello](./screenshots/ss-selection-of-model.png "Say hello")