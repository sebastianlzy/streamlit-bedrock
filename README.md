# Objective

1. Increase our understanding of our model's performance by analyzing the results of various models.
2. Understand our average response time by assessing the latency of each model.

# Demo
![demo](./screenshots/streamlit-bedrock.gif "demo")

# Features

1. Multi-processing
2. Results from multiple models 
3. Response time in seconds
4. Configuration to select which model
5. Ability to perform multimodal

# To be added

1. Cost calculation per request for each model
2. Add more models

# Models available

- [x] cohere
- [x] jurassic
- [x] claude_v2
- [x] llama13b
- [x] llama70b
- [x] titanText
- [x] mistral
- [x] claude sonnet
- [x] claude haiku

# Getting started

```
> pip install -r requirements.txt
> cp .env.sample .env # Update values inside
> source .env
> streamlit run your_script.py run-foundational-models.py

```


# Screenshots

### Selection of models
![selection-of-models](./screenshots/ss-selection-of-model.png "Selection of models")

### Multi-modal
![ss-multi-modal](./screenshots/ss-multi-modal-2.png "Multi-modal")
