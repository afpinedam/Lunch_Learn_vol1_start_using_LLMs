# LLM Chatbot Console Tool

Python script to interact with language models (LLMs) from the console, allowing cost estimation and response customization.

---

## Description

The program allows you to:

- Enter questions through the console  
- Estimate input tokens  
- Compare costs between models  
- Select a model  
- Configure token limits  
- Adjust creativity (temperature)  
- Get the model's response  

---

## Technique

- Estimation: **1 token ≈ 4 characters**
- Cost: calculated based on the price per **1M tokens**
- Parameters:
  - `max_tokens`: response length  
  - `temperature`: creativity level  

---

## Architecture

main()  
├── estimate_tokens()  
├── estimate_cost()  
├── select_model()  
├── select_max_tokens()  
├── select_temperature()  
└── query_model()

# Configuration

Add the key to the `.env` file:

# Models

Input pricing model according to the OpenAI pricing page (https://platform.openai.com/pricing):

gpt-4o-mini   $0.15 / 1M  
gpt-4.1-mini  $0.40 / 1M  
gpt-4.1       $2.00 / 1M  
gpt-5-nano    $0.05 / 1M  
gpt-5-mini    $0.25 / 1M
