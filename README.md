# Emotrix

## Introduction
Emotrix is an LLM-powered conversational agent that chats with the user and generates emotional analysis.<br>
It is powered by a continued fine-tuned LLaMA 3.2 3B 4-bit instruct model.

## Goals
The project's goal is to create a conversational tool that can not only chat with users by asking relevant question but also generate an emotional analysis from it.<br>
The second goal is to create a model that can generate synthetic conversation which can later be used for training more Mental Health models either using generated analysis or human annotation.
## Contributors
-  [Jebish Purbey](https://github.com/jebish)

## Project Architecture
-  LLaMA 3.2 3B 4-bit Instruct model<br>
-  Unsloth for faster inference<br>
-  FastAPI for API<br>
-  ngrok for Public URL<br>
-  nest_asyncio for asynchronous operation in Colab

# Status
-  Conversational agent ✅<br>
-  Emotional Analysis ✅<br>
-  Ready to generate synthetic dataset ✅<br>
-  API ✅<br>
-  Simple frontend ✅

## Known Issue
-  Does classification even with small or no conversation history<br>
-  Doesn't generate classification tag sometimes
-  Only generates classification tag sometimes

## High-Level Next Steps
-  Employ RLHF to align the output
-  Train on interpretable mental health dataset

# Usage
-  Install the required dependencies
-  Request access to the datasets
-  Use dataset.ipynb to download dataset text and audio files
-  Use text.ipynb to prepare the dataset for text classification training
-  Use instruct.ipynb to prepare the dataset for text generation training
-  Use generation_tuning.ipynb to fine-tune LLaMA 3.2 3B 4-bit instruct model for text generation and save it
-  Use classification_tuning.ipynb to fine-tune the saved LLaMA 3.2 3B 4-bit instruct model for classification and save the adapters
-  Run api.ipynb or api.py to create a backend/API
-  Run command -streamlit run app.py to start the streamlit session and chat with the bot

# Results
## Metrics Used
## Evaluation Results
