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

## High Level Next Steps
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

## Installation
To begin this project, use the included `Makefile`

#### Creating Virtual Environment

This package is built using `python-3.8`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.

#### pre-commit

`pre-commit` will automatically format and lint your code. You can install using this by using
`make use-pre-commit`. It will take effect on your next `git commit`

#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files. 

## Usage Instructions


# Data Source
## Code Structure
## Artifacts Location

# Results
## Metrics Used
## Evaluation Results
