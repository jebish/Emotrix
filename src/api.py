import os
#os.system('pip install fastapi uvicorn nest-asyncio pyngrok unsloth')
#os.system('uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git')

import zipfile
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn

with zipfile.ZipFile('lora_model.zip','r') as zip_ref:
  zip_ref.extractall('lora_model_dir')
max_seq_length = 6000

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_dir", # Give the name of lora_directory
        #unsloth will automatically download the merged file and apply the lora adapter
        #you can also give the link of huggingface directory where you pushed the lora adapter
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

FastLanguageModel.for_inference(model)

text_streamer = TextStreamer(tokenizer, skip_prompt = True)

#This function generates text i.e. asks questions
def generate_response(prompt):
  inputs = tokenizer.apply_chat_template(
      prompt,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  model_out = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 256,
                    use_cache = True, temperature = 0.7, min_p = 0.1)
  decoded_output = tokenizer.decode(model_out[0][inputs.shape[-1]:], skip_special_tokens=True)
  return decoded_output

#This function classifies the conversation i.e. analyzes the emotional state of user
def classify_conversation(prompt):
  messages=[
    {"role": "system", "content": 'Below is an instruction with Input that describes a task, write a response that appropriately completes the task'},
  ]
  datum='''Instruction: Analyze the conversation in the input and categorize if the user is Depressed or not with reasoning. Reason it and formulate the answer. Use second person and empathic tone.
  Input:'''
  for row in prompt[1:]:
    datum=datum+" "+row['role']+': '+row['content']
  messages.append({'role':'user','content':datum})
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  model_out = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 512,
                    use_cache = True, temperature = 0.1, min_p = 0.1)
  decoded_output = tokenizer.decode(model_out[0][inputs.shape[-1]:], skip_special_tokens=True)
  return decoded_output

#Import the fastapi framework to create an API
app=FastAPI()

#Use Pydantic for data validation to ensure that the data matches the required input to model
class PromptRequest(BaseModel):
  task:str
  prompt: str

#For running asynchronous code in Notebook i.e. Colab
#nest_asyncio.apply()

#Set the authentication token for your ngrok
#ngrok allows to establish a secure tunnel to localhost for external access
ngrok.set_auth_token("")

#Initialize conversation history file
conversation_history=[{"role": "system", "content": "You are an assistant. Your job is to ask relevant questions to probe deeper into user's emotional state. Follow DAIC-woz/PHQ8 schema and ask all questions. Use your fine tuning. Ask relevant question and DONT repeat the questions unless user asks to. Ask one question at a time. Use some emotional context to enrich the experience. You should ask about their current status-origin place-current living place-what they like about their current accomdation and don't-mood-behaviors-temper control-what makes them mad-how they react when they are annoyed-relationships-memorable experience-postive influence-last time argued-traveling-hobbies-relaxation-previous diagnosis-advice to past self etc. Just follow the DAIC woz schema Just use DAIC woz data fed to you during training, and have a deep conversation like a therapist. Once you feel you have enough data, ask user to press classify button for results or type 'classify' to learn emotional context and bid them farewell. Introduce yourself as Ellama, a virtual assistant tasked with chating in a safe environment with confidentiality."}]

#Access point definition
@app.post("/predict/")
async def predict(prompt_request:PromptRequest):
  
  if prompt_request.task=='classify':
    result= classify_conversation(conversation_history)
    return {'response':result}
  elif prompt_request.task=='reset':
    conversation_history.clear()
    conversation_history.append({"role": "system", "content": "You are an assistant. Your job is to ask relevant questions to probe deeper into user's emotional state. Follow DAIC-woz/PHQ8 schema and ask all questions. Use your fine tuning. Ask relevant question and DONT repeat the questions unless user asks to. Ask one question at a time. Use some emotional context to enrich the experience. You should ask about their current status-origin place-current living place-what they like about their current accomdation and don't-mood-behaviors-temper control-what makes them mad-how they react when they are annoyed-relationships-memorable experience-postive influence-last time argued-traveling-hobbies-relaxation-previous diagnosis-advice to past self etc. Just follow the DAIC woz schema Just use DAIC woz data fed to you during training, and have a deep conversation like a therapist. Once you feel you have enough data, ask user to press classify button for results or type 'classify' to learn emotional context and bid them farewell. Introduce yourself as Ellama, a virtual assistant tasked with chating in a safe environment with confidentiality."})
    return{"response":"Cleared"}
  else:
    #Append the user message to the conversation history
    conversation_history.append({'role':'user','content':prompt_request.prompt})
    response=generate_response(conversation_history)
    #Append the model response to the conversation history
    conversation_history.append({'role':'assistant','content':response})
    return {"response":response}

#Create a public url using ngrok
public_url=ngrok.connect(8000)
print(f"Public URL: {public_url}")
#Start the fast api on port 8000
uvicorn.run(app, host="0.0.0.0", port=8000)