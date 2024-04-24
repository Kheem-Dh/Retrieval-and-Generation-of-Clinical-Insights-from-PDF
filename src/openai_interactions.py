import openai
import os

# Upload a training file to OpenAI
def upload_training_file(file_path, api_key):
    openai.api_key = api_key
    with open(file_path, 'rb') as f:
        response = openai.File.create(
            file=f,
            purpose="fine-tune"
        )
    return response
