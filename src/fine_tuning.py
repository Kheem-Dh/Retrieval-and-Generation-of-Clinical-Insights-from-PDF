import openai
import os

# This code was used to fine-tune a model with a JSONL file
"""
# Fine-tune the OpenAI model
def fine_tune_model(training_file, model, api_key):
    openai.api_key = api_key
    response = openai.FineTuning.create(
        training_file=training_file,
        model=model,
        suffix="clinicalQA"
    )
    return response
"""
