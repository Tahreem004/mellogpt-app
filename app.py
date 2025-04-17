# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 23:30:08 2025

@author: tehre
"""

import gradio as gr
from transformers import pipeline

# Load the model from Hugging Face Hub
model_name = "TheBloke/MelloGPT-AWQ"  # This is the model you're using.
generator = pipeline('text-generation', model=model_name)

def generate_response(prompt):
    return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Create a Gradio interface for user interaction
iface = gr.Interface(fn=generate_response, 
                     inputs="text", 
                     outputs="text", 
                     title="MelloGPT for Mental Health")

# Launch the interface
iface.launch()
