# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 23:30:08 2025

@author: tehre
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/MelloGPT-AWQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/MelloGPT-AWQ")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch()
