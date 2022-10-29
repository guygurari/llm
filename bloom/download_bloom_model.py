"""
Download the 176B parameter BLOOM model from HuggingFace.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        offload_folder="offloaded_weights",
        )
