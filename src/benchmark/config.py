"""
Configuration settings for benchmark experiments.

This module defines model configurations, attribution methods, and other
settings used in benchmark experiments.
"""

# Number of prompts to process from dataset (set to None for all)
MAX_PROMPTS = 20  # Adjust based on available compute and time constraints

# Define models to test - starting with smaller subset for initial tests
MODELS = [
    {
        "name": "BART",
        "llm_id": "facebook/bart-large",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "name": "GPT-2",
        "llm_id": "gpt2",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # ----------
    #    {
    #        "name": "Llama-3-2-Instruct-3B",
    #        "llm_id": "meta-llama/Llama-3.2-3B-Instruct",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "meta-llama/Llama-3.2-1B",
    #        "name": "Llama-3-2-1B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "name": "Llama-3-1-8B",
    #        "llm_id": "meta-llama/Llama-3.1-8B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    # ----------
    #    {
    #        "name": "DeepSeek-R1-Distill-Llama-8B",
    #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "name": "DeepSeek-R1-Distill-Qwen-7B",
    #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    #        "name": "DeepSeek-R1-Distill-Qwen-14B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    # ----------
    #    {
    #        "llm_id": "Qwen/Qwen2.5-Omni-7B",
    #        "name": "Qwen2.5-Omni-7B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    #        "name": "Qwen2.5-VL-3B-Instruct",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    #        "name": "Qwen2.5-VL-7B-Instruct",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "Qwen/QwQ-32B",
    #        "name": "QwQ-32B",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    # ----------
    #    {
    #        "llm_id": "mistralai/Mistral-Small-3.1-24B-Base-2503",
    #        "name": "Mistral-Small-3.1-24B-Base-2503",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "mistralai/Mistral-Nemo-Base-2407",
    #        "name": "Mistral-Nemo-Base-2407",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    # ----------
    #    {
    #        "llm_id": "google/gemma-3-27b-it",
    #        "name": "Gemma-3-27b-it",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-27b-pt",
    #        "name": "Gemma-3-27b-pt",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-12b-it",
    #        "name": "Gemma-3-12b-it",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-12b-pt",
    #        "name": "Gemma-3-12b-pt",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-4b-it",
    #        "name": "Gemma-3-4b-it",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-4b-pt",
    #        "name": "Gemma-3-4b-pt",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-1b-it",
    #        "name": "Gemma-3-1b-it",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    },
    #    {
    #        "llm_id": "google/gemma-3-1b-pt",
    #        "name": "Gemma-3-1b-pt",
    #        "device": "auto",
    #        "torch_dtype": "float16",
    #        "type": "causal"
    #    }
]

# Define attribution methods to test
ATTRIBUTION_METHODS = [
    'input_x_gradient',
    'discretized_integrated_gradients',
    'value_zeroing',
    'saliency',
    'lime',
    'integrated_gradients',
    'layer_gradient_x_activation',
    'attention',
    'gradient_shap',
    # Additional methods can be uncommented as needed
    # 'layer_deeplift',
    # 'layer_integrated_gradients',
    # 'deeplift',
    # 'reagent',
    # 'occlusion',
    # 'sequential_integrated_gradients'
]

# Dataset configuration
DATASET_CONFIG = {
    'primary_source': "hf://datasets/google/FACTS-grounding-public/examples.csv",
    'fallback_path': "data/facts_examples.csv",
}