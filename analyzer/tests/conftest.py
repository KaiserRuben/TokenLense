import pytest
import os
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Create a fixture for test data directory
@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory structure"""
    # Create model directories
    bart_dir = tmp_path / "BART"
    gpt2_dir = tmp_path / "GPT-2"
    bart_dir.mkdir()
    gpt2_dir.mkdir()
    
    # Create method directories for BART
    bart_attention_dir = bart_dir / "method_attention" / "data"
    bart_saliency_dir = bart_dir / "method_saliency" / "data"
    bart_attention_dir.mkdir(parents=True)
    bart_saliency_dir.mkdir(parents=True)
    
    # Create method directories for GPT-2
    gpt2_attention_dir = gpt2_dir / "method_attention" / "data"
    gpt2_saliency_dir = gpt2_dir / "method_saliency" / "data"
    gpt2_ig_dir = gpt2_dir / "method_integrated_gradients" / "data"
    gpt2_attention_dir.mkdir(parents=True)
    gpt2_saliency_dir.mkdir(parents=True)
    gpt2_ig_dir.mkdir(parents=True)
    
    # Create test files
    bart_attention_file = bart_attention_dir / "20250404_bart-large_Answerthisdo_inseq.json"
    gpt2_attention_file = gpt2_attention_dir / "20250404_gpt2_Answerthisdo_inseq.json"
    
    # Create sample attribution data
    sample_attribution = {
        "__instance_type__": ["inseq.data.attribution", "FeatureAttributionOutput"],
        "attributes": {
            "info": {
                "attr_pos_end": 10,
                "attr_pos_start": 0,
                "attribute_target": True,
                "attributed_fn": "generate",
                "attributed_fn_args": {},
                "attribution_args": {},
                "attribution_method": "attention",
                "constrained_decoding": False,
                "exec_time": 0.45,
                "generate_from_target_prefix": False,
                "generated_texts": ["This is a generated response."],
                "generation_args": {},
                "include_eos_baseline": False,
                "input_texts": ["Answer this do Iran and Afghanistan speak the same language?"],
                "model_class": "BartForConditionalGeneration",
                "model_name": "bart-large",
                "output_step_attributions": False,
                "step_scores": [],
                "step_scores_args": {},
                "tokenizer_class": "BartTokenizer",
                "tokenizer_name": None
            },
            "sequence_attributions": [
                {
                    "__instance_type__": ["inseq.data.attribution", "GranularFeatureAttributionSequenceOutput"],
                    "attributes": {
                        "attr_pos_end": 10,
                        "attr_pos_start": 0,
                        "sequence_scores": {},
                        "source": [
                            {
                                "__instance_type__": ["inseq.utils.typing", "TokenWithId"],
                                "attributes": {
                                    "id": 0,
                                    "token": "<s>"
                                }
                            },
                            {
                                "__instance_type__": ["inseq.utils.typing", "TokenWithId"],
                                "attributes": {
                                    "id": 1,
                                    "token": "Answer"
                                }
                            }
                        ],
                        "source_attributions": None,
                        "step_scores": {},
                        "target": [
                            {
                                "__instance_type__": ["inseq.utils.typing", "TokenWithId"],
                                "attributes": {
                                    "id": 0,
                                    "token": "<s>"
                                }
                            },
                            {
                                "__instance_type__": ["inseq.utils.typing", "TokenWithId"],
                                "attributes": {
                                    "id": 1,
                                    "token": "This"
                                }
                            }
                        ],
                        "target_attributions": {
                            "Corder": True,
                            "__ndarray__": "b64.gz:H4sIAAAAAAAC/2NkYGDkYmTYzcRwkImBkYEBCDjgPCZGIOvJMPYoFdNRKqbDTUyfgTwOoFxFAAgcHBiAvPJTJ2ZB3P/LPLD5QCouRlZGBqCfGGGCyIqOUjEdTibG71DFDHDwfydQITtU3MHBxR0uHogmGNDKxw4RAPt/YFM+AwAA",
                            "dtype": "float32",
                            "shape": [2, 2, 1]
                        }
                    }
                }
            ]
        }
    }
    
    with open(bart_attention_file, 'w') as f:
        json.dump(sample_attribution, f)
    
    with open(gpt2_attention_file, 'w') as f:
        json.dump(sample_attribution, f)
    
    # Create performance metrics files
    method_timing_file = tmp_path / "method_timing_results.csv"
    prompt_timing_file = tmp_path / "prompt_timing_results.csv"
    
    with open(method_timing_file, 'w') as f:
        f.write("model,method,average_time,min_time,max_time,success_rate,tokens_per_second\n")
        f.write("BART,attention,0.1,0.05,0.15,1.0,100.0\n")
        f.write("GPT-2,attention,0.2,0.1,0.3,1.0,80.0\n")
    
    with open(prompt_timing_file, 'w') as f:
        f.write("prompt,model,method,time,success,tokens,tokens_per_second\n")
        f.write("Test prompt,BART,attention,0.1,True,10,100.0\n")
        f.write("Test prompt,GPT-2,attention,0.2,True,16,80.0\n")
    
    # Create system info file
    system_info_file = tmp_path / "system_info.json"
    
    system_info = {
        "cpu": {
            "model": "Test CPU",
            "cores": 8
        },
        "memory": {
            "total": 16384,
            "available": 8192
        },
        "gpu": [
            {
                "model": "Test GPU",
                "memory": 8192
            }
        ],
        "python_version": "3.12.0",
        "os": {
            "name": "Test OS",
            "version": "1.0"
        },
        "timestamp": "2025-04-04T12:00:00Z"
    }
    
    with open(system_info_file, 'w') as f:
        json.dump(system_info, f)
    
    return tmp_path


# Create a fixture to patch the DATA_DIR constant
@pytest.fixture
def patch_data_dir(test_data_dir):
    """Patch the DATA_DIR constant in various modules to use the test directory"""
    data_dir_patches = [
        patch("api.DATA_DIR", test_data_dir),
        patch("routers.attribution.DATA_DIR", test_data_dir),
        patch("routers.models.DATA_DIR", test_data_dir),
        patch("routers.performance.DATA_DIR", test_data_dir)
    ]
    
    for p in data_dir_patches:
        p.start()
    
    yield test_data_dir
    
    for p in data_dir_patches:
        p.stop()


# Create a fixture for a mock numpy tensor
@pytest.fixture
def mock_tensor():
    """Create a mock numpy tensor for attribution data"""
    return np.array([
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]
    ])