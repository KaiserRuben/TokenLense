import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import inseq_processor
from inseq_processor import (
    AggregationMethod,
    load_attribution,
    get_attribution_metadata,
    get_tokens_and_ids,
    get_attribution_matrix,
    process_attribution_file,
    compare_attributions,
    compare_token_importance
)


# Test the AggregationMethod enum
def test_aggregation_method_enum():
    assert AggregationMethod.SUM == "sum"
    assert AggregationMethod.MEAN == "mean"
    assert AggregationMethod.L2_NORM == "vnorm"
    assert AggregationMethod.MAX == "max"
    assert AggregationMethod.MIN == "min"
    assert AggregationMethod.ABS_MAX == "absmax"
    assert AggregationMethod.PROD == "prod"


# Mock inseq.FeatureAttributionOutput class for testing
class MockFeatureAttributionOutput:
    def __init__(self, info=None, sequence_attributions=None):
        self.info = info or {
            "model_name": "test-model",
            "attribution_method": "attention",
            "exec_time": 0.5,
            "input_texts": ["Test input"],
            "generated_texts": ["Test output"],
            "tokenizer_class": "TestTokenizer",
            "model_class": "TestModel"
        }
        
        self.sequence_attributions = sequence_attributions or [
            MockSequenceAttributionOutput()
        ]
    
    @classmethod
    def load(cls, path):
        return cls()


class MockSequenceAttributionOutput:
    def __init__(self):
        self.attributes = {
            "source": [
                MockTokenWithId(0, "<s>"),
                MockTokenWithId(1, "Test")
            ],
            "target": [
                MockTokenWithId(0, "<s>"),
                MockTokenWithId(1, "output")
            ],
            "target_attributions": {
                "tensor": np.zeros((2, 2, 4)),
                "__ndarray__": "mock_base64_data",
                "dtype": "float32",
                "shape": [2, 2, 4]
            }
        }


class MockTokenWithId:
    def __init__(self, id, token):
        self.attributes = {
            "id": id,
            "token": token
        }


# Test load_attribution function
@patch("inseq.FeatureAttributionOutput.load")
@patch("pathlib.Path.exists")
def test_load_attribution(mock_exists, mock_load):
    mock_exists.return_value = True  # Make it look like the file exists
    mock_load.return_value = MockFeatureAttributionOutput()
    
    result = load_attribution("mock/path/file.json")
    
    mock_load.assert_called_once_with("mock/path/file.json")
    assert result == mock_load.return_value


# Test get_attribution_metadata function
def test_get_attribution_metadata():
    mock_attr = MockFeatureAttributionOutput()
    
    metadata = get_attribution_metadata(mock_attr)
    
    assert metadata["model_name"] == "test-model"
    assert metadata["attribution_method"] == "attention"
    assert metadata["exec_time"] == 0.5
    assert metadata["prompt"] == "Test input"
    assert metadata["generation"] == "Test output"
    assert metadata["tokenizer_class"] == "TestTokenizer"
    assert metadata["model_class"] == "TestModel"


# Test get_tokens_and_ids function
def test_get_tokens_and_ids():
    mock_attr = MockFeatureAttributionOutput()
    
    source_tokens, target_tokens = get_tokens_and_ids(mock_attr)
    
    assert len(source_tokens) == 2
    assert source_tokens[0]["id"] == 0
    assert source_tokens[0]["token"] == "<s>"
    assert source_tokens[1]["id"] == 1
    assert source_tokens[1]["token"] == "Test"
    
    assert len(target_tokens) == 2
    assert target_tokens[0]["id"] == 0
    assert target_tokens[0]["token"] == "<s>"
    assert target_tokens[1]["id"] == 1
    assert target_tokens[1]["token"] == "output"


# Test get_attribution_matrix function
def test_get_attribution_matrix():
    mock_attr = MockFeatureAttributionOutput()
    
    # Set up a test tensor
    test_tensor = np.ones((2, 2, 4), dtype=np.float32)  # Explicitly specify float32
    mock_attr.sequence_attributions[0].attributes["target_attributions"]["tensor"] = test_tensor
    
    matrix, info = get_attribution_matrix(mock_attr, AggregationMethod.SUM)
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)  # Should be 2D after aggregation
    assert info["shape"] == [2, 2, 4]
    assert info["dtype"] == "float32"
    assert info["tensor_type"] == "attention"
    
    # Test other aggregation methods
    matrix, _ = get_attribution_matrix(mock_attr, AggregationMethod.MEAN)
    assert isinstance(matrix, np.ndarray)
    
    # Test with missing tensor
    del mock_attr.sequence_attributions[0].attributes["target_attributions"]["tensor"]
    
    # Mock the decode_ndarray function
    with patch("data_loader.decode_ndarray") as mock_decode:
        mock_decode.return_value = test_tensor
        matrix, info = get_attribution_matrix(mock_attr, AggregationMethod.SUM)
        assert isinstance(matrix, np.ndarray)


# Test process_attribution_file function
@patch("inseq_processor.load_attribution")
@patch("inseq_processor.get_attribution_metadata")
@patch("inseq_processor.get_tokens_and_ids")
@patch("inseq_processor.get_attribution_matrix")
def test_process_attribution_file(mock_matrix, mock_tokens, mock_metadata, mock_load):
    # Set up mocks
    mock_load.return_value = MockFeatureAttributionOutput()
    
    mock_metadata.return_value = {
        "model_name": "test-model",
        "attribution_method": "attention",
        "exec_time": 0.5,
        "prompt": "Test input",
        "generation": "Test output"
    }
    
    mock_tokens.return_value = (
        [{"id": 0, "token": "<s>"}, {"id": 1, "token": "Test"}],
        [{"id": 0, "token": "<s>"}, {"id": 1, "token": "output"}]
    )
    
    test_matrix = np.ones((2, 2))
    mock_matrix.return_value = (test_matrix, {
        "shape": [2, 2, 4],
        "dtype": "float32",
        "is_attention": True,
        "tensor_type": "attention"
    })
    
    # Call the function
    result = process_attribution_file("mock/path/file.json", AggregationMethod.SUM)
    
    # Check the result
    assert "metadata" in result
    assert "source_tokens" in result
    assert "target_tokens" in result
    assert "attribution_matrix" in result
    assert "matrix_info" in result
    assert "filename" in result
    assert "timestamp" in result
    assert "aggregation" in result
    
    # Check the matrix was properly converted to list
    assert isinstance(result["attribution_matrix"], list)
    assert len(result["attribution_matrix"]) == 2
    assert len(result["attribution_matrix"][0]) == 2


# Test compare_attributions function
@patch("inseq_processor.process_attribution_file")
def test_compare_attributions(mock_process):
    # Set up mock
    mock_process.side_effect = [
        {
            "metadata": {"model_name": "model1", "attribution_method": "method1"},
            "source_tokens": [{"id": 0, "token": "<s>"}],
            "attribution_matrix": [[0.1, 0.2]],
            "matrix_info": {"shape": [2, 2, 4]}
        },
        {
            "metadata": {"model_name": "model2", "attribution_method": "method2"},
            "source_tokens": [{"id": 0, "token": "<s>"}],
            "attribution_matrix": [[0.3, 0.4]],
            "matrix_info": {"shape": [2, 2, 4]}
        }
    ]
    
    # Call the function
    result = compare_attributions(
        ["mock/path/file1.json", "mock/path/file2.json"],
        AggregationMethod.SUM
    )
    
    # Check the result
    assert "results" in result
    assert "aggregation" in result
    assert "count" in result
    assert result["count"] == 2
    assert len(result["results"]) == 2


# Test compare_token_importance function
@patch("inseq_processor.load_attribution")
@patch("inseq_processor.get_attribution_matrix")
@patch("inseq_processor.get_attribution_metadata")
@patch("inseq_processor.get_tokens_and_ids")
def test_compare_token_importance(mock_tokens, mock_metadata, mock_matrix, mock_load):
    # Set up mocks
    mock_load.return_value = MockFeatureAttributionOutput()
    
    mock_metadata.return_value = {
        "model_name": "test-model",
        "attribution_method": "attention"
    }
    
    mock_tokens.return_value = (
        [{"id": 0, "token": "<s>"}, {"id": 1, "token": "Test"}],
        [{"id": 0, "token": "<s>"}, {"id": 1, "token": "output"}]
    )
    
    test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_matrix.return_value = (test_matrix, {
        "shape": [2, 2, 4],
        "dtype": "float32",
        "is_attention": True,
        "tensor_type": "attention"
    })
    
    # Call the function for target token
    result = compare_token_importance(
        ["mock/path/file.json"],
        token_index=1,
        is_target=True,
        aggregation_method=AggregationMethod.SUM
    )
    
    # Check the result
    assert "token_index" in result
    assert "is_target" in result
    assert "results" in result
    assert "count" in result
    assert "aggregation" in result
    assert result["token_index"] == 1
    assert result["is_target"] is True
    assert len(result["results"]) == 1
    
    # Check the token result
    token_result = result["results"][0]
    assert "model" in token_result
    assert "method" in token_result
    assert "token" in token_result
    assert "influences" in token_result
    assert "sum_influence" in token_result
    assert "max_influence" in token_result
    
    # Call the function for source token
    result = compare_token_importance(
        ["mock/path/file.json"],
        token_index=1,
        is_target=False,
        aggregation_method=AggregationMethod.SUM
    )
    
    assert result["is_target"] is False
    assert len(result["results"]) == 1