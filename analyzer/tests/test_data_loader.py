import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from data_loader import (
    decode_ndarray,
    load_attribution_file,
    aggregate_attribution,
    get_available_models_and_methods,
    extract_tokens_and_attributions
)


# Test decode_ndarray function
def test_decode_ndarray():
    # Create mock encoded data - this is not real encoded data but for testing purposes
    # In a real test, you would use a valid base64 gzipped array
    with patch("base64.b64decode") as mock_b64decode:
        with patch("gzip.decompress") as mock_decompress:
            # Set up mocks
            mock_b64decode.return_value = b"mock_binary_data"
            mock_decompress.return_value = b"mock_decompressed_data"
            
            # Create mock np.frombuffer that returns a test array
            with patch("numpy.frombuffer") as mock_frombuffer:
                mock_frombuffer.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                
                # Test the function
                result = decode_ndarray("b64.gz:test_encoded_data", "float32", [2, 2])
                
                # Check calls were made correctly
                mock_b64decode.assert_called_once_with("test_encoded_data")
                mock_decompress.assert_called_once_with(b"mock_binary_data")
                mock_frombuffer.assert_called_once()
                
                # Check the result
                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 2)
                
                # Without the b64.gz: prefix
                mock_b64decode.reset_mock()
                mock_decompress.reset_mock()
                mock_frombuffer.reset_mock()
                
                result = decode_ndarray("test_encoded_data", "float32", [2, 2])
                
                mock_b64decode.assert_called_once_with("test_encoded_data")


# Test load_attribution_file function
def test_load_attribution_file():
    # Mock file content
    file_content = json.dumps({
        "attributes": {
            "info": {"key": "value"},
            "sequence_attributions": [
                {
                    "attributes": {
                        "target_attributions": {
                            "__ndarray__": "test_encoded_data",
                            "dtype": "float32",
                            "shape": [2, 2, 4]
                        }
                    }
                }
            ]
        }
    })
    
    # Mock open to return our file content
    with patch("builtins.open", mock_open(read_data=file_content)):
        # Mock decode_ndarray to return a test tensor
        with patch("data_loader.decode_ndarray") as mock_decode:
            mock_decode.return_value = np.ones((2, 2, 4))
            
            # Test the function
            result = load_attribution_file("mock/path/file.json")
            
            # Check the result
            assert "attributes" in result
            assert "info" in result["attributes"]
            assert "sequence_attributions" in result["attributes"]
            
            # Check that the tensor was decoded and stored
            target_attr = result["attributes"]["sequence_attributions"][0]["attributes"]["target_attributions"]
            assert "tensor" in target_attr
            assert isinstance(target_attr["tensor"], np.ndarray)
            assert target_attr["tensor"].shape == (2, 2, 4)


# Test aggregate_attribution function
def test_aggregate_attribution():
    # Create test tensor (3D)
    tensor_3d = np.ones((3, 4, 5))
    
    # Test sum aggregation
    result = aggregate_attribution(tensor_3d, "sum", False)
    assert result.shape == (3, 4)
    assert np.all(result == 5.0)  # Sum of ones across dim 2 (size 5)
    
    # Test mean aggregation
    result = aggregate_attribution(tensor_3d, "mean", False)
    assert result.shape == (3, 4)
    assert np.all(result == 1.0)  # Mean of ones
    
    # Test L2 norm aggregation
    result = aggregate_attribution(tensor_3d, "l2_norm", False)
    assert result.shape == (3, 4)
    assert np.allclose(result, np.sqrt(5.0))  # sqrt(sum of squares of ones)
    
    # Test absolute sum aggregation
    result = aggregate_attribution(tensor_3d, "abs_sum", False)
    assert result.shape == (3, 4)
    assert np.all(result == 5.0)  # Sum of absolute ones
    
    # Test max aggregation
    result = aggregate_attribution(tensor_3d, "max", False)
    assert result.shape == (3, 4)
    assert np.all(result == 1.0)  # Max of ones
    
    # Create test tensor (4D for attention)
    tensor_4d = np.ones((3, 4, 5, 6))
    
    # Test sum aggregation for attention
    result = aggregate_attribution(tensor_4d, "sum", True)
    assert result.shape == (3, 4)
    assert np.all(result == 30.0)  # Sum over dims 2 and 3 (sizes 5, 6)
    
    # Test with invalid method
    with pytest.raises(ValueError):
        aggregate_attribution(tensor_3d, "invalid_method", False)


# Test get_available_models_and_methods function
def test_get_available_models_and_methods(tmp_path):
    # Create test directory structure
    model1_dir = tmp_path / "model1"
    model1_dir.mkdir()
    (model1_dir / "method_attention" / "data").mkdir(parents=True)
    (model1_dir / "method_saliency" / "data").mkdir(parents=True)
    
    model2_dir = tmp_path / "model2"
    model2_dir.mkdir()
    (model2_dir / "method_attention" / "data").mkdir(parents=True)
    
    # Add a non-method directory
    (model1_dir / "not_a_method").mkdir()
    
    # Add a hidden directory
    (tmp_path / ".hidden_dir").mkdir()
    
    # Test the function
    with patch("os.listdir") as mock_listdir:
        mock_listdir.side_effect = lambda path: {
            str(tmp_path): ["model1", "model2", ".hidden_dir"],
            str(model1_dir): ["method_attention", "method_saliency", "not_a_method"],
            str(model2_dir): ["method_attention"]
        }[path]
        
        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.return_value = True
            
            result = get_available_models_and_methods(str(tmp_path))
            
            assert "model1" in result
            assert "model2" in result
            assert ".hidden_dir" not in result
            
            assert "attention" in result["model1"]
            assert "saliency" in result["model1"]
            assert "attention" in result["model2"]
            assert "not_a_method" not in result["model1"]


# Test extract_tokens_and_attributions function
def test_extract_tokens_and_attributions():
    # Create test data
    data = {
        "attributes": {
            "sequence_attributions": [
                {
                    "attributes": {
                        "source": [
                            {"attributes": {"token": "token1", "id": 1}},
                            {"attributes": {"token": "token2", "id": 2}}
                        ],
                        "target": [
                            {"attributes": {"token": "target1", "id": 101}},
                            {"attributes": {"token": "target2", "id": 102}}
                        ],
                        "target_attributions": {
                            "tensor": np.ones((2, 2, 4))
                        }
                    }
                }
            ]
        }
    }
    
    # Test with default aggregation
    source_tokens, target_tokens, matrix = extract_tokens_and_attributions(data)
    
    assert source_tokens == ["token1", "token2"]
    assert target_tokens == ["target1", "target2"]
    assert matrix.shape == (2, 2)
    assert np.all(matrix == 4.0)  # Sum of ones across dim 2 (size 4)
    
    # Test with mean aggregation
    _, _, matrix = extract_tokens_and_attributions(data, "mean")
    assert np.all(matrix == 1.0)  # Mean of ones
    
    # Test with empty sequence_attributions
    data["attributes"]["sequence_attributions"] = []
    source_tokens, target_tokens, matrix = extract_tokens_and_attributions(data)
    assert source_tokens == []
    assert target_tokens == []
    assert matrix is None
    
    # Test with missing tensor
    data["attributes"]["sequence_attributions"] = [
        {
            "attributes": {
                "source": [{"attributes": {"token": "token1", "id": 1}}],
                "target": [{"attributes": {"token": "target1", "id": 101}}],
                "target_attributions": {}  # Missing tensor
            }
        }
    ]
    source_tokens, target_tokens, matrix = extract_tokens_and_attributions(data)
    assert source_tokens == ["token1"]
    assert target_tokens == ["target1"]
    assert matrix is None