#!/usr/bin/env python
"""
Data structure tests for Inseq integration.

This module contains tests that validate the Inseq data structure and its
conversion to the TokenAnalyzer format, particularly for matrices with
different dimensions and token formats.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from work.llama_token_analyzer.persistence.inseq_schema import (
    InseqFeatureAttributionOutput,
    InseqFeatureAttributionSequence,
    AttributionMethod
)
from work.llama_token_analyzer.core.inseq_analysis import InseqTokenAnalyzer
from work.llama_token_analyzer.persistence.schema import AnalysisResult, AssociationData


class TestInseqDataStructure(unittest.TestCase):
    """Test the Inseq data structure and conversion to TokenAnalyzer format"""

    def setUp(self):
        """Set up common test objects"""
        # Create mock model manager
        self.model_manager = MagicMock()
        self.model_manager.config.llm_id = "test/model"
        self.tokenizer = MagicMock()
        self.tokenizer.convert_tokens_to_ids.return_value = 123
        self.model_manager.tokenizer = self.tokenizer

    def test_1d_attribution_matrix(self):
        """Test handling a simple 1D attribution matrix"""
        # Create a sequence attribution with 1D matrix (simplest case)
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world"],
            target=["Bonjour"],
            source_attributions=np.array([[0.1, 0.2]]),  # [1 output token, 2 input tokens]
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world",
            attribution_method="saliency"
        )
        
        # Verify 2D matrix shape and content
        self.assertEqual(len(result.data.input_tokens), 2)  # 2 input tokens
        self.assertEqual(len(result.data.output_tokens), 1)  # 1 output token
        
        # Association matrix should be 2D with shape [1, 2]
        association_matrix = np.array(result.data.association_matrix)
        self.assertEqual(association_matrix.shape, (1, 2))
        self.assertEqual(association_matrix[0, 0], 0.1)
        self.assertEqual(association_matrix[0, 1], 0.2)

    def test_2d_attribution_matrix(self):
        """Test handling a 2D attribution matrix"""
        # Create a sequence attribution with 2D matrix (more complex)
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world", "!"],
            target=["Bonjour", "le", "monde"],
            source_attributions=np.array([
                [0.1, 0.2, 0.0],  # First output token's attributions to each input
                [0.0, 0.3, 0.1],  # Second output token's attributions
                [0.2, 0.1, 0.3]   # Third output token's attributions
            ]),  # [3 output tokens, 3 input tokens]
            attr_pos_start=0,
            attr_pos_end=3
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world!",
            attribution_method="saliency"
        )
        
        # Verify matrix shape and content
        self.assertEqual(len(result.data.input_tokens), 3)  # 3 input tokens
        self.assertEqual(len(result.data.output_tokens), 3)  # 3 output tokens
        
        # Association matrix should be 2D with shape [3, 3]
        association_matrix = np.array(result.data.association_matrix)
        self.assertEqual(association_matrix.shape, (3, 3))
        self.assertEqual(association_matrix[0, 0], 0.1)
        self.assertEqual(association_matrix[1, 1], 0.3)
        self.assertEqual(association_matrix[2, 2], 0.3)

    def test_3d_attribution_matrix(self):
        """Test handling a 3D attribution matrix (like from gradient attribution)"""
        hidden_size = 4  # Small hidden size for testing
        
        # Create a sequence attribution with 3D matrix (gradient-based methods)
        input_tokens = ["Hello", "world"]
        output_tokens = ["Bonjour", "le"]
        
        # Create 3D matrix: [2 output tokens, 2 input tokens, 4 hidden dims]
        source_attributions = np.zeros((2, 2, hidden_size))
        # Fill with some test data
        source_attributions[0, 0] = np.array([0.1, 0.2, 0.3, 0.4])  # First output token -> First input token
        source_attributions[0, 1] = np.array([0.2, 0.1, 0.2, 0.3])  # First output token -> Second input token
        source_attributions[1, 0] = np.array([0.0, 0.1, 0.2, 0.0])  # Second output token -> First input token
        source_attributions[1, 1] = np.array([0.4, 0.3, 0.2, 0.1])  # Second output token -> Second input token
        
        seq_attr = InseqFeatureAttributionSequence(
            source=input_tokens,
            target=output_tokens,
            source_attributions=source_attributions,
            attr_pos_start=0,
            attr_pos_end=2
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world",
            attribution_method="integrated_gradients"
        )
        
        # Verify matrix dimensions are reduced correctly using L2 norm
        association_matrix = np.array(result.data.association_matrix)
        self.assertEqual(association_matrix.shape, (2, 2))  # Should be reduced to 2D
        
        # Verify L2 norm aggregation was done correctly
        expected_0_0 = np.linalg.norm(np.array([0.1, 0.2, 0.3, 0.4]))
        expected_0_1 = np.linalg.norm(np.array([0.2, 0.1, 0.2, 0.3]))
        expected_1_0 = np.linalg.norm(np.array([0.0, 0.1, 0.2, 0.0]))
        expected_1_1 = np.linalg.norm(np.array([0.4, 0.3, 0.2, 0.1]))
        
        np.testing.assert_almost_equal(association_matrix[0, 0], expected_0_0)
        np.testing.assert_almost_equal(association_matrix[0, 1], expected_0_1)
        np.testing.assert_almost_equal(association_matrix[1, 0], expected_1_0)
        np.testing.assert_almost_equal(association_matrix[1, 1], expected_1_1)

    def test_4d_attribution_matrix(self):
        """Test handling a 4D attribution matrix (like from attention attribution)"""
        n_layers = 2
        n_heads = 3
        
        # Create a sequence attribution with 4D matrix (attention attribution)
        input_tokens = ["Hello", "world"]
        output_tokens = ["Bonjour"]
        
        # Create 4D matrix: [1 output token, 2 input tokens, 2 layers, 3 heads]
        source_attributions = np.zeros((1, 2, n_layers, n_heads))
        # Fill with test data
        for l in range(n_layers):
            for h in range(n_heads):
                source_attributions[0, 0, l, h] = 0.1 * (l + 1) * (h + 1)  # First output -> First input
                source_attributions[0, 1, l, h] = 0.2 * (l + 1) * (h + 1)  # First output -> Second input
        
        seq_attr = InseqFeatureAttributionSequence(
            source=input_tokens,
            target=output_tokens,
            source_attributions=source_attributions,
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world",
            attribution_method="attention"
        )
        
        # Verify matrix dimensions are reduced correctly (4D to 2D)
        association_matrix = np.array(result.data.association_matrix)
        self.assertEqual(association_matrix.shape, (1, 2))  # Should be reduced to 2D
        
        # Verify aggregation was applied correctly (L2 norm across last dimensions)
        expected_0_0 = np.linalg.norm(source_attributions[0, 0].flatten())
        expected_0_1 = np.linalg.norm(source_attributions[0, 1].flatten())
        
        np.testing.assert_almost_equal(association_matrix[0, 0], expected_0_0)
        np.testing.assert_almost_equal(association_matrix[0, 1], expected_0_1)

    def test_torch_tensor_attribution(self):
        """Test handling PyTorch tensor attribution matrices"""
        # Create a sequence attribution with PyTorch tensor
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world"],
            target=["Bonjour"],
            source_attributions=torch.tensor([[0.1, 0.2]]),  # PyTorch tensor
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world",
            attribution_method="saliency"
        )
        
        # Verify conversion from torch tensor to numpy array worked
        association_matrix = np.array(result.data.association_matrix)
        self.assertEqual(association_matrix.shape, (1, 2))
        
        # Use numpy's testing functions to handle float precision issues
        np.testing.assert_almost_equal(association_matrix[0, 0], 0.1)
        np.testing.assert_almost_equal(association_matrix[0, 1], 0.2)
        
    def test_normalized_attribution(self):
        """Test normalization of attribution matrices"""
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world", "!"],
            target=["Bonjour", "le"],
            source_attributions=np.array([
                [0.1, 0.2, 0.4],  # First output token's attributions
                [0.3, 0.5, 0.8],  # Second output token's attributions
            ]),
            attr_pos_start=0,
            attr_pos_end=2
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="Hello world!",
            attribution_method="saliency"
        )
        
        # Verify normalized attribution is computed correctly
        normalized_matrix = np.array(result.data.normalized_association)
        self.assertEqual(normalized_matrix.shape, (2, 3))
        
        # Check normalization per column
        for i in range(3):  # For each input token
            col_norm = normalized_matrix[:, i]
            if np.max(col_norm) > 0:  # If column has non-zero values
                self.assertGreaterEqual(np.min(col_norm), 0.0)  # Min should be >= 0
                self.assertLessEqual(np.max(col_norm), 1.0)    # Max should be <= 1
                
    def test_special_tokens_handling(self):
        """Test handling of special tokens"""
        # Create a sequence with special tokens
        seq_attr = InseqFeatureAttributionSequence(
            source=["<s>", "Ġhello", "Ġworld", "</s>"],  # LLaMA-style tokens with special markers
            target=["<s>", "Ġbonjour", "</s>"],
            source_attributions=np.array([
                [0.1, 0.2, 0.3, 0.0],
                [0.0, 0.3, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.1],
            ]),
            attr_pos_start=0,
            attr_pos_end=3
        )
        
        # Convert to token analyzer format
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="hello world",
            attribution_method="saliency"
        )
        
        # Verify token cleaning works correctly
        input_tokens = result.data.input_tokens
        self.assertEqual(input_tokens[0].token, "<s>")
        self.assertEqual(input_tokens[0].clean_token, "<s>")  # Special tokens should be preserved
        
        self.assertEqual(input_tokens[1].token, "Ġhello")
        self.assertEqual(input_tokens[1].clean_token, "hello")  # 'Ġ' should be stripped
        
        self.assertEqual(input_tokens[2].token, "Ġworld")
        self.assertEqual(input_tokens[2].clean_token, "world")  # 'Ġ' should be stripped

    @patch('inseq.load_model')
    def test_inseq_token_analyzer_integration(self, mock_load_model):
        """Test the integration between InseqTokenAnalyzer and the Pydantic models"""
        # Mock the Inseq model
        mock_inseq_model = MagicMock()
        mock_load_model.return_value = mock_inseq_model
        
        # Initialize the analyzer
        analyzer = InseqTokenAnalyzer(self.model_manager)
        
        # Create a fake attribution result with a 3D matrix
        hidden_size = 4
        fake_source_attributions = np.zeros((2, 3, hidden_size))
        for i in range(2):  # 2 output tokens
            for j in range(3):  # 3 input tokens
                fake_source_attributions[i, j] = np.random.rand(hidden_size)
        
        # Create sequence attribution
        seq_attr = InseqFeatureAttributionSequence(
            source=["token1", "token2", "token3"],
            target=["output1", "output2"],
            source_attributions=fake_source_attributions,
            attr_pos_start=0,
            attr_pos_end=2
        )
        
        # Create the output model
        inseq_output = InseqFeatureAttributionOutput(
            sequence_attributions=[seq_attr],
            info={"method": "saliency"}
        )
        
        # Mock the _to_pydantic_model to return our model
        with patch.object(InseqTokenAnalyzer, '_to_pydantic_model', return_value=inseq_output):
            # Test the conversion
            result = analyzer._convert_to_analysis_result(MagicMock(), "test prompt")
            
            # Verify the result
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(len(result.data.input_tokens), 3)
            self.assertEqual(len(result.data.output_tokens), 2)
            
            # Check matrix shape
            association_matrix = np.array(result.data.association_matrix)
            self.assertEqual(association_matrix.shape, (2, 3))
            
            # Check normalization
            normalized_matrix = np.array(result.data.normalized_association)
            self.assertEqual(normalized_matrix.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()