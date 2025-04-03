#!/usr/bin/env python
"""
Unit tests for the Inseq schema conversion logic.

This module tests the conversion from Inseq attribution data to our own schema,
ensuring proper handling of different matrix dimensions and tensor types.
"""

import unittest
import numpy as np
import torch
from typing import List, Tuple

from src import (
    InseqFeatureAttributionOutput,
    InseqFeatureAttributionSequence
)
from src import AnalysisResult


class TestInseqSchemaConversion(unittest.TestCase):
    """Test our Inseq schema conversion with different tensor types and shapes"""

    def _create_test_matrices(self) -> List[Tuple[str, np.ndarray]]:
        """Create test attribution matrices of different dimensions"""
        matrices = []
        
        # 1D matrix
        matrices.append(("1D", np.array([[0.1, 0.2]])))
        
        # 2D matrix
        matrices.append(("2D", np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5]
        ])))
        
        # 3D matrix (like from gradient attribution)
        hidden_size = 4
        matrix_3d = np.zeros((2, 3, hidden_size))
        for i in range(2):
            for j in range(3):
                matrix_3d[i, j] = np.random.rand(hidden_size)
        matrices.append(("3D", matrix_3d))
        
        # 4D matrix (like from attention attribution)
        n_layers = 2
        n_heads = 3
        matrix_4d = np.zeros((1, 2, n_layers, n_heads))
        for l in range(n_layers):
            for h in range(n_heads):
                matrix_4d[0, 0, l, h] = 0.1 * (l + 1) * (h + 1)
                matrix_4d[0, 1, l, h] = 0.2 * (l + 1) * (h + 1)
        matrices.append(("4D", matrix_4d))
        
        return matrices

    def test_different_matrix_dimensions(self):
        """Test handling of different attribution matrix dimensions"""
        for name, matrix in self._create_test_matrices():
            with self.subTest(f"{name} matrix"):
                # Create sequence attribution with the test matrix
                seq_attr = InseqFeatureAttributionSequence(
                    source=["Hello", "world"] + (["!"] if matrix.shape[1] > 2 else []),
                    target=["Bonjour"] * matrix.shape[0],
                    source_attributions=matrix,
                    attr_pos_start=0,
                    attr_pos_end=matrix.shape[0]
                )
                
                # Convert to token analyzer format
                result = seq_attr.to_token_analyzer_format(
                    model_id="test/model",
                    prompt="Test prompt",
                    attribution_method="test_method"
                )
                
                # Check basic structure
                self.assertIsInstance(result, AnalysisResult)
                
                # Check dimensions
                association_matrix = np.array(result.data.association_matrix)
                self.assertEqual(len(association_matrix.shape), 2)
                self.assertEqual(association_matrix.shape[0], matrix.shape[0])
                self.assertEqual(association_matrix.shape[1], matrix.shape[1])
                
                # Check normalization
                normalized = np.array(result.data.normalized_association)
                self.assertEqual(normalized.shape, association_matrix.shape)
                self.assertTrue(np.all(normalized >= 0))
                self.assertTrue(np.all(normalized <= 1))

    def test_torch_tensor_attribution(self):
        """Test handling PyTorch tensor attribution matrices"""
        # Create a sequence attribution with a PyTorch tensor
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world"],
            target=["Bonjour"],
            source_attributions=torch.tensor([[0.1, 0.2]]),
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

    def test_serialization(self):
        """Test serialization of the schema objects"""
        # Create a simple attribution sequence
        seq_attr = InseqFeatureAttributionSequence(
            source=["Hello", "world"],
            target=["Bonjour"],
            source_attributions=[[0.1, 0.2]],  # Use list directly for JSON compatibility
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        # Create the full output model
        inseq_output = InseqFeatureAttributionOutput(
            sequence_attributions=[seq_attr],
            info={"method": "saliency"}
        )
        
        # Test serialization
        json_data = inseq_output.model_dump_json()
        self.assertIsInstance(json_data, str)
        self.assertIn("sequence_attributions", json_data)


if __name__ == "__main__":
    unittest.main()