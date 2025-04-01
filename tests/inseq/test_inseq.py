#!/usr/bin/env python
"""
Simple test for the Inseq integration.

This script tests the basic functionality of the Inseq token analyzer
without requiring a full model load, by using a simple mock.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src import InseqTokenAnalyzer
from src import (
    InseqFeatureAttributionOutput,
    InseqFeatureAttributionSequence
)


class TestInseqIntegration(unittest.TestCase):
    """Test cases for the Inseq integration"""

    def setUp(self):
        """Set up test cases"""
        # Create mock model manager
        self.model_manager = MagicMock()
        self.model_manager.config.llm_id = "test/model"
        self.tokenizer = MagicMock()
        self.tokenizer.convert_tokens_to_ids.return_value = 123
        self.model_manager.tokenizer = self.tokenizer
        
        # Mock storage
        self.storage = MagicMock()
        self.storage.save.return_value = None

    @patch('inseq.load_model')
    def test_initialization(self, mock_load_model):
        """Test that the analyzer initializes correctly"""
        # Setup
        mock_inseq_model = MagicMock()
        mock_load_model.return_value = mock_inseq_model
        
        # Execute
        analyzer = InseqTokenAnalyzer(self.model_manager, attribution_method="saliency")
        
        # Assert
        mock_load_model.assert_called_once()
        self.assertEqual(analyzer.attribution_method, "saliency")
        self.assertEqual(analyzer.inseq_model, mock_inseq_model)

    @patch('inseq.load_model')
    def test_convert_attribution(self, mock_load_model):
        """Test conversion from Inseq attribution to AnalysisResult"""
        # Setup
        mock_inseq_model = MagicMock()
        mock_load_model.return_value = mock_inseq_model
        
        # Create proper Pydantic model for Inseq attribution
        seq_attr = InseqFeatureAttributionSequence(
            source=["token1", "token2"],
            target=["token3"],
            source_attributions=np.array([[0.1, 0.2]]),
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        inseq_output = InseqFeatureAttributionOutput(
            sequence_attributions=[seq_attr],
            info={"method": "saliency"}
        )
        
        # Mock the _to_pydantic_model to return our Pydantic model
        with patch.object(InseqTokenAnalyzer, '_to_pydantic_model', return_value=inseq_output):
            # Create analyzer
            analyzer = InseqTokenAnalyzer(self.model_manager)
            
            # Execute
            result = analyzer._convert_to_analysis_result(MagicMock(), "test prompt")
            
            # Assert
            self.assertEqual(len(result.data.input_tokens), 2)
            self.assertEqual(len(result.data.output_tokens), 1)
            self.assertEqual(result.metadata.prompt, "test prompt")
            self.assertEqual(result.metadata.llm_id, "test/model")
            
    def test_pydantic_model_conversion(self):
        """Test the Pydantic model conversion and serialization"""
        # Create a sequence attribution
        seq_attr = InseqFeatureAttributionSequence(
            source=["token1", "token2"],
            target=["token3"],
            source_attributions=np.array([[0.1, 0.2]]),
            attr_pos_start=0,
            attr_pos_end=1
        )
        
        # Create the full attribution output
        inseq_output = InseqFeatureAttributionOutput(
            sequence_attributions=[seq_attr],
            info={"method": "saliency"}
        )
        
        # Test to_token_analyzer_format method
        result = seq_attr.to_token_analyzer_format(
            model_id="test/model",
            prompt="test prompt",
            attribution_method="saliency"
        )
        
        # Assert the conversion worked correctly
        self.assertEqual(len(result.data.input_tokens), 2)
        self.assertEqual(len(result.data.output_tokens), 1)
        self.assertEqual(result.metadata.prompt, "test prompt")
        self.assertEqual(result.metadata.llm_id, "test/model")
        
        # Convert numpy array to list for JSON serialization
        seq_attr.source_attributions = seq_attr.source_attributions.tolist()
        
        # Test that we can convert to JSON
        json_data = inseq_output.model_dump_json()
        self.assertIsInstance(json_data, str)
        self.assertIn("sequence_attributions", json_data)


if __name__ == "__main__":
    unittest.main()