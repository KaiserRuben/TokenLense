#!/usr/bin/env python
"""
Unit tests for the InseqTokenAnalyzer.

This module tests the Inseq token analyzer implementation,
focusing on the API compatibility and proper schema conversion.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from returns.result import Success

from src import InseqTokenAnalyzer
from src import AnalysisResult
from src.persistence.schema import AssociationData, TokenData, AnalysisMetadata


class MockInseqAttributionResult:
    """Mock implementation of Inseq's attribution result"""
    def __init__(self, source_tokens, target_tokens, matrix):
        self.sequence_attributions = [
            MockSequenceAttribution(source_tokens, target_tokens, matrix)
        ]
        self.info = {"method": "mock_method"}
        
        
class MockSequenceAttribution:
    """Mock implementation of Inseq's sequence attribution"""
    def __init__(self, source, target, source_attributions):
        self.source = source
        self.target = target
        self.source_attributions = source_attributions
        self.target_attributions = None
        self.step_scores = {}
        self.sequence_scores = None
        self.attr_pos_start = 0
        self.attr_pos_end = len(target)


class TestInseqTokenAnalyzer(unittest.TestCase):
    """Test the Inseq token analyzer integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model manager
        self.model_manager = MagicMock()
        self.model_manager.config.llm_id = "mock/model"
        self.model_manager.model = "mock/model"
        
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids = lambda token: 42
        self.model_manager.tokenizer = tokenizer
        
        # Mock storage
        self.storage = MagicMock()
        self.storage.save.return_value = None
        
        # A sample prompt for testing
        self.test_prompt = "Hello, world!"
        
        # Sample source/target tokens and attribution matrix
        self.source_tokens = ["Hello", ",", "world", "!"]
        self.target_tokens = ["Hi", "there", "!"]
        self.attribution_matrix = np.random.rand(len(self.target_tokens), len(self.source_tokens))
        
        # Sample mock attribution result
        self.mock_attribution = MockInseqAttributionResult(
            self.source_tokens,
            self.target_tokens,
            self.attribution_matrix
        )

    @patch('inseq.load_model')
    def test_initialization(self, mock_load_model):
        """Test that the analyzer initializes correctly"""
        # Create mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Create analyzer with default method
        analyzer = InseqTokenAnalyzer(self.model_manager)
        
        # Check proper initialization
        self.assertEqual(analyzer.attribution_method, "saliency")
        mock_load_model.assert_called_once()
        
        # Create analyzer with custom method
        analyzer = InseqTokenAnalyzer(self.model_manager, attribution_method="attention")
        self.assertEqual(analyzer.attribution_method, "attention")
        
        # Test with invalid method - should default to saliency
        analyzer = InseqTokenAnalyzer(self.model_manager, attribution_method="invalid_method")
        self.assertEqual(analyzer.attribution_method, "saliency")

    @patch('inseq.load_model')
    def test_analysis_pipeline(self, mock_load_model):
        """Test the analysis pipeline creation and execution"""
        # Setup the mock model
        mock_model = MagicMock()
        mock_model.attribute.return_value = self.mock_attribution
        mock_load_model.return_value = mock_model
        
        # Create analyzer
        analyzer = InseqTokenAnalyzer(self.model_manager)
        
        # Create a valid AnalysisResult for the mock to return

        # Create a proper result object
        analysis_result = AnalysisResult(
            metadata=AnalysisMetadata(
                llm_id="test-model",
                llm_version="1.0",
                prompt="test prompt",
                generation_params={}
            ),
            data=AssociationData(
                input_tokens=[TokenData(token="test", token_id=1, clean_token="test")],
                output_tokens=[TokenData(token="result", token_id=2, clean_token="result")],
                association_matrix=[[0.5]],
                normalized_association=[[1.0]]
            )
        )
        analyzer._convert_to_analysis_result = MagicMock(return_value=analysis_result)
        
        # Create pipeline
        analyze = analyzer.create_analysis_pipeline(self.storage)
        
        # Run pipeline
        result = analyze(self.test_prompt)
        
        # The result is a Success container
        self.assertIsInstance(result, Success)
        
        # Get the inner value regardless of nesting
        value = result.unwrap()
        # Handle potential double-wrapping with Success containers
        if isinstance(value, Success):
            value = value.unwrap()
            
        # Now compare with the analysis_result
        self.assertIsInstance(value, AnalysisResult)
        self.assertEqual(value.metadata.prompt, analysis_result.metadata.prompt)
        self.assertEqual(value.metadata.llm_id, analysis_result.metadata.llm_id)
        
        # Verify structure matches
        self.assertEqual(len(value.data.input_tokens), len(analysis_result.data.input_tokens))
        self.assertEqual(len(value.data.output_tokens), len(analysis_result.data.output_tokens))
        
        # Verify storage was called
        self.storage.save.assert_called_once()
        

    @patch('inseq.load_model')  
    def test_convert_to_analysis_result(self, mock_load_model):
        """Test conversion from Inseq attribution to AnalysisResult"""
        # Setup mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Create analyzer
        analyzer = InseqTokenAnalyzer(self.model_manager)
        
        # Test conversion
        result = analyzer._convert_to_analysis_result(self.mock_attribution, self.test_prompt)
        
        # Verify result structure
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.metadata.prompt, self.test_prompt)
        self.assertEqual(result.metadata.llm_id, self.model_manager.config.llm_id)
        
        # Verify tokens and matrix
        self.assertEqual(len(result.data.input_tokens), len(self.source_tokens))
        self.assertEqual(len(result.data.output_tokens), len(self.target_tokens))
        
        # Check matrix
        matrix = np.array(result.data.association_matrix)
        self.assertEqual(matrix.shape, self.attribution_matrix.shape)


if __name__ == "__main__":
    unittest.main()