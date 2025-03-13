import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from returns.result import Result, Success, Failure
import inseq


@dataclass
class AttributionConfig:
    """Configuration for attribution analysis"""
    output_data_dir: str = "output/data"
    output_figures_dir: str = "output/figures"
    default_max_tokens: int = 50

    def __post_init__(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.output_data_dir, exist_ok=True)
        os.makedirs(self.output_figures_dir, exist_ok=True)


class AttributionAnalyzer:
    """Analyzes LLM behavior using different attribution methods"""

    def __init__(self, model_manager, config: Optional[AttributionConfig] = None):
        """
        Initialize the analyzer with a model manager and optional config

        Args:
            model_manager: Initialized ModelManager instance
            config: Optional AttributionConfig instance
        """
        self.manager = model_manager
        self.config = config or AttributionConfig()

        # Default attribution methods and their configurations
        self.methods: List[Tuple[str, dict]] = [
            ("saliency", {}),
            ("integrated_gradients", {"n_steps": 100, "dtype":torch.float32}),
            ("input_x_gradient", {}),
            ("attention", {}),
            ("occlusion", {})
        ]

        # Default test cases
        self.test_cases: Dict[str, str] = {
            "logical": "If we know that all A are B, and all B are C, then what can we conclude about A and C?",
            "world_knowledge": "The Great Wall of China was built over many centuries. The main dynasty responsible for its current form was",
            "instruction": "Write a haiku about autumn, following the strict 5-7-5 syllable pattern.",
            "ambiguity": "The trophy doesn't fit in the brown suitcase because it's too",
            "math_reasoning": "To solve this step by step: A store offers a 20% discount on a $80 item, then adds 8% tax. The final price is"
        }

    def run_attribution_test(
            self,
            method_name: str,
            prompt: str,
            method_kwargs: Optional[dict] = None,
            max_new_tokens: Optional[int] = None,
            return_show_html=False
    ) -> Result:
        try:
            method_kwargs = method_kwargs or {}
            max_new_tokens = max_new_tokens or self.config.default_max_tokens

            print(f"\nTesting {method_name} on: {prompt}")

            # Determine correct device and dtype
            device = self.manager.device if type(self.manager.device) is str else self.manager.device.type

            # Force float32 for MPS device and integrated_gradients
            if device == "mps" or method_name == "integrated_gradients":
                dtype = torch.float32
            else:
                dtype = None

            model = inseq.load_model(
                self.manager.model,
                method_name,
                tokenizer=self.manager.tokenizer,
                device=device,
            )

            # Run attribution
            output = model.attribute(
                prompt,
                generation_args={"max_new_tokens": max_new_tokens},
                dtype=dtype,
                **method_kwargs
            )

            # Save results
            output.save(Path(
                f"{self.config.output_data_dir}/{method_name}_output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"))
            if return_show_html:
                graph_html = output.show(return_html=True)
                with open(f"{self.config.output_figures_dir}/{method_name}_output.html", "w") as f:
                    f.write(graph_html)

            return Success(output)

        except Exception as e:
            return Failure(f"Attribution test failed: {str(e)}")

    def run_all_tests(self, custom_test_cases: Optional[Dict[str, str]] = None) -> Dict[str, List[Result]]:
        """
        Run all attribution methods on all test cases

        Args:
            custom_test_cases: Optional dictionary of custom test cases to run instead of defaults

        Returns:
            Dictionary mapping test names to lists of Results
        """
        test_cases = custom_test_cases or self.test_cases
        results = {}

        for test_name, prompt in test_cases.items():
            print(f"\n{'=' * 50}\nTest Case: {test_name}\n{'=' * 50}")
            results[test_name] = []

            for method_name, kwargs in self.methods:
                result = self.run_attribution_test(method_name, prompt, kwargs)
                results[test_name].append(result)

        return results

    def analyze_complex_reasoning(self, prompt: str) -> List[Result]:
        """
        Run attribution analysis on a complex reasoning prompt

        Args:
            prompt: Complex reasoning prompt to analyze

        Returns:
            List of Results from each attribution method
        """
        print("\nComplex Reasoning Test")
        results = []

        for method_name, kwargs in self.methods:
            result = self.run_attribution_test(method_name, prompt, kwargs)
            results.append(result)

        return results

    def add_attribution_method(self, method_name: str, config: dict = None):
        """
        Add a new attribution method to the analyzer

        Args:
            method_name: Name of the attribution method
            config: Optional configuration dictionary for the method
        """
        config = config or {}
        self.methods.append((method_name, config))

    def add_test_case(self, name: str, prompt: str):
        """
        Add a new test case to the analyzer

        Args:
            name: Name/identifier for the test case
            prompt: The prompt to test
        """
        self.test_cases[name] = prompt
