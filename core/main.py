from returns.result import Success, Failure

from core.attribution_comparison import AttributionConfig, AttributionAnalyzer
from core.model import ModelManager

def main():
    model_config = {
        "llm_id": "meta-llama/Llama-3.2-3B",
        "device": "auto",
        "torch_dtype": "float16"
    }

    attributionConfig = AttributionConfig(
        output_data_dir="custom/data",
        output_figures_dir="custom/figures"
    )
    complex_prompt = """
        Alice ist größer als Bob, aber kleiner als Charlie.
        David ist gleich groß wie Bob.
        Emma ist größer als Charlie.
        Wer ist die kleinste Person?
    """


    model_manager_result = ModelManager.initialize(model_config)
    match model_manager_result:
        case Success(model_manager):
            analyzer = AttributionAnalyzer(model_manager, attributionConfig)

            results = analyzer.run_all_tests()
            complex_results = analyzer.analyze_complex_reasoning(complex_prompt)

            return results, complex_results

        case Failure(error):
            raise RuntimeError(f"Failed to load model: {error}")


if __name__ == "__main__":
    main()