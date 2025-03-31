from work.llama_token_analyzer.persistence.storage import TokenAnalysisStorage
from work.llama_token_analyzer.visualization.plots import visualize_token_influence, visualize_summed_token_association
from returns.result import Result, Success, Failure
import logging

logger = logging.getLogger(__name__)

def visualize(result, storage: TokenAnalysisStorage, exclude_system=False, save=True):
    """
    Visualize token attribution results.
    
    Args:
        result: Analysis result to visualize
        storage: Storage instance for saving visualizations
        exclude_system: Whether to exclude system tokens
        save: Whether to save the visualizations
        
    Returns:
        Tuple of visualization results
    """
    # Get the prompt name for visualization
    if hasattr(result, 'metadata') and hasattr(result.metadata, 'prompt'):
        prompt_name = result.metadata.prompt[:50]
    else:
        # Use a default name if metadata is not available
        prompt_name = "attribution_result"
        
    return (
        visualize_token_influence(result, name=prompt_name, storage=storage, exclude_system=exclude_system, save=save),
        visualize_summed_token_association(result, name=prompt_name, storage=storage, show_boxplots=True,
                                          exclude_system=exclude_system, save=save)
    )
