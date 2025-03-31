from work.llama_token_analyzer.persistence.storage import TokenAnalysisStorage
from work.llama_token_analyzer.visualization.plots import visualize_token_influence, visualize_summed_token_association


def visualize(result, storage: TokenAnalysisStorage, exclude_system=False, save=True):
    prompt_name = result.metadata.prompt[:50]
    return (
    visualize_token_influence(result, name=prompt_name, storage=storage, exclude_system=exclude_system, save=save),
    visualize_summed_token_association(result, name=prompt_name, storage=storage, show_boxplots=True,
                                       exclude_system=exclude_system, save=save)
    )
