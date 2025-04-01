import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import logging
import time

from src.persistence.storage import is_system_token, TokenAnalysisStorage

logger = logging.getLogger(__name__)


def filter_tokens_and_matrix(input_tokens, output_tokens, association_matrix, exclude_system=False):
    """Filter tokens and matrix based on system token exclusion."""
    if not exclude_system:
        return input_tokens, output_tokens, association_matrix

    # Create input token mask
    input_mask = [not is_system_token(token) for token in input_tokens]
    # Create output token mask
    output_mask = [not is_system_token(token) for token in output_tokens]

    # Filter tokens
    filtered_input_tokens = [t for t, m in zip(input_tokens, input_mask) if m]
    filtered_output_tokens = [t for t, m in zip(output_tokens, output_mask) if m]

    # Filter matrix
    filtered_matrix = association_matrix[output_mask][:, input_mask]

    return filtered_input_tokens, filtered_output_tokens, filtered_matrix


def visualize_token_influence(analysis_result, name="", storage=TokenAnalysisStorage(base_path="output"),
                              exclude_system=False, save = True):
    """Create a heatmap visualization of token associations."""
    try:
        # Extract data - matrix is [output_tokens x input_tokens]
        input_tokens = [t.cleaned for t in analysis_result.data.input_tokens]
        output_tokens = [t.cleaned for t in analysis_result.data.output_tokens]
        association_matrix = np.array(analysis_result.data.association_matrix).astype(np.float64)

        # Filter system tokens if requested
        input_tokens, output_tokens, association_matrix = filter_tokens_and_matrix(
            input_tokens, output_tokens, association_matrix, exclude_system
        )

        # Log shapes
        logger.info(f"Matrix shape: {association_matrix.shape}")
        logger.info(f"Input tokens: {len(input_tokens)}")
        logger.info(f"Output tokens: {len(output_tokens)}")

        # Skip visualization if no tokens remain after filtering
        if len(input_tokens) == 0 or len(output_tokens) == 0:
            logger.warning("No tokens remain after filtering. Skipping visualization.")
            return None, None

        # Normalize the association matrix row-wise (per output token)
        association_matrix_norm = np.zeros_like(association_matrix)
        for i in range(association_matrix.shape[0]):
            scaler = MinMaxScaler()
            association_matrix_norm[i, :] = scaler.fit_transform(
                association_matrix[i, :].reshape(-1, 1)
            ).ravel()

        # Transpose for visualization (we want input tokens on y-axis)
        association_matrix_norm = association_matrix_norm.T

        # Create visualization with dynamic sizing
        token_ratio = len(input_tokens) / len(output_tokens)
        fig_width = min(24, max(12, len(output_tokens) * 0.4))
        fig_height = min(16, max(8, len(input_tokens) * 0.3))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create heatmap
        sns.heatmap(
            association_matrix_norm,
            ax=ax,
            cmap="viridis",
            cbar_kws={'label': 'Normalized Association'},
            annot=False,
            fmt='.2f'
        )

        # Configure axes
        ax.set_xticks(np.arange(len(output_tokens)) + 0.5)
        ax.set_xticklabels(output_tokens, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(input_tokens)) + 0.5)
        ax.set_yticklabels(input_tokens, rotation=0, ha='right')

        # Set labels and title
        title = "Token Association Heatmap"
        if exclude_system:
            title += " (Excluding System Tokens)"
        plt.title(title, fontsize=14, fontweight='bold')
        if name:
            plt.suptitle(name, fontsize=10)
        ax.set_xlabel("Generated Tokens", fontsize=12)
        ax.set_ylabel("Input Tokens", fontsize=12)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)

        # Save figure
        if save:
            sys_suffix = "_no_system" if exclude_system else ""
            save_name = f'{time.strftime("%Y%m%d-%H%M%S")}_token_association_{name.replace(" ", "_")}{sys_suffix}.png'
            storage.save_graph(plt, save_name)
        return fig, ax

    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        logger.error(f"Input tokens: {len(input_tokens) if 'input_tokens' in locals() else 'not created'}")
        logger.error(f"Output tokens: {len(output_tokens) if 'output_tokens' in locals() else 'not created'}")
        logger.error(
            f"Association matrix shape: {association_matrix.shape if 'association_matrix' in locals() else 'not created'}")
        raise


def visualize_summed_token_association(analysis_result, name="", storage=TokenAnalysisStorage(base_path="output"),
                                       show_boxplots=False, exclude_system=False, save = True):
    """Create a horizontal bar plot showing summed token associations."""
    try:
        # Extract data - matrix is [output_tokens x input_tokens]
        input_tokens = [t.cleaned for t in analysis_result.data.input_tokens]
        output_tokens = [t.cleaned for t in analysis_result.data.output_tokens]
        association_matrix = np.array(analysis_result.data.association_matrix).astype(np.float64)

        # Filter system tokens if requested
        input_tokens, output_tokens, association_matrix = filter_tokens_and_matrix(
            input_tokens, output_tokens, association_matrix, exclude_system
        )

        # Skip visualization if no tokens remain after filtering
        if len(input_tokens) == 0 or len(output_tokens) == 0:
            logger.warning("No tokens remain after filtering. Skipping visualization.")
            return None, None

        # Log shapes
        logger.info(f"Matrix shape: {association_matrix.shape}")
        logger.info(f"Input tokens: {len(input_tokens)}")
        logger.info(f"Output tokens: {len(output_tokens)}")

        # Transpose to get [input_tokens x output_tokens]
        association_matrix = association_matrix.T

        # Sum association across all output tokens
        summed_association = np.sum(association_matrix, axis=1)

        # Normalize the summed association
        scaler = MinMaxScaler()
        normalized_association = scaler.fit_transform(summed_association.reshape(-1, 1)).ravel()

        # Generate y positions
        y_pos = np.arange(len(input_tokens))

        # Create visualization with dynamic sizing
        height_per_token = 0.4
        fig_height = max(8, height_per_token * len(input_tokens))
        fig_width = 12 if show_boxplots else 10

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if show_boxplots:
            # For boxplot, we need the distributions across output tokens
            sns.boxplot(
                data=association_matrix,
                orient='h',
                ax=ax,
                color='lightgray',
                width=0.6
            )

            bars = ax.barh(
                y_pos,
                normalized_association,
                alpha=0.8,
                color='blue',
                height=0.8
            )

            # Set appropriate x limits
            ax.set_xlim(min(0, association_matrix.min()), max(1, association_matrix.max()))

            # Add second x-axis for normalized values
            ax2 = ax.twiny()
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Normalized Summed Importance", fontsize=12)
            ax.set_xlabel("Raw Association", fontsize=12)
        else:
            # Just plot the normalized bars
            bars = ax.barh(
                y_pos,
                normalized_association,
                color='blue',
                height=0.8
            )
            ax.set_xlabel("Normalized Summed Importance", fontsize=12)

        # Configure axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(input_tokens, fontsize=10)

        # Set labels and title
        title = "Overall Input Token Association"
        if exclude_system:
            title += " (Excluding System Tokens)"
        plt.title(title, fontsize=14, fontweight='bold')
        if name:
            plt.suptitle(name, fontsize=10)
        ax.set_ylabel("Input Tokens", fontsize=12)

        # Add value labels
        for bar, value in zip(bars, normalized_association):
            ax.text(
                max(value + 0.01, 0.01),
                bar.get_y() + bar.get_height() / 2,
                f'{value:.3f}',
                va='center',
                fontsize=9
            )

        # Adjust layout
        plt.tight_layout()

        # Save figure
        if save:
            sys_suffix = "_no_system" if exclude_system else ""
            save_name = f'{time.strftime("%Y%m%d-%H%M%S")}_summed_token_association_{name.replace(" ", "_")}{sys_suffix}.png'
            storage.save_graph(plt, save_name)
        return fig, ax

    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        logger.error(f"Input tokens: {len(input_tokens) if 'input_tokens' in locals() else 'not created'}")
        logger.error(f"Output tokens: {len(output_tokens) if 'output_tokens' in locals() else 'not created'}")
        logger.error(
            f"Association matrix shape: {association_matrix.shape if 'association_matrix' in locals() else 'not created'}")
        raise
