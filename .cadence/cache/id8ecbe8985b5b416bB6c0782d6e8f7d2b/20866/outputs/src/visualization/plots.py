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
    input_mask = np.array([not is_system_token(token) for token in input_tokens])
    # Create output token mask
    output_mask = np.array([not is_system_token(token) for token in output_tokens])

    # Filter tokens
    filtered_input_tokens = [t for t, m in zip(input_tokens, input_mask) if m]
    filtered_output_tokens = [t for t, m in zip(output_tokens, output_mask) if m]

    # Handle empty masks
    if len(filtered_input_tokens) == 0 or len(filtered_output_tokens) == 0:
        logger.warning("All tokens were filtered out. Creating empty matrix.")
        return filtered_input_tokens, filtered_output_tokens, np.zeros((len(filtered_output_tokens), len(filtered_input_tokens)))

    try:
        # Verify matrix shape before filtering
        if association_matrix.shape[0] != len(output_tokens) or association_matrix.shape[1] != len(input_tokens):
            logger.warning(f"Matrix shape mismatch before filtering: {association_matrix.shape} vs expected {len(output_tokens)}x{len(input_tokens)}")
            
            # Resize the matrix to match the expected shape
            resized_matrix = np.zeros((len(output_tokens), len(input_tokens)))
            rows_to_copy = min(association_matrix.shape[0], len(output_tokens))
            cols_to_copy = min(association_matrix.shape[1], len(input_tokens))
            resized_matrix[:rows_to_copy, :cols_to_copy] = association_matrix[:rows_to_copy, :cols_to_copy]
            association_matrix = resized_matrix
        
        # Filter matrix - we need boolean masks for numpy indexing
        filtered_matrix = association_matrix[output_mask, :][:, input_mask]
        
        # Verify the shape is correct after filtering
        if filtered_matrix.shape != (len(filtered_output_tokens), len(filtered_input_tokens)):
            logger.warning(f"Matrix shape mismatch after filtering: {filtered_matrix.shape} vs expected {len(filtered_output_tokens)}x{len(filtered_input_tokens)}")
            
            # Create a new matrix with correct dimensions
            correct_matrix = np.zeros((len(filtered_output_tokens), len(filtered_input_tokens)))
            rows_to_copy = min(filtered_matrix.shape[0], len(filtered_output_tokens))
            cols_to_copy = min(filtered_matrix.shape[1], len(filtered_input_tokens))
            correct_matrix[:rows_to_copy, :cols_to_copy] = filtered_matrix[:rows_to_copy, :cols_to_copy]
            filtered_matrix = correct_matrix
            
        return filtered_input_tokens, filtered_output_tokens, filtered_matrix
    except Exception as e:
        logger.error(f"Error filtering matrix: {str(e)}")
        # Return a matrix of zeros with the correct shape
        return filtered_input_tokens, filtered_output_tokens, np.zeros((len(filtered_output_tokens), len(filtered_input_tokens)))


def visualize_token_influence(analysis_result, name="", storage=TokenAnalysisStorage(base_path="output"),
                              exclude_system=False, save = True):
    """Create a heatmap visualization of token associations with causal structure."""
    try:
        # Extract data from analysis result
        input_tokens = [t.cleaned for t in analysis_result.data.input_tokens]
        output_tokens = [t.cleaned for t in analysis_result.data.output_tokens]
        association_matrix = np.array(analysis_result.data.association_matrix).astype(np.float64)
        
        # Log initial data
        logger.info(f"Original matrix shape: {association_matrix.shape}")
        logger.info(f"Input tokens: {len(input_tokens)}")
        logger.info(f"Output tokens: {len(output_tokens)}")
        
        # Check matrix shape vs token counts and ensure it's in [output_tokens x input_tokens] format
        if association_matrix.shape[0] == len(input_tokens) and association_matrix.shape[1] == len(output_tokens):
            logger.info("Matrix has shape [input_tokens x output_tokens], transposing to [output_tokens x input_tokens]")
            association_matrix = association_matrix.T
        elif association_matrix.shape[0] != len(output_tokens) or association_matrix.shape[1] != len(input_tokens):
            logger.warning(f"Matrix shape {association_matrix.shape} doesn't match token counts: output={len(output_tokens)}, input={len(input_tokens)}")
            # Create a correctly sized matrix filled with zeros
            correct_matrix = np.zeros((len(output_tokens), len(input_tokens)))
            # Try to copy as much data as possible
            min_rows = min(association_matrix.shape[0], len(output_tokens))
            min_cols = min(association_matrix.shape[1], len(input_tokens))
            correct_matrix[:min_rows, :min_cols] = association_matrix[:min_rows, :min_cols]
            association_matrix = correct_matrix

        # Filter system tokens if requested
        input_tokens, output_tokens, association_matrix = filter_tokens_and_matrix(
            input_tokens, output_tokens, association_matrix, exclude_system
        )

        # Skip visualization if no tokens remain after filtering
        if len(input_tokens) == 0 or len(output_tokens) == 0:
            logger.warning("No tokens remain after filtering. Skipping visualization.")
            return None, None

        # Normalize the association matrix row-wise (per output token)
        association_matrix_norm = np.zeros_like(association_matrix)
        for i in range(association_matrix.shape[0]):
            row_data = association_matrix[i, :]
            if np.any(row_data):  # Only normalize if there are non-zero values
                scaler = MinMaxScaler()
                association_matrix_norm[i, :] = scaler.fit_transform(
                    row_data.reshape(-1, 1)
                ).ravel()
        
        # ===== NEW CAUSAL VISUALIZATION APPROACH =====
        # For causal structure, we need:
        # - Y-axis: All tokens (input + output up to n-1)
        # - X-axis: Output tokens
        # - Only show associations where y-token comes before x-token
        
        # Create the combined token list for y-axis (inputs + outputs except last)
        # We exclude the last output token since it can't influence any future tokens
        all_tokens = input_tokens + output_tokens[:-1]
        
        # Calculate the total token count
        n_input = len(input_tokens)
        n_output = len(output_tokens)
        n_total = len(all_tokens)
        
        # Create a new causal attribution matrix [all_tokens x output_tokens]
        # Initially fill with zeros
        causal_matrix = np.zeros((n_total, n_output))
        
        # Fill the causal matrix enforcing the causality constraint:
        # - Top section: input tokens influence all outputs (copy from association_matrix)
        try:
            # Make sure we don't overindex
            rows_to_copy = min(n_input, association_matrix_norm.shape[1])
            cols_to_copy = min(n_output, association_matrix_norm.shape[0])
            
            # Transpose and copy what we can
            if rows_to_copy > 0 and cols_to_copy > 0:
                transposed = association_matrix_norm[:cols_to_copy, :rows_to_copy].T
                causal_matrix[:rows_to_copy, :cols_to_copy] = transposed
                logger.info(f"Copied matrix section: {rows_to_copy}x{cols_to_copy}")
            else:
                logger.warning(f"Cannot copy top section: input={n_input}, output={n_output}, matrix={association_matrix_norm.shape}")
        except Exception as e:
            logger.warning(f"Error copying top section: {e}")
        
        # - Bottom section: output tokens only influence future tokens
        try:
            max_output_idx = min(n_output-1, association_matrix_norm.shape[1])
            for i in range(max_output_idx):  # For each output token (except last)
                # Index in the causal matrix (after the input tokens)
                y_idx = n_input + i
                
                # This token can only influence tokens that come after it
                # For safety, ensure all indices are within bounds
                if y_idx < n_total and (i+1) < n_output and i < association_matrix_norm.shape[1]:
                    # Calculate max column we can safely use
                    max_col = min(n_output, association_matrix_norm.shape[0])
                    
                    # Only copy if we have data to copy and somewhere to put it
                    if max_col > (i+1):
                        try:
                            # Get values from [i+1 to max_col] of row i in association_matrix_norm
                            values_to_copy = association_matrix_norm[(i+1):max_col, i]
                            # Copy to row y_idx, columns [i+1 to max_col] in causal_matrix
                            causal_matrix[y_idx, (i+1):max_col] = values_to_copy
                        except Exception as e2:
                            logger.warning(f"Error copying for token {i}: {e2}")
        except Exception as e:
            logger.warning(f"Error filling bottom section: {e}")
        
        # Create visualization with dynamic sizing
        token_ratio = n_total / n_output
        fig_width = min(24, max(12, n_output * 0.4))
        fig_height = min(24, max(10, n_total * 0.3))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create the causal heatmap
        sns.heatmap(
            causal_matrix,
            ax=ax,
            cmap="viridis",
            cbar_kws={'label': 'Normalized Attribution'},
            annot=False,
            fmt='.2f',
            mask=(causal_matrix == 0)  # Mask zero values to visualize causality constraint
        )
        
        # Configure axes
        ax.set_xticks(np.arange(n_output) + 0.5)
        ax.set_xticklabels(output_tokens, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_total) + 0.5)
        ax.set_yticklabels(all_tokens, rotation=0, ha='right')
        
        # Add a separator line between input and output tokens on y-axis
        if n_input > 0 and n_output > 1:
            ax.axhline(y=n_input, color='r', linestyle='-', linewidth=2)
            
        # Set labels and title
        title = "Causal Token Attribution Map"
        if exclude_system:
            title += " (Excluding System Tokens)"
        plt.title(title, fontsize=14, fontweight='bold')
        if name:
            plt.suptitle(name, fontsize=10)
        ax.set_xlabel("Generated Tokens", fontsize=12)
        ax.set_ylabel("Context Tokens (Input + Generated)", fontsize=12)
        
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
        # Extract data - matrix should be [output_tokens x input_tokens]
        input_tokens = [t.cleaned for t in analysis_result.data.input_tokens]
        output_tokens = [t.cleaned for t in analysis_result.data.output_tokens]
        association_matrix = np.array(analysis_result.data.association_matrix).astype(np.float64)
        
        # Check matrix shape vs token counts and transpose if needed
        if association_matrix.shape[0] == len(input_tokens) and association_matrix.shape[1] == len(output_tokens):
            logger.info("Matrix has shape [input_tokens x output_tokens], transposing to [output_tokens x input_tokens]")
            association_matrix = association_matrix.T
        elif association_matrix.shape[0] != len(output_tokens) or association_matrix.shape[1] != len(input_tokens):
            logger.warning(f"Matrix shape {association_matrix.shape} doesn't match token counts: output={len(output_tokens)}, input={len(input_tokens)}")
            # Create a correctly sized matrix filled with zeros
            correct_matrix = np.zeros((len(output_tokens), len(input_tokens)))
            # Try to copy as much data as possible
            min_rows = min(association_matrix.shape[0], len(output_tokens))
            min_cols = min(association_matrix.shape[1], len(input_tokens))
            correct_matrix[:min_rows, :min_cols] = association_matrix[:min_rows, :min_cols]
            association_matrix = correct_matrix

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

        # ===== CAUSAL APPROACH FOR SUMMED ASSOCIATION =====
        # We need to consider both input tokens and generated tokens (except last)
        
        # Create the combined token list 
        all_tokens = input_tokens + output_tokens[:-1]
        
        # Calculate counts
        n_input = len(input_tokens)
        n_output = len(output_tokens)
        n_total = len(all_tokens)
        
        # Create and fill causal influence matrix [all_tokens, output_tokens]
        causal_matrix = np.zeros((n_total, n_output))
        
        # Ensure matrix is in [output_tokens x input_tokens] format
        if association_matrix.shape[0] == len(input_tokens) and association_matrix.shape[1] == len(output_tokens):
            association_matrix = association_matrix.T
            
        # Normalize the matrix for better comparison
        association_matrix_norm = np.zeros_like(association_matrix)
        for i in range(association_matrix.shape[0]):
            row_data = association_matrix[i, :]
            if np.any(row_data):  # Only normalize if there are non-zero values
                scaler = MinMaxScaler()
                association_matrix_norm[i, :] = scaler.fit_transform(
                    row_data.reshape(-1, 1)
                ).ravel()
        
        # Fill top section with safe bounds checking
        try:
            # Make sure we don't overindex
            rows_to_copy = min(n_input, association_matrix_norm.shape[1])
            cols_to_copy = min(n_output, association_matrix_norm.shape[0])
            
            # Transpose and copy what we can
            if rows_to_copy > 0 and cols_to_copy > 0:
                transposed = association_matrix_norm[:cols_to_copy, :rows_to_copy].T
                causal_matrix[:rows_to_copy, :cols_to_copy] = transposed
                logger.info(f"Copied matrix section: {rows_to_copy}x{cols_to_copy}")
            else:
                logger.warning(f"Cannot copy top section: input={n_input}, output={n_output}, matrix={association_matrix_norm.shape}")
        except Exception as e:
            logger.warning(f"Error copying top section: {e}")
        
        # Fill bottom section with safe bounds checking
        try:
            max_output_idx = min(n_output-1, association_matrix_norm.shape[1])
            for i in range(max_output_idx):  # For each output token (except last)
                # Index in the causal matrix (after the input tokens)
                y_idx = n_input + i
                
                # This token can only influence tokens that come after it
                # For safety, ensure all indices are within bounds
                if y_idx < n_total and (i+1) < n_output and i < association_matrix_norm.shape[1]:
                    # Calculate max column we can safely use
                    max_col = min(n_output, association_matrix_norm.shape[0])
                    
                    # Only copy if we have data to copy and somewhere to put it
                    if max_col > (i+1):
                        try:
                            # Get values from [i+1 to max_col] of row i in association_matrix_norm
                            values_to_copy = association_matrix_norm[(i+1):max_col, i]
                            # Copy to row y_idx, columns [i+1 to max_col] in causal_matrix
                            causal_matrix[y_idx, (i+1):max_col] = values_to_copy
                        except Exception as e2:
                            logger.warning(f"Error copying for token {i}: {e2}")
        except Exception as e:
            logger.warning(f"Error filling bottom section: {e}")
        
        # Sum influence across all influenced tokens (horizontally)
        summed_association = np.sum(causal_matrix, axis=1)
        
        # Normalize the summed influence
        if np.any(summed_association):
            scaler = MinMaxScaler()
            normalized_association = scaler.fit_transform(summed_association.reshape(-1, 1)).ravel()
        else:
            normalized_association = np.zeros_like(summed_association)

        # Generate y positions for all tokens (inputs + outputs except last)
        y_pos = np.arange(n_total)

        # Create visualization with dynamic sizing
        height_per_token = 0.4
        fig_height = max(8, height_per_token * n_total)
        fig_width = 12 if show_boxplots else 10

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if show_boxplots:
            # For boxplot, we need the distributions across output tokens
            sns.boxplot(
                data=causal_matrix,
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
            ax.set_xlim(min(0, causal_matrix.min()), max(1, causal_matrix.max()))

            # Add second x-axis for normalized values
            ax2 = ax.twiny()
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Normalized Causal Influence", fontsize=12)
            ax.set_xlabel("Raw Influence Distribution", fontsize=12)
        else:
            # Just plot the normalized bars
            bars = ax.barh(
                y_pos,
                normalized_association,
                color='blue',
                height=0.8
            )
            ax.set_xlabel("Normalized Causal Influence", fontsize=12)

        # Configure axes - show all tokens in the y-axis
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_tokens, fontsize=10)
        
        # Add a separator line between input and output tokens
        if n_input > 0 and n_output > 1:
            ax.axhline(y=n_input-0.5, color='r', linestyle='-', linewidth=2)

        # Set labels and title
        title = "Overall Token Causal Influence"
        if exclude_system:
            title += " (Excluding System Tokens)"
        plt.title(title, fontsize=14, fontweight='bold')
        if name:
            plt.suptitle(name, fontsize=10)
        ax.set_ylabel("Context Tokens (Input + Generated)", fontsize=12)

        # Add value labels for non-zero values
        for bar, value in zip(bars, normalized_association):
            if value > 0.01:  # Only label non-trivial values
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
