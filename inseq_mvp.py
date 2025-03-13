import inseq
import torch
import argparse
import os
from pathlib import Path


def patch_tokenizer(model):
    """Add unk_token_id to tokenizer if it doesn't exist."""
    if model.tokenizer.unk_token_id is None:
        print("Setting default unk_token_id to 0")
        model.tokenizer.unk_token = model.tokenizer.convert_ids_to_tokens(0)
        model.tokenizer.unk_token_id = 0
    return model


def determine_device():
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Inseq compatibility, we'll actually use CPU instead of MPS
        device = 'cpu'
        print("MPS detected, but using CPU for better Inseq compatibility")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


def validate_method(method, available_methods):
    """Validate if the requested attribution method is available."""
    if method not in available_methods:
        print(f"Warning: Method '{method}' not available.")
        print(f"Available methods: {', '.join(available_methods)}")
        print(f"Falling back to method: {available_methods[0]}")
        return available_methods[0]
    return method


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inseq Feature Attribution')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model to use for attribution')
    parser.add_argument('--method', type=str, default='integrated_gradients',
                        help='Feature attribution method to use')
    parser.add_argument('--input', type=str,
                        default='Write me a short poem about programming.',
                        help='Input text for attribution')
    parser.add_argument('--n_steps', type=int, default=5,
                        help='Number of steps for methods that require it')
    parser.add_argument('--output_dir', type=str, default='./attribution_output',
                        help='Directory to save output files')
    parser.add_argument('--step_scores', nargs='+', default=['probability'],
                        help='Step scores to compute alongside attribution')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage regardless of GPU availability')
    args = parser.parse_args()

    # Determine device
    device = 'cpu' if args.force_cpu else determine_device()

    # Force PyTorch to use CPU for gradient calculation
    # This is important for Inseq which might not be fully MPS compatible
    if device == 'mps':
        print("Warning: MPS might not be fully compatible with Inseq attribution methods")
        print("Consider using --force_cpu if you encounter errors")

    # Set PyTorch default device
    # This is to make sure all computations happen on the specified device
    torch.set_default_device(device)

    # Validate attribution method
    available_methods = inseq.list_feature_attribution_methods()
    method = validate_method(args.method, available_methods)

    # Load model with attribution method
    print(f"Loading model {args.model} with attribution method {method}...")
    try:
        model = inseq.load_model(args.model, method)

        # Apply patch for unk_token_id
        model = patch_tokenizer(model)

        # Try to move model to the appropriate device
        if hasattr(model, 'to'):
            try:
                model.to(device)
                print(f"Model moved to {device}")
            except Exception as e:
                print(f"Could not move model to {device}: {e}")
                print("Continuing with default device")

        # Make sure we use CPU for autograd operations
        # Some gradient operations aren't fully implemented on MPS
        if device == 'mps':
            print("Setting default device to CPU for autograd operations")
            torch._C._set_default_device('cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set more conservative defaults for attribution parameters
    # This can help with memory and compatibility issues
    n_steps = min(args.n_steps, 100)  # Use fewer steps

    # Perform attribution
    print(f"Performing attribution on input: {args.input}")
    try:
        # First, try with minimal parameters
        print("Attempting attribution with minimal parameters...")
        attribution = model.attribute(
            args.input,
            n_steps=n_steps,
        )
    except Exception as e:
        print(f"Error during attribution: {e}")
        print("Trying with forced CPU usage...")

        # If that fails, force CPU for the model and try again
        torch.set_default_device('cpu')
        if hasattr(model, 'to'):
            model.to('cpu')
            print("Model moved to CPU for compatibility")

        try:
            attribution = model.attribute(args.input)
        except Exception as e:
            print(f"Attribution failed completely: {e}")
            print("\nSuggestions:")
            print("1. Try using a smaller model like 'gpt2' or 'facebook/opt-125m'")
            print("2. Use a simpler attribution method like 'input_x_gradient'")
            print("3. Check Inseq documentation for MPS compatibility")
            return

    # Save attribution to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'attribution.json'

    # Use Inseq's native save method for JSON
    attribution.save(str(json_path))
    print(f"Attribution saved to {json_path}")

    # Display attribution in console
    print("\nAttribution results:")
    attribution.show()


if __name__ == "__main__":
    main()