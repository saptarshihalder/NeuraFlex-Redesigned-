import torch
import argparse
from neuraflex import NeuraFlex, generate_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using NeuraFlex LLM')
    parser.add_argument('--model_path', type=str, default='neuraflex_v0.0.1_final.pth', help='Path to the model weights')
    parser.add_argument('--prompt', type=str, default='NeuraFlex is', help='Starting text for generation')
    parser.add_argument('--length', type=int, default=200, help='Number of characters to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (higher = more random)')
    args = parser.parse_args()
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = NeuraFlex(
        vocab_size=256,  # Using ASCII characters
        embed_dim=128,
        num_heads=2,
        ff_dim=256,
        num_layers=2
    )
    model.version = "0.0.1"
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate text
    print("\n" + "=" * 50)
    print(f"NeuraFlex LLM v{model.version} - Text Generation")
    print("=" * 50)
    print(f"\nGenerating {args.length} characters with prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    
    generated_text = generate_text(
        model, 
        start_text=args.prompt, 
        max_length=args.length,
        temperature=args.temperature,
        device=device
    )
    
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main() 