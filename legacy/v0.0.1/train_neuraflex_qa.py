import argparse
import torch
from squad_dataset import load_squad_dataset, create_squad_dataloader
from neuraflex_qa import NeuraFlexQA, train_qa_model, answer_question
import random
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuraFlex QA model on SQuAD dataset")
    parser.add_argument("--subset_size", type=int, default=500, 
                        help="Number of SQuAD examples to use (default: 500)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model name (default: distilbert-base-uncased)")
    parser.add_argument("--save_dir", type=str, default="./",
                        help="Directory to save model (default: ./)")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after training")
    parser.add_argument("--test_samples", type=int, default=5,
                        help="Number of test samples to evaluate (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(f"Training with the following settings:")
    print(f"- Device: {device}")
    print(f"- SQuAD subset size: {args.subset_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Base model: {args.model_name}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    start_time = time.time()
    train_dataset, raw_data, tokenizer = load_squad_dataset(
        subset_size=args.subset_size,
        tokenizer_name=args.model_name
    )
    train_loader = create_squad_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Initialize model
    print("\nInitializing model...")
    model = NeuraFlexQA(base_model_name=args.model_name)
    
    # Train model
    print("\nTraining model...")
    train_start = time.time()
    model = train_qa_model(
        model=model,
        train_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save final model
    model_path = os.path.join(args.save_dir, "neuraflex_qa_final.pth")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Test model if requested
    if args.test:
        print("\nTesting model on sample questions...")
        test_samples = min(args.test_samples, len(raw_data))
        test_indices = random.sample(range(len(raw_data)), test_samples)
        
        correct = 0
        for idx in test_indices:
            sample = raw_data[idx]
            question = sample["question"]
            context = sample["context"]
            true_answer = sample["answers"]["text"][0]
            
            print(f"\nQuestion: {question}")
            print(f"Context: {context[:150]}...")
            print(f"True answer: {true_answer}")
            
            # Get model prediction
            pred_answer = answer_question(model, question, context, tokenizer, device)
            print(f"Predicted answer: {pred_answer}")
            
            # Simple exact match evaluation
            if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower():
                correct += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
        
        print(f"\nAccuracy on {test_samples} test samples: {correct/test_samples:.2f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 