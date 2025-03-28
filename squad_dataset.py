import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import os
import numpy as np

class SQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=384):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example["question"]
        context = example["context"]
        
        # Find answer positions
        answer_text = example["answers"]["text"][0]
        answer_start = example["answers"]["answer_start"][0]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            question, 
            context,
            max_length=self.max_length,
            truncation=True,
            stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Map token positions to original text positions
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        # Find start and end positions in tokenized space
        start_position = end_position = 0
        for i, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_position = i
            if start < answer_start + len(answer_text) <= end:
                end_position = i
                break
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "start_position": torch.tensor(start_position),
            "end_position": torch.tensor(end_position)
        }

def load_squad_dataset(subset_size=500, tokenizer_name="distilbert-base-uncased"):
    """
    Loads a subset of the SQuAD dataset
    
    Args:
        subset_size: Number of examples to load (default: 500)
        tokenizer_name: Name of the pretrained model for tokenization
        
    Returns:
        train_dataset: PyTorch dataset ready for training
        raw_data: The raw dataset samples
        tokenizer: The tokenizer object
    """
    # Load dataset from Hugging Face
    print(f"Loading SQuAD dataset (first {subset_size} examples)...")
    squad_dataset = load_dataset("squad", split=f"train[:{subset_size}]")
    
    # Save a copy of the raw data for reference
    raw_data_path = "squad_data.json"
    with open(raw_data_path, "w") as f:
        json.dump([{"question": item["question"], 
                   "context": item["context"], 
                   "answers": item["answers"]} 
                  for item in squad_dataset], f, indent=2)
    print(f"Raw data saved to {raw_data_path}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create PyTorch dataset
    train_dataset = SQuADDataset(squad_dataset, tokenizer)
    
    return train_dataset, squad_dataset, tokenizer

def create_squad_dataloader(dataset, batch_size=8, shuffle=True):
    """Create a DataLoader for SQuAD dataset training"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_sample_question_answer_pair(raw_data):
    """Returns a random sample Q&A pair from the data"""
    sample = raw_data[np.random.randint(0, len(raw_data))]
    return {
        "question": sample["question"],
        "context": sample["context"],
        "answer": sample["answers"]["text"][0]
    }

if __name__ == "__main__":
    # Example usage 
    train_dataset, raw_data, tokenizer = load_squad_dataset(subset_size=500)
    train_loader = create_squad_dataloader(train_dataset)
    
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Display a sample
    sample = get_sample_question_answer_pair(raw_data)
    print("\nSample Question-Answer Pair:")
    print(f"Question: {sample['question']}")
    print(f"Context: {sample['context'][:100]}...")
    print(f"Answer: {sample['answer']}") 