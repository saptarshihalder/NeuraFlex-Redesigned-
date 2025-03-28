import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering
import os
import json

class NeuraFlexQA(nn.Module):
    """
    NeuraFlex model for Question Answering using SQuAD dataset
    This extends the base NeuraFlex by adding specialized QA capabilities
    """
    def __init__(self, base_model_name="distilbert-base-uncased"):
        super().__init__()
        
        # Load pretrained model or initialize from scratch
        self.model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
        
        # Add metadata for versioning
        self.version = "0.1.0-qa"
        self.description = "NeuraFlex with SQuAD-based QA capabilities"
        
        # Knowledge retention system - store examples for few-shot learning
        self.knowledge_base = []
        self.knowledge_file = "neuraflex_qa_knowledge.json"
        self.load_knowledge()
        
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        """Forward pass for training and inference"""
        if start_positions is not None and end_positions is not None:
            # Training mode
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            return outputs.loss, outputs.start_logits, outputs.end_logits
        else:
            # Inference mode
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs.start_logits, outputs.end_logits
    
    def save(self, path="neuraflex_qa_model.pth"):
        """Save the model weights"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        
        # Also save metadata
        metadata = {
            "version": self.version,
            "description": self.description,
            "base_model": self.model.config._name_or_path
        }
        with open(path.replace(".pth", "_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path="neuraflex_qa_model.pth"):
        """Load the model weights"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"Model file {path} not found")
            return False
    
    def add_to_knowledge(self, question, context, answer):
        """Add a QA pair to the knowledge base for few-shot learning"""
        self.knowledge_base.append({
            "question": question,
            "context": context,
            "answer": answer
        })
        # Limit knowledge base size to prevent memory issues
        if len(self.knowledge_base) > 1000:
            self.knowledge_base = self.knowledge_base[-1000:]
        self.save_knowledge()
        
    def save_knowledge(self):
        """Save the knowledge base to disk"""
        with open(self.knowledge_file, "w") as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def load_knowledge(self):
        """Load the knowledge base from disk"""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, "r") as f:
                    self.knowledge_base = json.load(f)
                print(f"Loaded {len(self.knowledge_base)} QA pairs into knowledge base")
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
                self.knowledge_base = []
        else:
            self.knowledge_base = []
    
    def retrieve_similar_questions(self, question, top_k=3):
        """Simple retrieval of similar questions for reference (basic implementation)"""
        # In a real implementation, this would use proper embeddings and similarity search
        # This is a simplified version just for demonstration
        matches = []
        question_words = set(question.lower().split())
        
        for qa_pair in self.knowledge_base:
            known_q = qa_pair["question"]
            known_words = set(known_q.lower().split())
            # Simple overlap metric
            overlap = len(question_words.intersection(known_words)) / len(question_words.union(known_words)) if known_words else 0
            matches.append((overlap, qa_pair))
        
        # Return top k matches
        matches.sort(reverse=True, key=lambda x: x[0])
        return [m[1] for m in matches[:top_k]]

def train_qa_model(model, train_loader, num_epochs=3, learning_rate=5e-5, device="cpu"):
    """Train the QA model on SQuAD dataset"""
    # Move model to device
    model = model.to(device)
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_starts = 0
        correct_ends = 0
        total_examples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_position"].to(device)
            end_positions = batch["end_position"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss, start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            total_examples += input_ids.size(0)
            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)
            correct_starts += (start_pred == start_positions).sum().item()
            correct_ends += (end_pred == end_positions).sum().item()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Start Acc: {correct_starts/total_examples:.4f}, "
                      f"End Acc: {correct_ends/total_examples:.4f}")
        
        # Epoch summary
        print(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Start Acc: {correct_starts/total_examples:.4f}, "
              f"End Acc: {correct_ends/total_examples:.4f}")
        
        # Save checkpoint
        model.save(f"neuraflex_qa_checkpoint_e{epoch+1}.pth")
    
    # Save final model
    model.save()
    return model

def answer_question(model, question, context, tokenizer, device="cpu"):
    """Use the trained model to answer a question based on the provided context"""
    model.eval()
    model.to(device)
    
    # Check if we have similar questions in our knowledge base
    similar_questions = model.retrieve_similar_questions(question)
    
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    # Extract offset mapping to align tokens with original text
    offset_mapping = inputs.pop("offset_mapping")
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get predictions
    with torch.no_grad():
        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Get top-5 predictions for both start and end
    start_indices = torch.topk(start_logits[0], 5).indices
    end_indices = torch.topk(end_logits[0], 5).indices
    
    # Try all combinations of starts and ends to find the best answer
    best_answer = ""
    max_score = float('-inf')
    
    for start_idx in start_indices:
        for end_idx in end_indices:
            # Skip invalid spans
            if end_idx < start_idx or end_idx - start_idx > 25:  # Max answer length of 25 tokens
                continue
                
            # Calculate score for this span
            score = start_logits[0][start_idx] + end_logits[0][end_idx]
            
            if score > max_score:
                # Get the answer text for this span
                # Map from token positions to character positions in original text
                token_start_idx = start_idx.item()
                token_end_idx = end_idx.item()
                
                # Skip special tokens like [CLS], [SEP], etc.
                if token_start_idx <= 0 or token_end_idx >= input_ids.size(1) - 1:
                    continue
                
                # Convert token indices to original text offsets
                char_start_pos = offset_mapping[0][token_start_idx][0].item()
                char_end_pos = offset_mapping[0][token_end_idx][1].item()
                
                # Extract answer from context (if available)
                if char_start_pos < len(context) and char_end_pos <= len(context):
                    candidate_answer = context[char_start_pos:char_end_pos]
                    
                    # Update if this is a better answer
                    if candidate_answer and len(candidate_answer) > 0:
                        best_answer = candidate_answer
                        max_score = score
    
    # Fallback to token-based extraction if character-based method fails
    if not best_answer:
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        
        # Ensure end comes after start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Convert token indices to actual text
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        best_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
        
        # Clean up answer
        best_answer = best_answer.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
    
    # Extract from context for better results
    # If answer is too short or looks like a repeated question fragment, use keyword extraction
    if len(best_answer) <= 2 or best_answer.lower() in question.lower():
        # Look for keywords in question to find relevant parts in context
        keywords = [word.lower() for word in question.split() 
                   if word.lower() not in ('what', 'where', 'when', 'who', 'how', 'why', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'on', 'at')]
        
        # Find the most relevant sentence in context containing keywords
        sentences = context.split('.')
        best_match = None
        best_match_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count keyword matches
            match_score = sum(1 for keyword in keywords if keyword in sentence.lower())
            if match_score > best_match_score:
                best_match_score = match_score
                best_match = sentence
        
        # Use the best matching sentence if we found one
        if best_match and best_match_score > 0:
            best_answer = best_match + '.'
    
    # Add to knowledge base for future reference
    model.add_to_knowledge(question, context, best_answer)
    
    return best_answer

if __name__ == "__main__":
    # Example usage
    from squad_dataset import load_squad_dataset, create_squad_dataloader
    import torch
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, raw_data, tokenizer = load_squad_dataset(subset_size=500)
    train_loader = create_squad_dataloader(train_dataset, batch_size=8)
    
    # Initialize model
    model = NeuraFlexQA()
    
    # Train model
    model = train_qa_model(model, train_loader, num_epochs=3, device=device)
    
    # Test with a sample
    sample = raw_data[0]
    question = sample["question"]
    context = sample["context"]
    
    print("\nTesting QA model:")
    print(f"Question: {question}")
    print(f"Context: {context[:100]}...")
    
    answer = answer_question(model, question, context, tokenizer, device)
    print(f"Predicted Answer: {answer}")
    print(f"Actual Answer: {sample['answers']['text'][0]}") 