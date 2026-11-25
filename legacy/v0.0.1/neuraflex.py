import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
import re
import random
import decimal

# Set the precision context globally for high-precision math
decimal.getcontext().prec = 28  # Higher than needed to avoid rounding errors

# NeuraFlex LLM - A tiny language model implementation
# Version: 0.0.1

# Knowledge base for NeuraFlex
NEURAFLEX_KNOWLEDGE = {
    "name": "NeuraFlex",
    "version": "0.0.1",
    "creator": "Saptarshi Halder",
    "purpose": "A tiny language model implementation designed to run on standard laptops",
    "parameters": "~100,000 parameters",
    "architecture": "Mini-Transformer with 2 layers, 128-dim embeddings, 2 attention heads, and 256-dim feedforward network",
    "infrastructure": "Lightweight transformer architecture running on CPU, with character-level tokenization",
    "capabilities": "Basic text generation, simple Q&A, educational demonstrations, precise mathematical calculations",
    "limitations": "Small context window (64 chars), basic understanding, limited knowledge base"
}

# Question patterns for consistent answers
QUESTION_PATTERNS = {
    "creator": [
        r"who (created|made|built|developed|authored) (you|neuraflex)",
        r"(creator|author|developer|maker)('s| of|) (name|is|)",
        r"who is (behind|responsible for) (you|neuraflex)",
        r"who (owns|maintains) (you|neuraflex)",
        r"(tell|state|give) (me |)(your |the |)creator"
    ],
    "identity": [
        r"what('s| is) your (name|identity)",
        r"who are you",
        r"(what|which) (model|llm|ai) are you",
        r"(tell|state|give) (me |)your name",
        r"introduce yourself"
    ],
    "version": [
        r"(what|which) version",
        r"(version|release) number",
        r"(what|which) (version|release) (are you|of neuraflex)"
    ],
    "purpose": [
        r"(what|why) (are you|were you) (made|created|built|developed) for",
        r"what('s| is) your (purpose|function|goal)",
        r"why (do|does) (you|neuraflex) exist"
    ],
    "architecture": [
        r"(what|how) (infrastructure|architecture)",
        r"how (are you|is neuraflex) (built|designed|structured|made)",
        r"(what|which) (model|network) (structure|design)",
        r"technical (specs|specifications|details)",
        r"how (do you|does neuraflex) work"
    ],
    "parameters": [
        r"how (many|much) parameters",
        r"(parameter|model) (count|size|scale)",
        r"how (big|large|small) (are you|is neuraflex)"
    ]
}

# Math pattern for detecting calculation requests
MATH_PATTERN = re.compile(
    r'(-?\d+\.?\d*)\s*([+\-*/×÷])\s*(-?\d+\.?\d*)'
)

class HighPrecisionMath:
    """Handle high-precision math operations up to 12 decimal places"""
    
    def __init__(self):
        # Map operators to decimal module functions
        self.op_map = {
            '+': decimal.Decimal.__add__,
            '-': decimal.Decimal.__sub__,
            '*': decimal.Decimal.__mul__,
            '×': decimal.Decimal.__mul__,
            '/': decimal.Decimal.__truediv__,
            '÷': decimal.Decimal.__truediv__
        }
    
    def calculate(self, num1, operator, num2):
        """Perform high-precision calculation"""
        try:
            # Convert strings to Decimal objects for precision
            d1 = decimal.Decimal(num1)
            d2 = decimal.Decimal(num2)
            
            # Get the operation function and apply it
            op_func = self.op_map.get(operator)
            if not op_func:
                return None
                
            # Perform the calculation
            result = op_func(d1, d2)
            
            # Format the result for human-readable output
            return self.format_decimal(result)
        except (decimal.InvalidOperation, KeyError):
            return None
    
    def format_decimal(self, dec_value):
        """Format a Decimal to maintain precision but remove unnecessary zeros"""
        # Convert to string with high precision
        str_val = format(dec_value, 'f')
        
        # Handle integers specially
        if dec_value % 1 == 0:
            return str(int(dec_value))
            
        # Keep up to 12 decimal places, removing trailing zeros
        if '.' in str_val:
            integer_part, decimal_part = str_val.split('.')
            # Limit to 12 places and remove trailing zeros
            decimal_part = decimal_part[:12].rstrip('0')
            if decimal_part:
                return f"{integer_part}.{decimal_part}"
            return integer_part
            
        return str_val

class NeuraFlex(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=2, ff_dim=256, num_layers=2):
        super().__init__()
        self.version = "0.0.1"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, embed_dim))
        
        # Mini transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        
        # Add the high-precision math module
        self.math_engine = HighPrecisionMath()
        
    def forward(self, x):
        # Add positional embeddings
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Create a mask for transformer (optional but helpful)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        return self.fc(x)

class TextDataset(Dataset):
    def __init__(self, text, block_size=64):
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size]
        return chunk[:-1], chunk[1:]  # Input and target

def match_question_type(question):
    """Match a question to predefined patterns and return the question type"""
    question = question.lower().strip("?. ")
    
    for qtype, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question):
                return qtype
    
    return "general"

def get_answer_for_question_type(qtype):
    """Get an appropriate answer for a given question type"""
    
    answer_templates = {
        "creator": [
            f"I was created by {NEURAFLEX_KNOWLEDGE['creator']}.",
            f"My developer is {NEURAFLEX_KNOWLEDGE['creator']}.",
            f"{NEURAFLEX_KNOWLEDGE['creator']} built me.",
            f"I'm developed by {NEURAFLEX_KNOWLEDGE['creator']}."
        ],
        "identity": [
            f"I am {NEURAFLEX_KNOWLEDGE['name']}, {NEURAFLEX_KNOWLEDGE['purpose']}.",
            f"My name is {NEURAFLEX_KNOWLEDGE['name']}. I'm a small language model.",
            f"I'm {NEURAFLEX_KNOWLEDGE['name']}, version {NEURAFLEX_KNOWLEDGE['version']}."
        ],
        "version": [
            f"I am version {NEURAFLEX_KNOWLEDGE['version']}.",
            f"My version is {NEURAFLEX_KNOWLEDGE['version']}.",
            f"{NEURAFLEX_KNOWLEDGE['name']} version {NEURAFLEX_KNOWLEDGE['version']}."
        ],
        "purpose": [
            f"I was made {NEURAFLEX_KNOWLEDGE['purpose']}.",
            f"My purpose is to demonstrate language model capabilities at a small scale.",
            f"I exist to show how language models work without requiring massive resources."
        ],
        "architecture": [
            f"I use {NEURAFLEX_KNOWLEDGE['architecture']}.",
            f"My infrastructure consists of {NEURAFLEX_KNOWLEDGE['infrastructure']}.",
            f"I'm built with {NEURAFLEX_KNOWLEDGE['architecture']}."
        ],
        "parameters": [
            f"I have {NEURAFLEX_KNOWLEDGE['parameters']}.",
            f"My model size is {NEURAFLEX_KNOWLEDGE['parameters']}.",
            f"I'm a small model with {NEURAFLEX_KNOWLEDGE['parameters']}."
        ],
        "general": [
            f"I am {NEURAFLEX_KNOWLEDGE['name']}, a small language model created by {NEURAFLEX_KNOWLEDGE['creator']}.",
            f"I'm a tiny language model with {NEURAFLEX_KNOWLEDGE['parameters']}.",
            f"I'm {NEURAFLEX_KNOWLEDGE['name']}, designed to run on standard laptops."
        ]
    }
    
    return random.choice(answer_templates.get(qtype, answer_templates["general"]))

def check_for_math_problem(text):
    """Check if the text contains a math problem and solve it if found"""
    # Try to extract a math problem
    match = MATH_PATTERN.search(text)
    if match:
        num1, operator, num2 = match.groups()
        math_engine = HighPrecisionMath()
        result = math_engine.calculate(num1, operator, num2)
        if result is not None:
            return f"{num1} {operator} {num2} = {result}"
    return None

def create_math_training_examples():
    """Create a set of math training examples for the model"""
    examples = []
    operations = ['+', '-', '*', '/']
    
    # Add simple integer arithmetic
    for _ in range(50):
        a = random.randint(1, 10**12)
        b = random.randint(1, 100)
        op = random.choice(operations)
        
        if op == '/' and b == 0:
            b = 1  # Avoid division by zero
            
        examples.append(f"Q: What is {a} {op} {b}?")
        
        # Calculate result
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            # Use decimal for division to maintain precision
            result = decimal.Decimal(a) / decimal.Decimal(b)
            result = str(result).rstrip('0').rstrip('.') if '.' in str(result) else str(result)
        
        examples.append(f"A: {a} {op} {b} = {result}")
    
    # Add decimal examples for precision testing
    for _ in range(50):
        a = round(random.uniform(0, 10**6), random.randint(1, 12))
        b = round(random.uniform(0, 10**3), random.randint(1, 12))
        op = random.choice(operations)
        
        if op == '/' and abs(b) < 0.00001:
            b = 1.0  # Avoid division by very small numbers
            
        examples.append(f"Q: Calculate {a} {op} {b} to 12 decimal places")
        
        # Calculate using the HighPrecisionMath class
        math_engine = HighPrecisionMath()
        result = math_engine.calculate(str(a), op, str(b))
        
        examples.append(f"A: {a} {op} {b} = {result}")
        
    return "\n".join(examples)

def create_sample_text(filename="sample_input.txt", size=100000):
    """Creates a sample text file if none exists"""
    if os.path.exists(filename):
        return
    
    print(f"Creating sample text file: {filename}")
    
    # Basic description
    sample_text = f"""
    {NEURAFLEX_KNOWLEDGE['name']} is {NEURAFLEX_KNOWLEDGE['purpose']}.
    Despite its small size, it demonstrates the core principles behind large language models
    like GPT, but at a fraction of the cost and computational requirements.
    
    {NEURAFLEX_KNOWLEDGE['name']} was created by {NEURAFLEX_KNOWLEDGE['creator']}.
    The creator of this model is {NEURAFLEX_KNOWLEDGE['creator']}.
    {NEURAFLEX_KNOWLEDGE['creator']} developed this language model.
    
    The model architecture is {NEURAFLEX_KNOWLEDGE['architecture']}.
    It runs on {NEURAFLEX_KNOWLEDGE['infrastructure']}.
    
    The model has {NEURAFLEX_KNOWLEDGE['parameters']}.
    Version {NEURAFLEX_KNOWLEDGE['version']} is the current release.
    
    Q: Who created you?
    A: I was created by {NEURAFLEX_KNOWLEDGE['creator']}.
    
    Q: What is your name?
    A: My name is {NEURAFLEX_KNOWLEDGE['name']}.
    
    Q: What version are you?
    A: I am version {NEURAFLEX_KNOWLEDGE['version']}.
    
    Q: How do you work?
    A: I use {NEURAFLEX_KNOWLEDGE['architecture']}.
    
    Q: What is your purpose?
    A: I was designed {NEURAFLEX_KNOWLEDGE['purpose']}.
    
    Q: How many parameters do you have?
    A: I have {NEURAFLEX_KNOWLEDGE['parameters']}.

    High-precision Mathematics Examples:
    {create_math_training_examples()}
    """
    
    # Expand the text to reach desired size
    sample_text = sample_text * (size // len(sample_text) + 1)
    sample_text = sample_text[:size]
    
    with open(filename, 'w') as f:
        f.write(sample_text)

def train_model(model, dataloader, num_epochs=10, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    print(f"Training NeuraFlex v{model.version} on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"neuraflex_v{model.version}_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"neuraflex_v{model.version}_final.pth")
    return model

def generate_text(model, start_text="The", max_length=100, temperature=1.0, device="cpu"):
    model.eval()
    model = model.to(device)
    
    # Check for math problem
    math_result = check_for_math_problem(start_text)
    if math_result:
        return math_result
    
    # Check if this is a question about NeuraFlex
    if '?' in start_text or any(word in start_text.lower() for word in ["who", "what", "how", "which", "where", "tell", "give"]):
        # Identify question type and get appropriate answer
        qtype = match_question_type(start_text)
        answer = get_answer_for_question_type(qtype)
        
        print(f"Answering as type: {qtype}")
        return answer
    
    print(f"Generating with NeuraFlex v{model.version}...")
    
    # Regular text generation for non-questions
    input_seq = torch.tensor([ord(c) for c in start_text], dtype=torch.long).to(device)
    generated_text = start_text
    
    with torch.no_grad():
        for _ in range(max_length):
            # Take the last block_size tokens
            if len(input_seq) > 63:  # model context size is 64, we use 63 to predict the next token
                input_seq = input_seq[-63:]
            
            # Forward pass
            outputs = model(input_seq.unsqueeze(0))
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(next_token_logits, dim=0)
            
            # Sample from the distribution
            next_token = torch.multinomial(probabilities, 1)
            
            # Add to input sequence
            input_seq = torch.cat([input_seq, next_token])
            
            # Convert to character and add to result
            next_char = chr(next_token.item())
            generated_text += next_char
    
    return generated_text

def answer_question(model, question, device="cpu"):
    """Specialized function to answer questions about NeuraFlex"""
    qtype = match_question_type(question)
    return get_answer_for_question_type(qtype)

if __name__ == "__main__":
    print("=" * 50)
    print("NeuraFlex LLM v0.0.1 - Training Module")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Create sample text if needed
    create_sample_text()
    
    # 2. Load text data
    text = open("sample_input.txt").read()
    print(f"Loaded {len(text)} characters of text")
    
    # 3. Create dataset and dataloader
    dataset = TextDataset(text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. Initialize model (~100K parameters)
    model = NeuraFlex(
        vocab_size=256,  # Using ASCII characters
        embed_dim=128,
        num_heads=2,
        ff_dim=256,
        num_layers=2
    )
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # 5. Train the model
    model = train_model(model, dataloader, num_epochs=10, device=device)
    
    # 6. Generate some text
    generated = generate_text(model, start_text="NeuraFlex is ", max_length=200)
    print("\nGenerated Text Sample:")
    print("-" * 50)
    print(generated)
    print("-" * 50)
    
    # 7. Test question answering
    questions = [
        "Who created you?",
        "What's your name?",
        "What version are you?",
        "How many parameters do you have?",
        "What is your architecture?",
        "What is your purpose?"
    ]
    
    print("\nQuestion Answering Test:")
    print("-" * 50)
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {generate_text(model, q, max_length=0)}")
        print()
    
    # 8. Test math capabilities
    math_questions = [
        "123456789.987654321 + 0.123456789",
        "999999999999 - 888888888888",
        "123.456789 * 987.654321",
        "987654321.123456789 / 123.456789"
    ]
    
    print("\nMath Calculation Test:")
    print("-" * 50)
    for q in math_questions:
        print(f"Q: {q}")
        print(f"A: {generate_text(model, q, max_length=0)}")
        print()
    
    print("-" * 50) 