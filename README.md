# NeuraFlex - Tiny LLM Implementation

**Version: 0.0.1**

NeuraFlex is a minimal language model implementation designed to run on standard laptops, demonstrating the core principles behind large language models at a fraction of the computational requirements.

## Overview

This project implements a small-scale language model (approximately 100K parameters) based on the transformer architecture. Despite its small size, NeuraFlex demonstrates the fundamental concepts behind modern LLMs like GPT, but designed to run on consumer hardware.

## Features

- **Lightweight**: ~100K parameters, runs on CPU
- **Transformer-based**: Employs the same architecture principles as larger models
- **Character-level**: Works with raw text instead of requiring specialized tokenization
- **Educational**: Clear, documented code that shows how LLMs function
- **Resource-efficient**: Runs on standard laptops without specialized hardware

## Requirements

- Python 3.6+
- PyTorch (1.0+)

```bash
pip install torch
```

## Usage

### Training the Model

To train the model on the included sample text (or your own text):

```bash
python neuraflex.py
```

The script will automatically:
1. Create sample training data if none exists
2. Initialize a NeuraFlex model with ~100K parameters
3. Train for 10 epochs (configurable)
4. Save checkpoints and a final model
5. Generate a sample text output

### Generating Text

After training, you can generate text with the trained model:

```bash
python generate_neuraflex.py --prompt "NeuraFlex can" --length 300
```

### Web Interface

To use the ChatGPT-like interface:

1. Install required dependencies:
```bash
pip install flask
```

2. Start the web server:
```bash
python app.py
```

3. Open your browser to:
```bash
http://localhost:5000
```

Features:
- Real-time chat interface
- Conversation history
- Temperature control
- Responsive design

Options:
- `--model_path`: Path to the trained model (default: neuraflex_v0.0.1_final.pth)
- `--prompt`: Starting text for generation (default: "NeuraFlex is")
- `--length`: Number of characters to generate (default: 200)
- `--temperature`: Sampling temperature; higher values give more random results (default: 1.0)

## How It Works

NeuraFlex implements a miniature transformer-based language model:

1. **Embedding Layer**: Converts characters to vector representations
2. **Positional Encoding**: Adds positional information to the embeddings
3. **Transformer Layers**: Processes input with self-attention and feed-forward networks
4. **Output Layer**: Projects to character probability distribution

During generation, the model:
1. Takes an input sequence (prompt)
2. Predicts the next character distribution
3. Samples from this distribution to select the next character
4. Adds this character to the input and repeats

## Limitations

As a tiny model with limited parameters and training data, NeuraFlex will produce basic text compared to commercial LLMs. It serves primarily as a demonstration and educational tool rather than a production-ready system.

## Future Improvements

- Implement better tokenization (BPE or WordPiece)
- Add attention efficiency optimizations
- Support for training on custom datasets
- Parameter-efficient fine-tuning
- Quantization support

## License

MIT 