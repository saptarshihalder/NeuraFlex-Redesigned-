from flask import Flask, request, jsonify, render_template_string
import torch
import os
import json

# Import regular NeuraFlex model
from neuraflex import (
    NeuraFlex, 
    generate_text, 
    match_question_type, 
    get_answer_for_question_type, 
    check_for_math_problem
)

# Import QA model
from neuraflex_qa import NeuraFlexQA, answer_question
from transformers import AutoTokenizer

app = Flask(__name__)

# Initialize models
neuraflex_model = None
qa_model = None
qa_tokenizer = None
contexts = {}  # Store contexts for each session

def load_models():
    """Load both NeuraFlex and QA models"""
    global neuraflex_model, qa_model, qa_tokenizer
    
    models_loaded = True
    
    # 1. Load primary NeuraFlex model
    try:
        # Initialize model
        neuraflex_model = NeuraFlex(
            vocab_size=256,
            embed_dim=128,
            num_heads=2,
            ff_dim=256,
            num_layers=2
        )
        neuraflex_model.version = "0.0.1"
        
        # Look for model file
        model_path = 'neuraflex_v0.0.1_final.pth'
        if os.path.exists(model_path):
            neuraflex_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"NeuraFlex base model loaded from {model_path}")
        else:
            print(f"Warning: NeuraFlex model file {model_path} not found.")
            models_loaded = False
            
        neuraflex_model.eval()
    except Exception as e:
        print(f"Error loading NeuraFlex model: {e}")
        models_loaded = False
    
    # 2. Load QA model
    try:
        # Load model
        qa_model_path = 'neuraflex_qa_final.pth'
        model_name = "distilbert-base-uncased"  # Default model name
        
        # Check if we have metadata file with model info
        meta_path = qa_model_path.replace(".pth", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                model_name = metadata.get("base_model", model_name)
        
        # Initialize model and tokenizer
        qa_model = NeuraFlexQA(base_model_name=model_name)
        qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load weights if available
        if os.path.exists(qa_model_path):
            qa_model.load(qa_model_path)
            print(f"QA model loaded from {qa_model_path}")
        else:
            print(f"Warning: QA model file {qa_model_path} not found.")
            models_loaded = False
        
        qa_model.eval()
    except Exception as e:
        print(f"Error loading QA model: {e}")
        models_loaded = False
    
    return models_loaded

# HTML Template for the combined interface
COMBINED_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NeuraFlex AI</title>
    <style>
        body { 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px; 
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .header h1 {
            margin: 0;
            color: #2c3e50;
        }
        .header-meta {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
            background-color: #f8f9fa;
        }
        .tab:hover {
            background-color: #e9ecef;
        }
        .tab.active {
            background-color: #007bff;
            color: white;
            border-color: #007bff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Common components */
        .panel { 
            border: 1px solid #e0e0e0; 
            border-radius: 8px; 
            padding: 20px; 
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .history { 
            height: 400px; 
            overflow-y: auto; 
            margin-bottom: 20px; 
            padding: 10px;
            border-radius: 5px;
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
        }
        .user-message { 
            margin: 10px 0; 
            padding: 12px 15px; 
            border-radius: 18px; 
            background-color: #007bff; 
            color: white;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
            border-bottom-right-radius: 5px;
        }
        .bot-message { 
            margin: 10px 0; 
            padding: 12px 15px; 
            border-radius: 18px; 
            background-color: #e9ecef; 
            color: #343a40;
            max-width: 80%;
            margin-right: auto;
            word-wrap: break-word;
            border-bottom-left-radius: 5px;
        }
        .question { 
            margin: 10px 0; 
            padding: 12px 15px; 
            border-radius: 18px; 
            background-color: #007bff; 
            color: white;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
        }
        .answer { 
            margin: 10px 0; 
            padding: 12px 15px; 
            border-radius: 18px; 
            background-color: #e9ecef; 
            color: #343a40;
            max-width: 80%;
            margin-right: auto;
            word-wrap: break-word;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ced4da;
            border-radius: 5px;
            margin-bottom: 10px;
            font-family: Arial, sans-serif;
            resize: vertical;
        }
        input[type="text"] { 
            width: 100%;
            padding: 12px; 
            border: 1px solid #ced4da;
            border-radius: 20px;
            margin-bottom: 10px;
            font-size: 16px;
        }
        button { 
            padding: 10px 20px; 
            background-color: #007bff; 
            color: white; 
            border: none; 
            border-radius: 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #0069d9;
        }
        .loading {
            display: none;
            margin: 10px 0;
            color: #666;
        }
        .highlight {
            background-color: #ffff99;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .settings {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        .settings label {
            margin-right: 5px;
        }
        .features {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
        .info {
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .input-container input {
            flex: 1;
            margin-bottom: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>NeuraFlex AI</h1>
                <p class="header-meta">Tiny LLM with QA capabilities</p>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('chat')">Chat</div>
            <div class="tab" onclick="switchTab('qa')">Question Answering</div>
        </div>
        
        <!-- Chat Tab Content -->
        <div id="chat-tab" class="tab-content active">
            <div class="panel">
                <div class="settings">
                    <label for="temperature">Temperature:</label>
                    <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.8">
                    <span id="temp-value">0.8</span>
                    
                    <label for="length">Max Length:</label>
                    <input type="number" id="length" min="10" max="500" value="200" style="width: 60px">
                </div>
                
                <div class="history" id="chat-history">
                    <div class="bot-message">
                        Hi! I'm NeuraFlex, a tiny language model created by Saptarshi Halder. 
                        You can ask me about myself, try math problems, or just chat with me!
                    </div>
                    <div class="bot-message">
                        I can solve math problems with high precision (12 decimal places)! Try something like "123.456 + 789.012".
                    </div>
                </div>
                
                <div class="loading" id="chat-loading">NeuraFlex is thinking...</div>
                
                <div class="input-container">
                    <input type="text" id="chat-input" placeholder="Type your message..." autocomplete="off">
                    <button onclick="sendChatMessage()" id="chat-send-button">Send</button>
                </div>
                
                <div class="features">
                    Features: Identity Q&A, High-precision math, Text generation
                </div>
            </div>
        </div>
        
        <!-- QA Tab Content -->
        <div id="qa-tab" class="tab-content">
            <div class="panel">
                <h3>Context Information</h3>
                <p>Enter or update the context used for answering questions:</p>
                <textarea id="context" rows="5" placeholder="Enter text context here...">{{ context }}</textarea>
                <button onclick="updateContext()">Update Context</button>
                
                <h3>Ask a Question</h3>
                <div class="history" id="qa-history"></div>
                
                <div class="loading" id="qa-loading">Thinking...</div>
                
                <div class="input-container">
                    <input type="text" id="qa-input" placeholder="Ask a question about the context..." autocomplete="off">
                    <button onclick="askQuestion()" id="qa-button">Ask</button>
                </div>
                
                <div class="info">
                    <p>This model has been trained on the SQuAD dataset to answer questions based on context.</p>
                    <p>First provide a context, then ask specific questions about that text.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
        }
        
        // Chat functionality
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const chatInput = document.getElementById('chat-input');
        const chatSendButton = document.getElementById('chat-send-button');
        const chatLoading = document.getElementById('chat-loading');
        
        tempSlider.oninput = function() {
            tempValue.textContent = this.value;
        };
        
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });

        function appendChatMessage(message, isUser) {
            const historyDiv = document.getElementById('chat-history');
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'user-message' : 'bot-message';
            messageDiv.textContent = message;
            
            historyDiv.appendChild(messageDiv);
            historyDiv.scrollTop = historyDiv.scrollHeight;
        }

        async function sendChatMessage() {
            const message = chatInput.value.trim();
            
            if (!message) return;
            
            appendChatMessage(message, true);
            chatInput.value = '';
            
            // Disable input while generating
            chatInput.disabled = true;
            chatSendButton.disabled = true;
            chatLoading.style.display = 'block';
            
            try {
                const temperature = parseFloat(tempSlider.value);
                const length = parseInt(document.getElementById('length').value);
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message,
                        temperature: temperature,
                        length: length
                    })
                });
                
                const data = await response.json();
                appendChatMessage(data.response, false);
            } catch (error) {
                appendChatMessage('Error communicating with NeuraFlex. Please check if the model is loaded correctly.', false);
            } finally {
                // Re-enable input after response received
                chatInput.disabled = false;
                chatSendButton.disabled = false;
                chatLoading.style.display = 'none';
                chatInput.focus();
            }
        }
        
        // QA functionality
        const qaInput = document.getElementById('qa-input');
        const qaButton = document.getElementById('qa-button');
        const qaLoading = document.getElementById('qa-loading');
        const sessionId = Date.now().toString();  // Simple session ID
        
        qaInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });

        function appendQA(question, answer, highlight = null) {
            const historyDiv = document.getElementById('qa-history');
            
            // Add question
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question';
            questionDiv.textContent = question;
            historyDiv.appendChild(questionDiv);
            
            // Add answer
            const answerDiv = document.createElement('div');
            answerDiv.className = 'answer';
            
            if (highlight && answer.includes(highlight)) {
                // Highlight the answer in the response
                const parts = answer.split(highlight);
                let highlightedAnswer = '';
                
                for (let i = 0; i < parts.length; i++) {
                    highlightedAnswer += parts[i];
                    if (i < parts.length - 1) {
                        highlightedAnswer += `<span class="highlight">${highlight}</span>`;
                    }
                }
                
                answerDiv.innerHTML = highlightedAnswer;
            } else {
                answerDiv.textContent = answer;
            }
            
            historyDiv.appendChild(answerDiv);
            historyDiv.scrollTop = historyDiv.scrollHeight;
        }

        async function askQuestion() {
            const question = qaInput.value.trim();
            const context = document.getElementById('context').value.trim();
            
            if (!question) return;
            if (!context) {
                appendQA(question, "Please provide a context first before asking questions.");
                return;
            }
            
            // Disable input while processing
            qaInput.disabled = true;
            qaButton.disabled = true;
            qaLoading.style.display = 'block';
            
            try {
                const response = await fetch('/qa', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        question: question,
                        context: context,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                appendQA(question, data.answer, data.highlight);
                
                // Clear question input
                qaInput.value = '';
            } catch (error) {
                appendQA(question, 'Error processing your question. Please try again.');
            } finally {
                // Re-enable input
                qaInput.disabled = false;
                qaButton.disabled = false;
                qaLoading.style.display = 'none';
                qaInput.focus();
            }
        }
        
        async function updateContext() {
            const context = document.getElementById('context').value.trim();
            
            try {
                const response = await fetch('/update_context', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        context: context,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    appendQA("", 'Context updated successfully. You can now ask questions about this text.');
                }
            } catch (error) {
                console.error('Error updating context:', error);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the main interface"""
    return render_template_string(COMBINED_TEMPLATE, context="")

@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat message and return a response"""
    if neuraflex_model is None:
        return jsonify({"response": "Model not loaded. Please check server logs."})
    
    data = request.json
    message = data.get('message', '')
    temperature = data.get('temperature', 0.8)
    max_length = data.get('length', 200)
    
    # Process the message
    try:
        # Check for math problems first
        math_result = check_for_math_problem(message)
        if math_result:
            return jsonify({"response": math_result})
        
        # Check for questions about the model
        question_type = match_question_type(message)
        if question_type != "general":
            answer = get_answer_for_question_type(question_type)
            return jsonify({"response": answer})
        
        # Generate text response
        response = generate_text(neuraflex_model, message, max_new_tokens=max_length, temperature=temperature)
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({"response": f"Error: {str(e)}"})

@app.route('/update_context', methods=['POST'])
def update_context():
    """Update the context for a session"""
    data = request.json
    session_id = data.get('session_id', 'default')
    context = data.get('context', '')
    
    contexts[session_id] = context
    
    return jsonify({"status": "success"})

@app.route('/qa', methods=['POST'])
def qa():
    """Process a question and return an answer"""
    if qa_model is None or qa_tokenizer is None:
        return jsonify({
            "answer": "QA model not loaded. Please check server logs.", 
            "highlight": None
        })
    
    data = request.json
    question = data.get('question', '')
    context = data.get('context', '')
    session_id = data.get('session_id', 'default')
    
    # Store context if not already set
    if context:
        contexts[session_id] = context
    
    # Get context from session
    current_context = contexts.get(session_id, '')
    if not current_context:
        return jsonify({
            "answer": "Please provide a context first before asking questions.", 
            "highlight": None
        })
    
    # Process question using the model
    try:
        # Get answer from model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_answer = answer_question(qa_model, question, current_context, qa_tokenizer, device)
        
        return jsonify({
            "answer": model_answer, 
            "highlight": model_answer  # Highlight the answer in the response
        })
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({
            "answer": f"Error processing your question: {str(e)}", 
            "highlight": None
        })

if __name__ == '__main__':
    if load_models():
        print("All models loaded successfully!")
    else:
        print("Warning: Some models could not be loaded.")
        
    print("Starting NeuraFlex AI server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 