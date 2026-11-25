from flask import Flask, request, jsonify, render_template_string
from neuraflex import NeuraFlex, generate_text, match_question_type, get_answer_for_question_type, check_for_math_problem
import torch
import os

app = Flask(__name__)
model = None

def load_model():
    global model
    try:
        # Initialize model
        model = NeuraFlex(
            vocab_size=256,
            embed_dim=128,
            num_heads=2,
            ff_dim=256,
            num_layers=2
        )
        model.version = "0.0.1"
        
        # Look for model file
        model_path = 'neuraflex_v0.0.1_final.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found.")
            print("Please train the model first with 'python neuraflex.py'")
            
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NeuraFlex Chat</title>
    <style>
        body { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
        }
        .chat-container { 
            border: 1px solid #e0e0e0; 
            border-radius: 8px; 
            padding: 20px; 
            margin-top: 20px;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-messages { 
            height: 500px; 
            overflow-y: auto; 
            margin-bottom: 20px; 
            padding: 10px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .message { 
            margin: 10px 0; 
            padding: 12px 15px; 
            border-radius: 18px; 
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message { 
            background-color: #007bff; 
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        .bot-message { 
            background-color: #e9ecef; 
            color: #343a40;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        .message-container {
            display: flex;
            margin-bottom: 10px;
        }
        .input-container {
            display: flex;
            margin-top: 15px;
        }
        input[type="text"] { 
            flex: 1;
            padding: 12px; 
            border: 1px solid #ced4da;
            border-radius: 20px;
            margin-right: 10px;
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
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>NeuraFlex v0.0.1</h1>
        <p>Tiny LLM (~100K parameters)</p>
    </div>
    
    <div class="chat-container">
        <div class="settings">
            <label for="temperature">Temperature:</label>
            <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.8">
            <span id="temp-value">0.8</span>
            
            <label for="length">Max Length:</label>
            <input type="number" id="length" min="10" max="500" value="200" style="width: 60px">
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message-container">
                <div class="message bot-message">
                    Hi! I'm NeuraFlex v0.0.1, a tiny language model with only ~100K parameters, created by Saptarshi Halder. 
                    You can ask me about myself, my creator, or my architecture. Or just chat with me!
                </div>
            </div>
            <div class="message-container">
                <div class="message bot-message">
                    I can also solve math problems with high precision (up to 12 decimal places)! Try something like "123.456 + 789.012" or "987654321.123456789 / 123.456789".
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">NeuraFlex is thinking...</div>
        
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
            <button onclick="sendMessage()" id="send-button">Send</button>
        </div>
        
        <div class="features">
            Features: Model identity Q&A, High-precision math (12 decimal places), Text generation
        </div>
    </div>

    <script>
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const loading = document.getElementById('loading');
        
        tempSlider.oninput = function() {
            tempValue.textContent = this.value;
        }
        
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function appendMessage(message, isUser) {
            const messagesDiv = document.getElementById('chat-messages');
            const containerDiv = document.createElement('div');
            containerDiv.className = 'message-container';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = message;
            
            containerDiv.appendChild(messageDiv);
            messagesDiv.appendChild(containerDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            appendMessage(message, true);
            input.value = '';
            
            // Disable input while generating
            userInput.disabled = true;
            sendButton.disabled = true;
            loading.style.display = 'block';
            
            try {
                const temperature = parseFloat(document.getElementById('temperature').value);
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
                appendMessage(data.response, false);
            } catch (error) {
                appendMessage('Error communicating with NeuraFlex. Please check if the model is loaded correctly.', false);
            } finally {
                // Re-enable input after response received
                userInput.disabled = false;
                sendButton.disabled = false;
                loading.style.display = 'none';
                userInput.focus();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    global model
    if model is None:
        success = load_model()
        if not success:
            return jsonify({'response': 'Error: Model failed to load. Please train the model first with "python neuraflex.py"'})
    
    data = request.json
    user_input = data['message']
    temperature = data.get('temperature', 0.8)
    max_length = data.get('length', 200)
    
    try:
        # Check for math problems first
        math_result = check_for_math_problem(user_input)
        if math_result:
            return jsonify({'response': math_result})
        
        # Check if this is a question about NeuraFlex
        if '?' in user_input or any(word in user_input.lower() for word in ["who", "what", "how", "which", "where", "tell me", "state", "give me"]):
            qtype = match_question_type(user_input)
            response = get_answer_for_question_type(qtype)
            return jsonify({'response': response})
        
        # Regular text generation
        response = generate_text(
            model,
            start_text=user_input,
            max_length=int(max_length),
            temperature=float(temperature),
            device='cpu'
        )
        # Only return the generated part, not including the prompt
        return jsonify({'response': response[len(user_input):]})
    except Exception as e:
        return jsonify({'response': f'Error generating response: {str(e)}'})

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    print("NeuraFlex Chat UI starting...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True) 