import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join("sentiment_results", "bert_output")
app = Flask(__name__)

print(f"Loading model from {MODEL_DIR}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run the training script and the model files are in the correct directory.")
    tokenizer = None
    model = None

def predict_sentiment(text):
    """Predicts sentiment (Positive/Negative) for a given text."""
    if not model or not tokenizer:
        return "Model not loaded", 0.0

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_label_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label_idx].item()
    
    sentiment = "Positive" if pred_label_idx == 1 else "Negative"
    
    return sentiment, confidence

@app.route('/', methods=['GET'])
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns the prediction."""
    if request.method == 'POST':

        review_text = request.form['review_text']
        
        if review_text:
            sentiment, confidence = predict_sentiment(review_text)
            
            
            return render_template('index.html', 
                                   prediction=sentiment, 
                                   confidence=f"{confidence*100:.2f}%", 
                                   review_text=review_text)
        else:
            
            return render_template('index.html', error="Please enter some text to analyze.")


    return redirect(url_for('home'))



if __name__ == '__main__':
    
    
    app.run(debug=True)