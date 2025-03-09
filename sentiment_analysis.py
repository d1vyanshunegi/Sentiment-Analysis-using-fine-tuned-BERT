from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the saved model and tokenizer
model_path = "fine_tuned_bert_imdb"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example text for testing
text = "Average at best. Won't recommend."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)

# Run the text through the model (make sure to disable gradients for inference)
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted label (0: negative, 1: positive)
predicted_class = torch.argmax(logits, dim=1).item()
print("Predicted sentiment:", "positive" if predicted_class == 1 else "negative")
