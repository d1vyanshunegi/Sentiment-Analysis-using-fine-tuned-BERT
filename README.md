# Sentiment-Analysis-using-fine-tuned-BERT
This project fine-tunes a pre-trained Transformer (BERT) for sentiment classification on the IMDb movie reviews dataset. This project uses Hugging Face’s Transformers and Datasets libraries and follows best practices:

Data Preparation & Tokenization:
Loads the IMDb dataset, tokenizes text using a pre-trained BERT tokenizer, and dynamically pads inputs.

Model Setup:
Loads a pre-trained BERT model for sequence classification (with two output labels).

Training Workflow:
Uses the Hugging Face Trainer API to perform training and evaluation (including custom metric computation for accuracy and F1 score).

Checkpointing & Model Saving:
Saves the best model based on evaluation performance.
