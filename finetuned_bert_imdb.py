import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy and weighted F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for IMDb Sentiment Classification")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    # The line below is modified to pass an empty list to parse_args when no
    # command-line arguments are provided.
    args = parser.parse_args([])

    # ------------------------------
    # Data Preparation
    # ------------------------------
    # Load the IMDb dataset from Hugging Face Datasets
    dataset = load_dataset("imdb")

    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tokenization function: truncates to a maximum length
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_length)

    # Apply tokenization to the entire dataset in batched mode
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Create a data collator that dynamically pads inputs within a batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Split the training set further into training and validation (90/10 split)
    split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets["train"]
    val_dataset = split_datasets["test"]
    test_dataset = tokenized_datasets["test"]

    # ------------------------------
    # Model and Training Setup
    # ------------------------------
    # Load pre-trained BERT for sequence classification (binary: positive/negative)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Define training arguments (directory paths, batch sizes, learning rate, etc.)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
    )

    # Create the Trainer with model, datasets, data collator, and metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ------------------------------
    # Training and Evaluation
    # ------------------------------
    trainer.train()

    # Evaluate the model on the test dataset
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)

    # ------------------------------
    # Save the Fine-Tuned Model
    # ------------------------------
    trainer.save_model("fine_tuned_bert_imdb")
    tokenizer.save_pretrained("fine_tuned_bert_imdb")
    print("Model and tokenizer saved in 'fine_tuned_bert_imdb'.")

if __name__ == "__main__":
    main()