"""
Train Classifier
================
Train TF-IDF + Random Forest baseline and optional DistilBERT fine-tuning.
Source data: ArXiv, PubMed, Wikipedia live API corpus.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline(X, y, test_size=0.2, random_state=42):
    """Train a TF-IDF + Random Forest baseline with a Logistic Regression ensemble."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Primary model: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    # Secondary model: Logistic Regression (for ensemble confidence)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    # Save models
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")

    print(f"[train_classifier] Random Forest trained: {rf.score(X_test, y_test):.4f} accuracy")
    print(f"[train_classifier] Logistic Regression trained: {lr.score(X_test, y_test):.4f} accuracy")

    return rf, lr, X_train, X_test, y_train, y_test


def train_distilbert(df: pd.DataFrame, label_col="category", test_size=0.2, random_state=42, epochs=3):
    """Fine-tune DistilBERT for classification (optional, GPU recommended)."""
    try:
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset, DatasetDict
        import torch
    except ImportError:
        print("[train_classifier] transformers/torch not installed — skipping DistilBERT")
        return None

    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    df["label"] = le.transform(df[label_col])

    # Truncate text for DistilBERT (512 token limit)
    df["bert_text"] = df["text"].str[:1000]

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[["bert_text", "label"]].rename(columns={"bert_text": "text"})),
        "test": Dataset.from_pandas(test_df[["bert_text", "label"]].rename(columns={"bert_text": "text"})),
    })

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(le.classes_),
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "distilbert"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=str(MODELS_DIR / "distilbert_logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(MODELS_DIR / "distilbert_best")
    print("[train_classifier] DistilBERT fine-tuning complete")
    return trainer


def run_training():
    """Run the full training pipeline."""
    X = np.load(PROCESSED_DIR / "X_tfidf.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    rf, lr, X_train, X_test, y_train, y_test = train_baseline(X, y)

    # Optionally train DistilBERT if transformers is available
    df = pd.read_parquet(PROCESSED_DIR / "cleaned_documents.parquet")
    try:
        train_distilbert(df)
    except Exception as exc:
        print(f"[train_classifier] DistilBERT skipped: {exc}")

    return rf, lr, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_training()
