import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class TextClassifier:
    def __init__(self, embedding_method="bert", model_name="distilbert-base-uncased"):
        self.embedding_method = embedding_method
        self.vectorizer = None
        self.classifier = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}

        try:
            if embedding_method == "bert":
                print("Loading Hugging Face BERT model...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)

            elif embedding_method == "sentence-bert":
                print("Loading Sentence-BERT model...")
                self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

        except Exception as e:
            print(f"⚠ Failed to initialize {embedding_method}: {e}")
            self.embedding_method = None  # disable if loading fails

    def preprocess_data(self, file_path):
        """Extract ticket number + issue description, assign category labels"""
        print(f"📂 Reading from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rows = []
        # ✅ Correct regex (escaped properly)
        ticket_pattern = re.findall(r"(Ticket\s*#[-\w]+(?:\s*-\s*[^\n\r]+)?)", content)

        for ticket_text in ticket_pattern:
            ticket_number_match = re.search(r"(Ticket\s*#[-\w]+)", ticket_text)
            if ticket_number_match:
                ticket_number = ticket_number_match.group(1)
            else:
                continue

            issue_text = ticket_text.lower()

            if "database" in issue_text:
                y_label = "database crash"
            elif "software" in issue_text or "application" in issue_text:
                y_label = "software crash"
            elif "server" in issue_text or "power supply" in issue_text:
                y_label = "server crash"
            elif any(k in issue_text for k in ["system", "cpu", "memory", "blue screen", "thermal"]):
                y_label = "system crash"
            elif any(k in issue_text for k in ["network", "connection", "nic", "ethernet"]):
                y_label = "network issue"
            else:
                y_label = "other issue"

            rows.append({"ticket_number": ticket_number, "x_text": issue_text, "y_label": y_label})

        df = pd.DataFrame(rows)
        print(f"✅ Found {len(df)} tickets")
        return df

    def create_embeddings(self, texts):
        """Generate embeddings based on chosen method"""
        print(f"Embedding with: {self.embedding_method}")

        try:
            if self.embedding_method == "count":
                if self.vectorizer is None:
                    self.vectorizer = CountVectorizer()
                    return self.vectorizer.fit_transform(texts).toarray()
                return self.vectorizer.transform(texts).toarray()

            elif self.embedding_method == "tfidf":
                if self.vectorizer is None:
                    self.vectorizer = TfidfVectorizer()
                    return self.vectorizer.fit_transform(texts).toarray()
                return self.vectorizer.transform(texts).toarray()

            elif self.embedding_method == "bert":
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).numpy()

            elif self.embedding_method == "sentence-bert":
                return self.sbert.encode(texts)

        except Exception as e:
            print(f"⚠ Embedding failed for {self.embedding_method}: {e}")
            return None

    def train_and_predict_all(self, df):
        """Train classifier and predict labels for all tickets, print accuracy"""
        if df.empty:
            print("❌ No data.")
            return df

        if not self.embedding_method:
            print("❌ Skipping due to initialization failure.")
            return df

        embeddings = self.create_embeddings(df["x_text"].tolist())
        if embeddings is None:
            print("❌ Skipping due to embedding failure.")
            return df

        unique_labels = df["y_label"].unique()
        self.label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
        self.reverse_label_mapping = {i: lab for lab, i in self.label_mapping.items()}
        y = df["y_label"].map(self.label_mapping).values

        print("\n🚀 Training XGBoost...")
        self.classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        self.classifier.fit(embeddings, y)

        y_pred = self.classifier.predict(embeddings)
        df["predicted_label"] = [self.reverse_label_mapping[i] for i in y_pred]

        acc = accuracy_score(y, y_pred)
        print(f"\n📊 Accuracy on training data: {acc:.4f}")

        print("\n📑 Classification Report:")
        print(classification_report(y, y_pred, target_names=unique_labels))

        print("\n--- Results on all tickets ---")
        print(df.to_string(index=False))

        out_file = f"ticket_predictions_{self.embedding_method}.csv"
        df.to_csv(out_file, index=False)
        print(f"\n💾 Saved predictions to {out_file}")

        return df


if __name__ == "__main__":
    combined_file_path = "C:/Users/yugay/OneDrive/Desktop/Web page/web/combined_html_content.txt"
    methods = ["count", "tfidf", "bert", "sentence-bert"]

    for m in methods:
        print(f"\n================ {m.upper()} ================")
        try:
            clf = TextClassifier(embedding_method=m)
            df = clf.preprocess_data(combined_file_path)
            if not df.empty:
                results = clf.train_and_predict_all(df)
        except Exception as e:
            print(f"⚠ Skipping {m} due to error: {e}")
