from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import EmotionDataset

class BERTTrainer:
    def __init__(self, model_name="bert-base-uncased", num_labels=6, max_len=128, batch_size=16, epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs

    def load_data_from_csv(self, filepath):
        data = pd.read_csv(filepath)
        texts = data['Text'].tolist()
        labels = data['Label'].tolist()
        return texts, labels

    def train_from_csv(self, filepath):
        texts, labels = self.load_data_from_csv(filepath)
        return self.train(texts, labels)

    def train(self, texts, labels):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

        train_dataset = EmotionDataset.EmotionDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_dataset = EmotionDataset.EmotionDataset(val_texts, val_labels, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader)}")
            self.validate(val_loader)

        return label_encoder

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Loss: {total_loss / len(val_loader)}, Accuracy: {accuracy}")
