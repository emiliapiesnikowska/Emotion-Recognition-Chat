from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

import torch
import joblib
import os
import tkinter as tk
import BERTTrainer

class EmotionChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Chat")
        self.root.geometry("500x700")
        self.root.configure(bg="#1e1e1e")

        self.trainer, self.label_encoder = self.initialize_trainer()

        if self.trainer and self.label_encoder:
            self.initialize_ui()
        else:
            messagebox.showerror("Error", "Nie można uruchomić aplikacji bez prawidłowego modelu. Sprawdź pliki modelu.")
            self.root.destroy()

    def initialize_trainer(self):
        bert_trainer = BERTTrainer.BERTTrainer()
        try:
            if not os.path.exists("bert_model.pth"):
                label_encoder = bert_trainer.train_from_csv("data/emotion_dataset.csv")
                torch.save(bert_trainer.model.state_dict(), "bert_model.pth")
                joblib.dump(label_encoder, "label_encoder.pkl")
            else:
                bert_trainer.model.load_state_dict(torch.load("bert_model.pth", map_location=torch.device("cpu")))
                label_encoder = joblib.load("label_encoder.pkl")
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            bert_trainer = None
            label_encoder = None
        return bert_trainer, label_encoder

    def initialize_ui(self):
        self.chat_frame = tk.Frame(self.root, bg="#2e2e2e", highlightbackground="#444", highlightthickness=1)
        self.chat_frame.pack(fill="both", expand=True, pady=10, padx=10)

        self.chat_canvas = tk.Canvas(self.chat_frame, bg="#2e2e2e", highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.scrollable_frame = tk.Frame(self.chat_canvas, bg="#2e2e2e")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )

        self.chat_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Textbox for user input
        self.input_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.input_frame.pack(fill="x", padx=10, pady=10)

        self.user_input = tk.Entry(self.input_frame, font=("Arial", 12), bg="#2e2e2e", fg="#ffffff", insertbackground="#ffffff", bd=0, relief="flat", highlightthickness=2, highlightcolor="#4CAF50")
        self.user_input.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=8, ipadx=10)

        # Custom rounded send button
        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            font=("Arial", 12),
            bg="#4CAF50",
            fg="#ffffff",
            activebackground="#45a049",
            relief="flat",
            command=lambda: self.process_message(None),
            bd=0
        )
        self.send_button.pack(side="right", ipadx=20, ipady=8)
        self.send_button.bind("<Enter>", lambda e: self.send_button.configure(bg="#45a049"))
        self.send_button.bind("<Leave>", lambda e: self.send_button.configure(bg="#4CAF50"))

        self.images = self.load_images()

    def load_images(self):
        emotions = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Love"]
        image_dict = {}
        for emotion in emotions:
            try:
                image = Image.open(f"data/{emotion}.png")
                image = image.resize((100, 100))
                image_dict[emotion] = ImageTk.PhotoImage(image)
            except FileNotFoundError:
                messagebox.showerror("Error", f"Image for emotion '{emotion}' not found.")
        return image_dict

    def process_message(self, event):
        user_message = self.user_input.get()
        if not user_message.strip():
            messagebox.showwarning("Warning", "Message cannot be empty!")
            return

        # Predict emotion using BERTTrainer
        detected_emotion = self.predict_emotion(user_message)

        self.add_message_to_chat(user_message, detected_emotion)

    def predict_emotion(self, text):
        if not self.trainer:
            return "neutral"

        encoding = self.trainer.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].to(self.trainer.device)
        attention_mask = encoding['attention_mask'].to(self.trainer.device)

        with torch.no_grad():
            outputs = self.trainer.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction_idx = torch.argmax(logits, dim=1).item()

        return self.label_encoder.inverse_transform([prediction_idx])[0]

    def add_message_to_chat(self, user_message, detected_emotion):
        user_bubble_frame = tk.Frame(self.scrollable_frame, bg="#1e1e1e")
        user_bubble = tk.Label(
            user_bubble_frame,
            text=user_message,
            bg="#4CAF50",
            fg="#ffffff",
            font=("Arial", 12),
            wraplength=250,
            justify="left",
            padx=15,
            pady=10,
            bd=0,
            relief="flat"
        )
        user_bubble.pack(anchor="e")

        user_bubble_frame.pack(pady=5, padx=(100, 10), anchor="e")

        if detected_emotion in self.images:
            print(f"Obraz znaleziony dla emocji: {detected_emotion}")
            emotion_bubble_frame = tk.Frame(self.scrollable_frame, bg="#2e2e2e", padx=10, pady=10)
            emotion_image = tk.Label(
                emotion_bubble_frame,
                image=self.images[detected_emotion],
                bg="#2e2e2e",
                bd=0
            )
            emotion_image.pack(side="left", padx=10)
            emotion_text = tk.Label(
                emotion_bubble_frame,
                text=detected_emotion.capitalize(),
                bg="#2e2e2e",
                fg="#ffffff",
                font=("Arial", 12),
                padx=10,
                wraplength=250,
                justify="left",
                bd=0
            )
            emotion_text.pack(side="left")
            emotion_bubble_frame.pack(anchor="w", pady=5, padx=(10, 100))
        else:
            print(f"Obraz NIE znaleziony dla emocji: {detected_emotion}")
            error_bubble = tk.Label(
                self.scrollable_frame,
                text="Emotion not found.",
                bg="#FF5555",
                fg="#ffffff",
                font=("Arial", 12),
                wraplength=350,
                justify="left",
                padx=15,
                pady=10,
                bd=0,
                relief="flat"
            )
            error_bubble.pack(pady=5, padx=10, anchor="w")

        self.chat_canvas.yview_moveto(1.0)
        self.user_input.delete(0, tk.END)
