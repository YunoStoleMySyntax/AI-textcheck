import os
import sys
import subprocess
import importlib
import re
import string
from collections import Counter
import numpy as np
from tkinter import filedialog, messagebox, Tk, Label, Button, StringVar, OptionMenu, Frame, Text, Scrollbar, END, Menu, Entry
import tkinter as tk
import threading
import logging
import pdfplumber
from docx import Document
import nltk
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy.stats import entropy
import torch
import asyncio
import json
#YunoStoleMySyntax- This is my script to improve a language model to be more human-like and catch some cheaters in the progress. 
# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load models for different text lengths
small_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
small_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
large_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
large_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Supported languages
supported_languages = {
    'English': 'english',
    'Spanish': 'spanish',
    'French': 'french',
    'German': 'german',
    'Dutch': 'dutch',
    'Italian': 'italian',
    'Portuguese': 'portuguese',
    'Russian': 'russian',
    'Chinese': 'chinese',
    'Japanese': 'japanese',
    'Korean': 'korean'
}

preferences_file = "preferences.json"

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install {package}: {e}")
        messagebox.showerror("Installation Error", f"Failed to install {package}. Please install it manually.")

required_packages = ["nltk", "torch", "transformers", "scikit-learn", "pdfplumber", "python-docx"]
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        install(package)

def save_preferences(preferences):
    try:
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f)
    except Exception as e:
        logging.error(f"Error saving preferences: {e}")

def load_preferences():
    if os.path.exists(preferences_file):
        try:
            with open(preferences_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading preferences: {e}")
            return {}
    return {}

preferences = load_preferences()
selected_languages = preferences.get('languages', list(supported_languages.keys()))
threshold_value = preferences.get('threshold', 50.0)

def calculate_perplexity(text, model, tokenizer):
    try:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        input_ids = encodings.input_ids
        if input_ids.size(1) == 0:
            return float('inf')

        max_length = model.config.n_positions
        stride = max_length // 2
        lls = []

        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i - stride, 0)
            end_loc = min(i + max_length, input_ids.size(1))
            input_ids_chunk = input_ids[:, begin_loc:end_loc]

            target_ids = input_ids_chunk.clone()
            target_ids[:, :-1] = -100

            with torch.no_grad():
                outputs = model(input_ids_chunk, labels=target_ids)
                log_likelihood = outputs.loss * input_ids_chunk.size(1)
            lls.append(log_likelihood)

        total_log_likelihood = torch.stack(lls).sum()
        ppl = torch.exp(total_log_likelihood / input_ids.size(1))
        return ppl.item()
    except Exception as e:
        logging.error(f"Error calculating perplexity: {e}")
        return float('inf')

def calculate_entropy(text, language):
    try:
        if language not in supported_languages:
            return float('inf')
        word_freq = Counter(word_tokenize(text.lower(), language=supported_languages[language]))
        total_words = sum(word_freq.values())
        word_probs = [freq / total_words for freq in word_freq.values()]
        return entropy(word_probs) if word_probs else float('inf')
    except Exception as e:
        logging.error(f"Error calculating entropy: {e}")
        return float('inf')

def get_ngram_frequencies(text, n=2, language='english'):
    try:
        if language not in supported_languages:
            return Counter()
        words = word_tokenize(text.lower(), language=supported_languages[language])
        n_grams = ngrams(words, n)
        return Counter(n_grams)
    except Exception as e:
        logging.error(f"Error getting n-gram frequencies: {e}")
        return Counter()

def average_sentence_length(text):
    try:
        sentences = re.split(r'[.!?]', text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    except Exception as e:
        logging.error(f"Error calculating average sentence length: {e}")
        return 0

def punctuation_distribution(text):
    try:
        punctuation_counts = Counter(char for char in text if char in string.punctuation)
        total_punctuations = sum(punctuation_counts.values())
        return {p: count / total_punctuations for p, count in punctuation_counts.items()} if total_punctuations else {}
    except Exception as e:
        logging.error(f"Error calculating punctuation distribution: {e}")
        return {}

def calculate_tfidf_cosine_similarity(text, corpus, language):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus + [text])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return np.mean(cosine_similarities) if cosine_similarities.size else 0.0
    except Exception as e:
        logging.error(f"Error calculating TF-IDF cosine similarity: {e}")
        return 0.0

def pos_tag_distribution(text, language):
    try:
        if language not in supported_languages:
            return {}
        words = word_tokenize(text, language=supported_languages[language])
        tags = pos_tag(words, lang=language)
        tag_counts = Counter(tag for word, tag in tags)
        total_tags = sum(tag_counts.values())
        return {tag: count / total_tags for tag, count in tag_counts.items()} if total_tags else {}
    except Exception as e:
        logging.error(f"Error calculating POS tag distribution: {e}")
        return {}

def detect_ai_patterns(text, corpus, threshold, model, tokenizer, language='english'):
    try:
        perplexity_score = calculate_perplexity(text, model, tokenizer)
        entropy_score = calculate_entropy(text, language)
        bigram_freq = get_ngram_frequencies(text, n=2, language=language)
        trigram_freq = get_ngram_frequencies(text, n=3, language=language)
        avg_sent_len = average_sentence_length(text)
        punct_dist = punctuation_distribution(text)
        tfidf_similarity = calculate_tfidf_cosine_similarity(text, corpus, language)
        pos_dist = pos_tag_distribution(text, language)

        # Normalization improvements
        normalized_perplexity = min(1.0, max(0.0, (perplexity_score - 10) / 100))
        normalized_entropy = min(1.0, max(0.0, (5 - entropy_score) / 5))
        normalized_repetitive_phrases = (sum(count > 1 for count in bigram_freq.values()) +
                                         sum(count > 1 for count in trigram_freq.values())) / 100
        normalized_avg_sent_len = avg_sent_len / 20
        normalized_punct_dist = len(set(punct_dist.values())) / 10
        normalized_tfidf_similarity = tfidf_similarity
        normalized_pos_dist = len(set(pos_dist.values())) / 10

        ai_likelihood = (normalized_perplexity + normalized_entropy + normalized_repetitive_phrases +
                         normalized_avg_sent_len + normalized_punct_dist + normalized_tfidf_similarity +
                         normalized_pos_dist) / 7 * 100

        threshold = min(100, max(0, threshold))
        explanations = []

        if normalized_perplexity > 0.5:
            explanations.append("High perplexity might indicate AI-generated text.")
        if normalized_entropy < 0.5:
            explanations.append("Low entropy could be a sign of AI-generated text.")
        if normalized_repetitive_phrases > 0.5:
            explanations.append("High frequency of repetitive phrases can be characteristic of AI-generated text.")
        if normalized_avg_sent_len < 0.5:
            explanations.append("Short average sentence length might be indicative of AI text.")
        if normalized_punct_dist > 0.5:
            explanations.append("Uniform punctuation distribution could suggest AI-generated content.")
        if normalized_tfidf_similarity < 0.2:
            explanations.append("Low TF-IDF similarity with the corpus may imply AI generation.")
        if normalized_pos_dist > 0.5:
            explanations.append("Uniformity in POS tags can be a sign of AI text.")

        is_ai = ai_likelihood > threshold
        explanations.append(f"Overall AI likelihood score: {ai_likelihood:.2f}%")
        explanations.append(f"Threshold for AI detection: {threshold}%")
        explanations.append("Conclusion: " + ("Likely AI-generated" if is_ai else "Likely human-written"))

        indicators = {
            'perplexity': perplexity_score,
            'entropy': entropy_score,
            'repetitive_phrases': sum(count > 1 for count in bigram_freq.values()) + sum(count > 1 for count in trigram_freq.values()),
            'average_sentence_length': avg_sent_len,
            'punctuation_uniformity': len(set(punct_dist.values())) < 3,
            'tfidf_similarity': tfidf_similarity,
            'pos_uniformity': len(set(pos_dist.values())) < 5
        }

        return indicators, ai_likelihood, "\n".join(explanations)
    except Exception as e:
        logging.error(f"Error detecting AI patterns: {e}")
        return {}, 0, "An error occurred during analysis."

def extract_text_from_file(file_path):
    try:
        if file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages)
        elif file_path.lower().endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from file {file_path}: {e}")
        return ""

def process_file(file_path, threshold, language):
    text = extract_text_from_file(file_path)
    if text:
        corpus = []  # Provide a corpus or load a predefined corpus for TF-IDF calculation
        model = large_model if len(text) > 1000 else small_model
        tokenizer = large_tokenizer if len(text) > 1000 else small_tokenizer
        indicators, ai_likelihood, explanation = detect_ai_patterns(text, corpus, threshold, model, tokenizer, language)
        return indicators, ai_likelihood, explanation
    return {}, 0, "No text extracted from file."

class AITextDetectorApp:
    def __init__(self, root):
        self.root = root
        root.title("AI Text Detector")
        self.create_widgets()

    def create_widgets(self):
        self.file_path_var = StringVar()
        self.language_var = StringVar(value=selected_languages[0])
        self.threshold_var = StringVar(value=str(threshold_value))

        frame = Frame(self.root)
        frame.pack(padx=10, pady=10)

        Label(frame, text="File Path:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        Entry(frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        Button(frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        Label(frame, text="Language:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        OptionMenu(frame, self.language_var, *supported_languages.keys()).grid(row=1, column=1, padx=5, pady=5)

        Label(frame, text="Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        Entry(frame, textvariable=self.threshold_var).grid(row=2, column=1, padx=5, pady=5)

        Button(frame, text="Detect AI", command=self.detect_ai).grid(row=3, column=0, columnspan=3, pady=10)

        self.result_text = Text(self.root, height=20, width=80)
        self.result_text.pack(padx=10, pady=10)
        self.result_text.config(state=tk.DISABLED)

        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.file_menu = Menu(self.menu)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=root.quit)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_var.set(file_path)

    def detect_ai(self):
        file_path = self.file_path_var.get()
        language = self.language_var.get()
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the threshold.")
            return

        if not file_path:
            messagebox.showerror("Input Error", "Please select a file.")
            return

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, f"Processing file: {file_path}\n")
        self.result_text.insert(END, f"Language: {language}\n")
        self.result_text.insert(END, f"Threshold: {threshold}\n")

        def worker():
            indicators, ai_likelihood, explanation = process_file(file_path, threshold, language)
            self.result_text.insert(END, "\nIndicators:\n")
            for key, value in indicators.items():
                self.result_text.insert(END, f"{key}: {value}\n")
            self.result_text.insert(END, f"\nAI Likelihood: {ai_likelihood:.2f}%\n")
            self.result_text.insert(END, f"\nExplanation:\n{explanation}\n")
            self.result_text.config(state=tk.DISABLED)

        threading.Thread(target=worker).start()

if __name__ == "__main__":
    root = Tk()
    app = AITextDetectorApp(root)
    root.mainloop()
