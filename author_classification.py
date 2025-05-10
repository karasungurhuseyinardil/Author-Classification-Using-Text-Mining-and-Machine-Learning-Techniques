import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from joblib import Parallel, delayed
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

# Load texts and labels from folder
def load_texts_from_folder(folder_path):
    texts = []
    labels = []
    for author in os.listdir(folder_path):
        author_folder = os.path.join(folder_path, author)
        if os.path.isdir(author_folder):
            for file_name in os.listdir(author_folder):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(author_folder, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                        labels.append(author)
    return texts, labels

# Extract embeddings using BERT with batch processing
def extract_bert_embeddings(texts, tokenizer, model, max_length=512, batch_size=16, device=None):
    embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings).cpu().numpy()

# Extract TF-IDF features (word-based)
def extract_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_features = vectorizer.fit_transform(texts)
    return tfidf_features, vectorizer

# Extract word-based n-grams (2-grams and 3-grams)
def extract_word_ngrams(texts, max_features=5000, ngram_range=(2, 3)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    ngram_features = vectorizer.fit_transform(texts)
    return ngram_features, vectorizer

# Extract character-based n-grams (2-grams and 3-grams)
def extract_char_ngrams(texts, max_features=5000, ngram_range=(2, 3)):
    vectorizer = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=ngram_range)
    char_ngram_features = vectorizer.fit_transform(texts)
    return char_ngram_features, vectorizer

# Train a classifier
def train_classifier(model_name, X_train, y_train):
    if model_name == 'SVM':
        model = SVC(kernel='linear', cache_size=1000)  # Increase cache size for faster SVM
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Parallelize random forest
    elif model_name == 'NaiveBayes':
        model = MultinomialNB()
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_name == 'XGBoost':
        model = XGBClassifier(n_jobs=-1)  # Parallelize XGBoost
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100,))  # Removed n_jobs as it is not supported
    else:
        raise ValueError("Unsupported model")
    
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Main function (updated to include all methods)
def main():
    # Dataset path
    dataset_path = r"dataset_authorship"
    print("Loading data...")
    texts, labels = load_texts_from_folder(dataset_path)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    X_train_tfidf, tfidf_vectorizer = extract_tfidf_features(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Extract word-based n-grams (2-grams and 3-grams)
    print("Extracting word-based n-grams (2-grams and 3-grams)...")
    X_train_word_ngrams, word_ngram_vectorizer = extract_word_ngrams(X_train, ngram_range=(2, 3))
    X_test_word_ngrams = word_ngram_vectorizer.transform(X_test)

    # Extract character-based n-grams (2-grams and 3-grams)
    print("Extracting character-based n-grams (2-grams and 3-grams)...")
    X_train_char_ngrams, char_ngram_vectorizer = extract_char_ngrams(X_train, ngram_range=(2, 3))
    X_test_char_ngrams = char_ngram_vectorizer.transform(X_test)

    # Extract BERT embeddings
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)

    print("Extracting BERT embeddings for training data...")
    X_train_bert = extract_bert_embeddings(X_train, tokenizer, bert_model, device=device)
    print("Extracting BERT embeddings for testing data...")
    X_test_bert = extract_bert_embeddings(X_test, tokenizer, bert_model, device=device)

    return (X_train_tfidf, X_test_tfidf, 
            X_train_word_ngrams, X_test_word_ngrams, 
            X_train_char_ngrams, X_test_char_ngrams, 
            X_train_bert, X_test_bert, 
            y_train, y_test, label_encoder)

# Call main and use returned values
if __name__ == "__main__":
    (X_train_tfidf, X_test_tfidf, 
     X_train_word_ngrams, X_test_word_ngrams, 
     X_train_char_ngrams, X_test_char_ngrams, 
     X_train_bert, X_test_bert, 
     y_train, y_test, label_encoder) = main()

    # Example: Train and evaluate a model using TF-IDF features
    model_name = 'MLP'  # Choose the model type
    print(f"Training {model_name} model with TF-IDF features...")
    model = train_classifier(model_name, X_train_tfidf, y_train)

    # Save the trained model
    model_path = "trained_author_model_tfidf.joblib"
    print(f"Saving the trained model to {model_path}...")
    dump(model, model_path)

    # Save the label encoder
    label_encoder_path = "label_encoder_tfidf.joblib"
    print(f"Saving the label encoder to {label_encoder_path}...")
    dump(label_encoder, label_encoder_path)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_tfidf, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")