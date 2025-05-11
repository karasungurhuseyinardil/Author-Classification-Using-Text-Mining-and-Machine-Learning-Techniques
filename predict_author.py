import torch
from joblib import load
from transformers import BertTokenizer, BertModel

# Load the trained model and label encoder
model_path = "trained_author_model.joblib"
label_encoder_path = "label_encoder.joblib"
print("Loading the trained model and label encoder...")
model = load(model_path)
label_encoder = load(label_encoder_path)

# Load BERT tokenizer and model
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

# Function to extract BERT embeddings for a single text
def extract_single_bert_embedding(text, tokenizer, model, max_length=512, device=None):
    model.eval()
    encodings = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()

# Function to predict the author of a given text
def predict_author(text):
    print("Extracting BERT embedding for the input text...")
    embedding = extract_single_bert_embedding(text, tokenizer, bert_model, device=device)
    print("Predicting the author...")
    label_index = model.predict(embedding)
    author = label_encoder.inverse_transform(label_index)
    return author[0]

# Example usage
if __name__ == "__main__":
    input_text = """Your input text here."""
    predicted_author = predict_author(input_text)
    print(f"The predicted author is: {predicted_author}")