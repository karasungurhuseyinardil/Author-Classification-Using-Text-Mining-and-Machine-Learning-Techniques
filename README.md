# Author Classification Using Text Mining and Machine Learning Techniques

This project is developed as part of the COME 448 Data Mining and Knowledge Discovery course. The goal of this project is to classify documents based on their authorship using different feature extraction techniques and machine learning models. Given a dataset containing documents from multiple authors, students will analyze and compare various methods to achieve the best classification performance.

## Project Structure

```
author_classification.py
predict_author.py
dataset_authorship/
    AAltan/
        2012_09_11.txt
        2012_09_14.txt
        ...
    AAydintasbas/
        ...
    ...
```

- **`author_classification.py`**: The main Python script that handles data loading, feature extraction, model training, and evaluation.
- **`predict_author.py`**: A script to predict the author of a given text using the trained model.
- **`dataset_authorship/`**: A folder containing text files organized by author names. Each subfolder corresponds to an author, and the text files contain their writings.

## Features

1. **Data Loading**: Texts and their corresponding author labels are loaded from the `dataset_authorship` folder.
2. **Feature Extraction**: BERT embeddings are extracted for each text using the `transformers` library.
3. **Model Training**: Various machine learning models are supported, including:
   - Support Vector Machines (SVM)
   - Random Forest
   - Naive Bayes
   - Decision Tree
   - XGBoost
   - Multi-Layer Perceptron (MLP)
4. **Model Evaluation**: The trained models are evaluated using metrics such as accuracy, precision, recall, and F1-score.
5. **Save Model**: The trained model and label encoder are saved for future use.
6. **Predict Author**: A script to predict the author of a given text using the saved model.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `numpy`
  - `scikit-learn`
  - `transformers`
  - `torch`
  - `xgboost`
  - `joblib`

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train and Save the Model
1. Place the dataset in the `dataset_authorship` folder, ensuring that each author's texts are in separate subfolders.
2. Run the `author_classification.py` script:

```bash
python author_classification.py
```

3. The script will:
   - Load the dataset.
   - Extract BERT embeddings for the texts.
   - Train the selected machine learning model (default: XGBoost).
   - Save the trained model to `trained_author_model.joblib`.
   - Save the label encoder to `label_encoder.joblib`.

### 2. Predict Author
1. Run the `predict_author.py` script:

```bash
python predict_author.py
```

2. Modify the `input_text` variable in the script to include the text you want to classify:

```python
input_text = """Your input text here."""
```

3. The script will load the saved model and label encoder, extract BERT embeddings for the input text, and predict the author. The result will be displayed in the console:

```
The predicted author is: [Author Name]
```

## Results

The script outputs the following evaluation metrics for the selected model:
- Accuracy
- Precision
- Recall
- F1-Score

## Customization

- To change the machine learning model, modify the `model_name` variable in the `author_classification.py` script. Supported values are:
  - `'SVM'`
  - `'RandomForest'`
  - `'NaiveBayes'`
  - `'DecisionTree'`
  - `'XGBoost'`
  - `'MLP'`

- You can also adjust parameters such as the BERT model, batch size, and train-test split ratio in the `main()` function.

## Acknowledgments

This project uses the following libraries and tools:
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
