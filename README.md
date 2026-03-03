# NLP Sentiment Analysis

A hybrid deep learning project for multi-class sentiment and emotion classification on short text data (tweets).

## Overview

This project implements a hybrid architecture that combines:

- TextCNN for contextual feature extraction  
- SentiWordNet-based lexical sentiment features  

The objective is to improve classification performance by leveraging both deep semantic learning and traditional sentiment scoring techniques.

The model performs multi-class classification (6 emotion categories).

---

## Architecture

The system consists of two parallel branches:

1. TextCNN Branch
   - Embedding layer (20,000 vocabulary size, 150 dimensions)
   - 1D Convolutions with kernel sizes 3, 4, 5
   - Global Max Pooling
   - Feature concatenation

2. Lexicon Feature Branch
   - Sentiment features extracted using:
     - Tokenization
     - POS tagging
     - WordNet mapping
     - SentiWordNet scoring
   - Aggregated positive, negative, and objective scores
   - Dense processing layer

Both branches are concatenated and passed through fully connected layers with a Softmax output (6 classes).

---

## Text Preprocessing

- Lowercasing
- Emoji conversion to text
- URL removal
- Mention removal
- Hashtag cleaning
- Whitespace normalization

---

## Loss Function

Focal Loss is used to address potential class imbalance:

focal_loss(gamma=2.0, alpha=0.25)

---

## Training Configuration

- Epochs: 12
- Batch size: 32
- Validation split: 10%
- Early stopping
- Learning rate reduction on plateau

---

## Tech Stack

- Python
- TensorFlow / Keras
- NLTK
- SentiWordNet
- Scikit-learn
- NumPy
- Pandas

---
## Project Structure

NLP-Sentiment-Analysis/
│
├── data/
│   └── train.csv                  # Dataset file
│
├── notebooks/
│   └── draft-1.ipynb              # Main training & experimentation notebook
│
├── src/
│   ├── preprocessing.py           # Text cleaning and preprocessing functions
│   ├── sentiment_features.py      # SentiWordNet feature extraction
│   ├── model.py                   # TextCNN + hybrid model architecture
│   └── loss.py                    # Focal loss implementation
│
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Ignored files and folders

## Installation

Install dependencies:

pip install nltk numpy pandas scikit-learn tensorflow emoji

Download required NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')

---

## How to Run

1. Clone the repository  
   git clone https://github.com/your-username/NLP-Sentiment-Analysis.git

2. Open the notebook  
   jupyter notebook draft-1.ipynb

3. Update dataset path if necessary  
4. Run all cells  

---

## Future Improvements

- Use pre-trained embeddings (GloVe, FastText)
- Replace CNN with BiLSTM or Transformer
- Add attention mechanism
- Hyperparameter tuning
- Deploy as a REST API

---

.
