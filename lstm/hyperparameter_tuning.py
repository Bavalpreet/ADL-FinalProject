import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import re
import tqdm
import contractions
from bs4 import BeautifulSoup
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I | re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs

def load_and_preprocess_data(file_path, num_samples=None):
    df = pd.read_csv(file_path)
    if num_samples:
        df = df.sample(num_samples)
    X = df['review_text'].values
    y = df['class_index'].values - 1
    return X, y

def tokenize_text(texts, tokenizer):
    return [tokenizer.texts_to_sequences([text])[0] for text in texts]

def create_model(embedding_dim=100, lstm_units=128, dropout_rate=0.5, vocab_size=None, max_sequence_length=None):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_shape=(max_sequence_length,)))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_dim=100, lstm_units=128, dropout_rate=0.5, vocab_size=None, max_sequence_length=None):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.model = create_model(embedding_dim, lstm_units, dropout_rate, vocab_size, max_sequence_length)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=5, batch_size=64, verbose=0)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def get_params(self, deep=True):
        return {
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length
        }

# Read dataset from CSV files
train_df = pd.read_csv('yelp_train.csv')
test_df = pd.read_csv('yelp_test.csv')

# Preprocess the corpus
X_train = pre_process_corpus(train_df['review_text'].values)
X_test = pre_process_corpus(test_df['review_text'].values)

# Tokenize and pad sequences
tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
X_train = tokenize_text(X_train, tokenizer)
X_test = tokenize_text(X_test, tokenizer)
max_sequence_length = 800
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Encode labels
y_train = train_df['class_index'].values - 1
y_test = test_df['class_index'].values - 1

# Define the parameter grid for grid search
param_grid = {
    'embedding_dim': [100, 200, 300],
    'lstm_units': [64, 128, 256],
    'dropout_rate': [0.3, 0.5, 0.7]
}

# Perform grid search
keras_classifier = KerasClassifier(vocab_size=vocab_size, max_sequence_length=max_sequence_length)
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
