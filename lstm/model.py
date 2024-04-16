import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from bs4 import BeautifulSoup
import unicodedata
import re
import contractions
from tqdm import tqdm
import os
import pandas as pd



def load_dataset(train_path, test_path):
    """
    Load train and test datasets from CSV files.

    Args:
    train_path (str): File path of the training dataset.
    test_path (str): File path of the test dataset.

    Returns:
    pd.DataFrame, pd.DataFrame: Loaded train and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_text(text):
    """
    Preprocess text data by removing HTML tags, accents, and special characters.

    Args:
    text (str): Input text data.

    Returns:
    str: Preprocessed text.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Fix contractions
    text = contractions.fix(text)
    return text

def preprocess_corpus(corpus):
    """
    Preprocess a corpus of text data.

    Args:
    corpus (list): List of text data.

    Returns:
    list: Preprocessed text corpus.
    """
    preprocessed_corpus = [preprocess_text(text) for text in tqdm(corpus)]
    return preprocessed_corpus

def tokenize_text(texts, tokenizer):
    """
    Tokenize text data using a tokenizer.

    Args:
    texts (list): List of text data.
    tokenizer: Tokenizer object.

    Returns:
    np.ndarray: Tokenized sequences.
    """
    return tokenizer.texts_to_sequences(texts)

def pad_sequences(sequences, maxlen):
    """
    Pad sequences to a maximum length.

    Args:
    sequences (np.ndarray): Sequences to pad.
    maxlen (int): Maximum length to pad sequences to.

    Returns:
    np.ndarray: Padded sequences.
    """
    return sequence.pad_sequences(sequences, maxlen=maxlen)

def encode_labels(labels):
    """
    Encode labels using LabelEncoder.

    Args:
    labels (list): List of labels.

    Returns:
    np.ndarray: Encoded labels.
    """
    le = LabelEncoder()
    return le.fit_transform(labels)

def build_lstm_model(vocab_size, embedding_dim, max_sequence_length):
    """
    Build an LSTM model.

    Args:
    vocab_size (int): Vocabulary size.
    embedding_dim (int): Dimension of word embeddings.
    max_sequence_length (int): Maximum sequence length.

    Returns:
    tf.keras.Sequential: LSTM model.
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_shape=(max_sequence_length,)))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=30):
    """
    Train the LSTM model.

    Args:
    model: LSTM model.
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation labels.
    batch_size (int): Batch size for training.
    epochs (int): Number of epochs for training.

    Returns:
    tf.keras.History: Training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the LSTM model.

    Args:
    model: LSTM model.
    X_test (np.ndarray): Test features.
    y_test (np.ndarray): Test labels.

    Returns:
    tuple: Tuple containing accuracy, precision, recall, F1-score, and confusion matrix.
    """
    scores = model.evaluate(X_test, y_test, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = scores[1]
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    return accuracy, precision, recall, f1, conf_matrix

def plot_confusion_matrix(conf_matrix):
    """
    Plot the confusion matrix.

    Args:
    conf_matrix (np.ndarray): Confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
