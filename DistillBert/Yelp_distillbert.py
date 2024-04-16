import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_preprocess_data(file_path):
    # Load dataset
    df_train = pd.read_csv(file_path)
    # Extract features and labels
    X = df_train['review_text'].values
    y = df_train['class_index'].values

    # Adjust class indices to start from 0
    y_adjusted = y - 1

    return X, y_adjusted

def tokenize_text(texts, tokenizer):
    # Tokenization
    return [tokenizer(text, truncation=True, padding=True) for text in tqdm(texts)]

def preprocess_labels(labels, num_classes):
    # Convert adjusted labels to categorical format
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)

def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into train and validation sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def define_model(num_classes):
    # Define DistilBERT configuration
    config = DistilBertConfig()

    # Define the model architecture with the correct number of output units
    return TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

def compile_model(model):
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

def pad_sequences(X_encodings):
    # Determine the maximum sequence length
    max_length = max(len(encoding['input_ids']) for encoding in X_encodings)

    # Pad sequences to the maximum length
    input_ids = np.array([encoding['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(encoding['input_ids'])) for encoding in X_encodings])
    attention_mask = np.array([encoding['attention_mask'] + [0] * (max_length - len(encoding['attention_mask'])) for encoding in X_encodings])

    return input_ids, attention_mask

def train_model(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32):
    # Train the model and track the training history
    return model.fit([X_train[0], X_train[1]], y_train,
                     validation_data=([X_val[0], X_val[1]], y_val),
                     epochs=epochs, batch_size=batch_size)

# Load and preprocess the dataset
X, y_adjusted = load_and_preprocess_data('/kaggle/input/yelpreview/yelp_train.csv')

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
X_encodings = tokenize_text(X, tokenizer)

# Convert adjusted labels to categorical format
num_classes = len(np.unique(y_adjusted))
y_categorical = preprocess_labels(y_adjusted, num_classes)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = split_data(X_encodings, y_categorical)

# Define the model architecture
model = define_model(num_classes)

# Compile the model
compile_model(model)

# Pad sequences
X_train_padded = pad_sequences(X_train)
X_val_padded = pad_sequences(X_val)

# Train the model
history = train_model(model, X_train_padded, y_train, X_val_padded, y_val, epochs=20, batch_size=16)
