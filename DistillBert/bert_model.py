import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def load_and_preprocess_data(file_path, num_samples=1000):
    # Load dataset
    df_train = pd.read_csv(file_path)
    df_train = df_train[:num_samples]

    # Extract features and labels
    X = df_train['review_text'].values
    y = df_train['class_index'].values

    # Adjust class indices to start from 0
    y_adjusted = y - 1

    return X, y_adjusted

def tokenize_text(texts, tokenizer):
    # Tokenization
    return [tokenizer(text, truncation=True, padding=True) for text in texts]

def preprocess_labels(labels, num_classes):
    # Convert adjusted labels to categorical format
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)

def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into train and validation sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def data_generator(X, y, batch_size, tokenizer):
    num_samples = len(X)
    num_batches = num_samples // batch_size
    while True:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            batch_X_encodings = [tokenizer(text, truncation=True, padding=True) for text in batch_X]
            max_length = max(len(encoding['input_ids']) for encoding in batch_X_encodings)
            X_input_ids = np.array([encoding['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(encoding['input_ids'])) for encoding in batch_X_encodings])
            X_attention_mask = np.array([encoding['attention_mask'] + [0] * (max_length - len(encoding['attention_mask'])) for encoding in batch_X_encodings])
            yield [X_input_ids, X_attention_mask], batch_y

def train_bert_model(file_path, num_samples=1000, batch_size=16, epochs=20):
    # Load and preprocess the dataset
    X, y_adjusted = load_and_preprocess_data(file_path, num_samples)

    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_encodings = tokenize_text(X, tokenizer)

    # Convert adjusted labels to categorical format
    num_classes = len(np.unique(y_adjusted))
    y_categorical = preprocess_labels(y_adjusted, num_classes)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = split_data(X_encodings, y_categorical)

    # Define DistilBERT configuration
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Determine the number of training steps per epoch
    num_train_steps = len(X_train) // batch_size

    # Determine the number of validation steps
    num_val_steps = len(X_val) // batch_size

    # Define the checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint filepath
    checkpoint_filepath = os.path.join(checkpoint_dir, "cp-{epoch:04d}.keras")
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Train the model and save weights after each epoch
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Train the model
        history = model.fit(
            data_generator(X_train, y_train, batch_size, tokenizer),
            steps_per_epoch=num_train_steps,
            validation_data=data_generator(X_val, y_val, batch_size, tokenizer),
            validation_steps=num_val_steps,
            epochs=1,
            verbose=1,
            callbacks=[model_checkpoint_callback]
        )

    return model

def evaluate_bert_model(model, X_val, y_val, batch_size=16):
    # Evaluate the model on the validation set
    num_val_steps = len(X_val) // batch_size
    y_pred_output = model.predict(data_generator(X_val, y_val, batch_size, tokenizer), steps=num_val_steps)

    # Extract predicted labels from the output
    y_pred = np.argmax(y_pred_output[0], axis=1)

    # Convert one-hot encoded labels back to original labels
    y_val_labels = np.argmax(y_val[:192], axis=1)

    # Calculate classification report
    classification_rep = classification_report(y_val_labels, y_pred)

    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_val_labels, y_pred)

    return classification_rep, conf_mat
