from model import build_lstm_model, train_model, evaluate_model, plot_confusion_matrix, load_dataset, preprocess_corpus, tokenize_text, pad_sequences, encode_labels
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

# Load datasets
train_df, test_df = load_dataset('yelp_train.csv', 'yelp_test.csv')

# Preprocess text data
X_train = preprocess_corpus(train_df['review_text'].values)
X_test = preprocess_corpus(test_df['review_text'].values)

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
y_train = encode_labels(train_df['class_index'].values)
y_test = encode_labels(test_df['class_index'].values)

# Build LSTM model
embedding_dim = 800
model = build_lstm_model(vocab_size, embedding_dim, max_sequence_length)

# Train the model
history = train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=30)

# Evaluate the model
accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Plot confusion matrix
plot_confusion_matrix(conf_matrix)
