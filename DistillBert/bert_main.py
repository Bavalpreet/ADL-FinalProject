from bert_model import load_and_preprocess_data, tokenize_text, preprocess_labels, split_data, train_model, evaluate_model
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load and preprocess the dataset
X_train, y_train = load_and_preprocess_data('yelp_train.csv')
X_test, y_test = load_and_preprocess_data('yelp_test.csv')

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
X_train_encodings = tokenize_text(X_train, tokenizer)
X_test_encodings = tokenize_text(X_test, tokenizer)

# Convert adjusted labels to categorical format
num_classes = len(np.unique(y_train))
y_train_categorical = preprocess_labels(y_train, num_classes)
y_test_categorical = preprocess_labels(y_test, num_classes)

# Split data into train, validation, and test sets
X_train, X_val, y_train, y_val = split_data(X_train_encodings, y_train_categorical)
X_test, y_test = X_test_encodings, y_test_categorical

# Define DistilBERT configuration
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define batch size
batch_size = 16

# Train the model
train_model(model, X_train, y_train, X_val, y_val, batch_size, tokenizer, num_epochs=20)

# Evaluate the model
classification_rep, conf_mat = evaluate_model(model, X_test, y_test, batch_size, tokenizer)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_mat)
