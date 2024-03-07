from transformers import pipeline
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('spam.csv')

# Remove leading/trailing whitespaces in column names
data.columns = data.columns.str.strip()

# Drop rows with missing labels or SMS
data.dropna(subset=['label', 'sms'], inplace=True)

# Preprocess data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['sms'] = data['sms'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Convert labels to binary (0 for ham, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['sms'], data['label'], test_size=0.2, random_state=42)

# Example text
new_messages = ["Congratulations! You've won a free vacation.", 
                "Hey, what's up?"]

# Initialize pre-trained text classification model
classifier = pipeline("text-classification")

# Perform text classification for new messages
predictions = classifier(new_messages)

# Print predictions
for message, prediction in zip(new_messages, predictions):
    label = "Spam" if prediction['label'] == 'LABEL_1' else "Ham"
    print(message, ":", label)
