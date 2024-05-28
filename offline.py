import nltk
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import sqlite3

# Download NLTK resources if necessary
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Database Setup
conn = sqlite3.connect('travel_knowledge.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge (
        question TEXT,
        category TEXT,
        answer TEXT
    )
''')

# 2. Load Data from JSON (or other sources)
with open("Travel.json", "r") as f:
    training_data = json.load(f)

# Insert data into the database
for item in training_data:
    cursor.execute('''
        INSERT INTO knowledge VALUES (?, ?, ?)
    ''', (item["question"], item["category"], item["answer"]))

conn.commit()  # Save changes to the database

# 3. Preprocess Data
def create_word_features(text):
    """Processes text into features."""
    words = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return dict([(word, True) for word in words])

# 4. Train the Model (Update this section)
def train_model():
    """Trains the model using data from the database."""
    global classifier  # Make 'classifier' accessible within this function
    # Get training data from the database
    cursor.execute("SELECT question, category FROM knowledge")
    training_data = [(create_word_features(row[0]), row[1]) for row in cursor]
    classifier = SklearnClassifier(MultinomialNB())  # Create a new classifier
    classifier.train(training_data)  # Train the model

# 5. User Interaction
def get_user_input():
    """Gets input from the user."""
    user_input = input("Ask me a travel question: ")
    return user_input

def classify_and_respond(user_input):
    """Classifies the question and provides a response."""
    features = create_word_features(user_input)
    category = classifier.classify(features)

    # Get the answer from the database based on the category
    cursor.execute("SELECT answer FROM knowledge WHERE category = ?", (category,))
    answer = cursor.fetchone()[0]

    print(f"Answer: {answer}")  # Just print the answer

# 6. Run the AI
# Initial training
train_model()

user_question = get_user_input()
classify_and_respond(user_question)  # Run once and exit

conn.close()  # Close the database connection