import pandas as pd
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# download NLTK stopwords (only first time)
nltk.download('stopwords')

# 🧾 Step 1: Create a small sample dataset
data = {
    'text': [
        "The government announces new education policy for 2025.",
        "NASA confirms water found on Mars surface.",
        "Chocolate cures all diseases, scientists prove it!",
        "Celebrity claims the earth is flat and surrounded by walls.",
        "Apple launches new iPhone with advanced AI camera.",
        "Breaking: drinking coffee makes you invisible overnight!",
        "The Prime Minister inaugurated a new highway project today.",
        "Aliens have landed in the city park according to social media posts!"
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0]   # 1 = True (Real), 0 = Fake
}

df = pd.DataFrame(data)

# 🧹 Step 2: Clean and preprocess text
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(preprocess)

# 🧠 Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# 🧩 Step 4: Vectorize text
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🤖 Step 5: Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 🧪 Step 6: Evaluate
y_pred = model.predict(X_test_tfidf)
print("Model Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

# 📰 Step 7: Enter your own news to test
print("\n--- FAKE NEWS DETECTION ---")
user_input = input("Enter a news headline or text: ")

# preprocess and predict
processed = preprocess(user_input)
vectorized = vectorizer.transform([processed])
prediction = model.predict(vectorized)[0]

# 🎯 Output Result
if prediction == 1:
    print("\n🟢 This news is TRUE / REAL ✅")
else:
    print("\n🔴 This news is FAKE ❌")


