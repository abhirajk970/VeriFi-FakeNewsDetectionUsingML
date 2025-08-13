import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
train_data = pd.read_csv('train.csv').fillna(' ')
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')


# Add labels to additional datasets
fake_news['label'] = 1
real_news['label'] = 0

# Process data
train_data['head'] = train_data['author'] + ' ' + train_data['title']
train_data = train_data[['head', 'label']]
fake_news['head'] = fake_news['title']
real_news['head'] = real_news['title']
fake_news = fake_news[['head', 'label']]
real_news = real_news[['head', 'label']]

# Combine all datasets
combined_data = pd.concat([train_data, fake_news, real_news], ignore_index=True)

# Initialize PorterStemmer
ps = PorterStemmer()

# Define a stemming function
def stemming(head):
    stemmed_head = re.sub('[^a-zA-Z]', ' ', head)
    stemmed_head = stemmed_head.lower()
    stemmed_head = stemmed_head.split()
    stemmed_head = [ps.stem(word) for word in stemmed_head if word not in stopwords.words('english')]
    return ' '.join(stemmed_head)

# Apply stemming
combined_data['head'] = combined_data['head'].apply(stemming)

# Vectorize the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_data['head'].values)
y = combined_data['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model using GridSearchCV
model = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Save the best model and vectorizer
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved!")
