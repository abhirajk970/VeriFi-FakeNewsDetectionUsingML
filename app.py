from flask import Flask, render_template, request
import requests
import joblib

# Load saved model and vectorizer
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Flask app setup
app = Flask(__name__)

# API keys and base URLs
NEWS_API_KEY = '46ea4ef988f54352b3f42e4011dfe861'
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc?query="

def fetch_related_articles(query):
    # Limit the query to a few words for better results
    query = ' '.join(query.split()[:5])

    # Fetch articles from NewsAPI
    newsapi_url = f"https://newsapi.org/v2/everything?q={query}&apiKey={'46ea4ef988f54352b3f42e4011dfe861'}"
    newsapi_articles = []
    try:
        response = requests.get(newsapi_url)
        if response.status_code == 200:
            data = response.json()
            newsapi_articles = [
                {'title': article['title'], 'url': article['url'], 'source': 'NewsAPI'}
                for article in data.get('articles', [])
            ]
    except Exception as e:
        print(f"Error with NewsAPI: {e}")

    # Fetch articles from GDELT API
    gdelt_url = f"{GDELT_BASE_URL}{query}&mode=artlist&maxrecords=5&format=json"
    gdelt_articles = []
    try:
        response = requests.get(gdelt_url)
        if response.status_code == 200:
            data = response.json()
            gdelt_articles = [
                {'title': article['title'], 'url': article['url'], 'source': 'GDELT'}
                for article in data.get('articles', [])
            ]
    except Exception as e:
        print(f"Error with GDELT API: {e}")

    # Combine and deduplicate articles
    all_articles = newsapi_articles + gdelt_articles
    # Deduplicate based on title
    unique_articles = {article['title']: article for article in all_articles}.values()

    return list(unique_articles)[:5]


# Define the prediction function
def prediction(input_text):
    input_data = vectorizer.transform([input_text])
    prediction = best_model.predict(input_data)
    return prediction[0]

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form.get('news', '').strip()
        if not input_text:
            return render_template('index.html', prediction_text="Please enter a news headline.")

        # Get the prediction
        pred = prediction(input_text)
        result = 'The News is Fake' if pred == 1 else 'The News is Real'

        # Fetch related articles
        related_articles = fetch_related_articles(input_text)

        return render_template(
            'index.html',
            prediction_text=result,
            articles=related_articles
        )

if __name__ == '__main__':
    app.run(debug=True)
