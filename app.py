import os
import logging
import pandas as pd
import yfinance as yf
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
from newsapi import NewsApiClient
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Caching config
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Login setup
login_manager = LoginManager()
login_manager.init_app(app)

# Sentiment models
finbert_sentiment = pipeline("text-classification", model="ProsusAI/finbert")
sia = SentimentIntensityAnalyzer()

# API Keys
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not NEWS_API_KEY or not HUGGINGFACE_API_TOKEN:
    logging.error("API keys missing. Please check your .env file.")
    exit(1)

# User class
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id, "example_user")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(1, username)
        login_user(user)
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Financial news route
@app.route('/get_news', methods=['GET'])
def get_news():
    news_data = fetch_financial_news()
    if news_data is None:
        return jsonify({"error": "No financial news available."}), 400

    analyzed_news = analyze_sentiment(news_data)
    analyzed_news = finbert_sentiment_analysis(analyzed_news)
    return jsonify(analyzed_news.to_dict(orient="records"))

# Ask-question route using Hugging Face
@app.route("/ask-question", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    response = process_question(query)
    return jsonify({"answer": response})

# Hugging Face inference logic
def process_question(query):
    try:
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": f"<|system|>You are a helpful financial assistant.<|user|>{query}<|assistant|>",
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            return generated_text.strip()
        else:
            print(f"❌ Hugging Face API Error: {response.status_code} - {response.text}")
            return "Sorry, Hugging Face API could not process the question."
    except Exception as e:
        print(f"❌ Error in process_question: {e}")
        return "Sorry, something went wrong while processing your question."


# Financial news fetch function
def fetch_financial_news(max_retries=3, backoff_factor=2):
    logging.info("📰 Fetching financial news...")
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    sources = ['bbc-news', 'cnn', 'business-insider', 'reuters', 'the-wall-street-journal']
    news_data = []

    for source in sources:
        for attempt in range(max_retries):
            try:
                articles = newsapi.get_top_headlines(sources=source, language="en")
                if articles and "articles" in articles:
                    news_data.extend([article["title"] for article in articles["articles"] if article["title"]])
                break
            except Exception as e:
                logging.warning(f"⚠️ Error fetching news from {source} (attempt {attempt + 1}): {e}")
                time.sleep(backoff_factor * (attempt + 1))
    return pd.DataFrame(news_data, columns=["Headline"]) if news_data else None

# Sentiment analysis functions
def analyze_sentiment(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for sentiment analysis.")
        return None
    logging.info("📈 Performing Vader sentiment analysis...")
    df["Vader Sentiment Score"] = df[column].apply(lambda text: sia.polarity_scores(str(text))["compound"])
    df["Vader Sentiment"] = df["Vader Sentiment Score"].apply(lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
    return df

def finbert_sentiment_analysis(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for FinBERT sentiment analysis.")
        return None
    logging.info("🤖 Performing FinBERT sentiment analysis...")
    df["FinBERT Sentiment"] = df[column].apply(lambda text: finbert_sentiment(str(text))[0]["label"])
    return df

if __name__ == "__main__":
    app.run(debug=True)
