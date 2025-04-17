from flask import Flask, render_template, request, redirect
import json
import re
from collections import defaultdict, Counter

app = Flask(__name__)

# Load news data with content
with open("news.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)

# User interest profile: keyword → score
user_profile = Counter()

# Inverted index: word → set(article_ids)
inverted_index = defaultdict(set)

def build_inverted_index(articles):
    for article in articles:
        text = article['content'].lower()
        words = re.findall(r'\w+', text)
        for word in words:
            inverted_index[word].add(article['id'])
    print("\n Inverted Index (based on content):")
    for word, doc_ids in sorted(inverted_index.items()):
        print(f"{word}: {sorted(doc_ids)}")

def get_articles_by_interest():
    scores = defaultdict(int)

    for word, weight in user_profile.items():
        for article_id in inverted_index.get(word, []):
            scores[article_id] += weight

    # Rank articles by score (high to low)
    ranked = sorted(news_data, key=lambda a: scores[a['id']], reverse=True)
    return ranked

@app.route("/")
def index():
    # Show ranked articles based on current user profile
    ranked_articles = get_articles_by_interest()
    return render_template("index.html", news=ranked_articles)

@app.route("/feedback", methods=["POST"])
def feedback():
    news_id = int(request.form.get("news_id"))
    action = request.form.get("action")

    article = next((n for n in news_data if n["id"] == news_id), None)
    if article:
        content_words = re.findall(r'\w+', article['content'].lower())
        for word in content_words:
            if action == "like":
                user_profile[word] += 1
            elif action == "dislike":
                user_profile[word] -= 1

        print(f"\n[User Feedback] {action.upper()} → {article['title']}")
        print(f"[User Profile Snapshot] Top keywords:")
        for word, score in user_profile.most_common(10):
            print(f"  {word}: {score}")

    return redirect("/")

if __name__ == "__main__":
    build_inverted_index(news_data)
    app.run(debug=True)
