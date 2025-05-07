from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from indexer import get_all_articles, extract_keywords


all_articles = get_all_articles(max_results=1000)
all_texts = [
    (a.title or "") + " " + (a.description or "") for a in all_articles
]

vectorizer = TfidfVectorizer(tokenizer=extract_keywords)
vectorizer.fit(all_texts)


joblib.dump(vectorizer.idf_, "idf.pkl")
joblib.dump(vectorizer.vocabulary_, "vocab.pkl")

print("idf.pkl 和vocab.pkl 已生成！")
