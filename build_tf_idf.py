from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from indexer import get_all_articles, extract_keywords


all_articles = get_all_articles(max_results=1000)
all_texts = [
    (a.title or "") + " " + (a.description or "") + " " + (a.content or "")
        for a in all_articles
]

vectorizer = TfidfVectorizer(tokenizer=extract_keywords)
vectorizer.fit(all_texts)


joblib.dump(vectorizer.idf_, "global_idf.pkl")
joblib.dump(vectorizer.vocabulary_, "global_vocab.pkl")

print("global_idf.pkl and global_vocab.pkl generated！")
