import requests
import feedparser
import schedule
import time
import os
from datetime import datetime
from elasticsearch_dsl import Q,Document, Date, Keyword, Text, Boolean, connections, Index
from collections import Counter
from datetime import datetime
import re
from newspaper import Article as NewsArticle  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


global_idf = None
global_vocab = None

idf_path = "global_idf.pkl"
vocab_path = "global_vocab.pkl"

if os.path.exists(idf_path) and os.path.exists(vocab_path):
    global_idf = joblib.load(idf_path)
    global_vocab = joblib.load(vocab_path)
    print("TF-IDF global TF-IDF loaded")
else:
    print("TF-IDF not detecting global_idf.pkl or global_vocab.pkl，")


def get_tfidf_matrix(texts):
    global global_idf, global_vocab

    if global_idf is None or global_vocab is None:
        if not os.path.exists(idf_path) or not os.path.exists(vocab_path):
            raise RuntimeError("global_idf.pkl or global_vocab.pkl doesn't exist")
        global_idf = joblib.load(idf_path)
        global_vocab = joblib.load(vocab_path)

    count_vectorizer = CountVectorizer(tokenizer=extract_keywords, vocabulary=global_vocab)
    tf_matrix = count_vectorizer.fit_transform(texts)

    transformer = TfidfTransformer()
    transformer.idf_ = global_idf 

    return transformer.transform(tf_matrix), count_vectorizer.get_feature_names_out()





def fetch_full_content(url):
    try:
        news = NewsArticle(url)
        news.download()
        news.parse()
        return news.text
    except Exception as e:
        print(f"Failed to fetch full content from {url}: {e}")
        return ""

connections.create_connection(
    hosts=["http://localhost:9200"],  # local Docker 
    timeout=20,
    verify_certs=False,
    ssl_show_warn=False
)


RSS_FEEDS = [
    # World
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Africa.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Americas.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/AsiaPacific.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Europe.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml",

    # U.S.
    "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Education.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Upshot.xml",

    # NY Region
    "https://rss.nytimes.com/services/xml/rss/nyt/NYRegion.xml",

    # Business
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/EnergyEnvironment.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/SmallBusiness.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Dealbook.xml",

    # Science
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Environment.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Space.xml",

    # Technology
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/PersonalTech.xml",

    # Health
    "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Well.xml",

    # Sports
    "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/CollegeBasketball.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/CollegeFootball.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Golf.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Hockey.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/ProBasketball.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/ProFootball.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Soccer.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Tennis.xml",

    # Arts
    "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/ArtandDesign.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/BookReview.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Dance.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Movies.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Music.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Television.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Theater.xml",

    # Style
    "https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/DiningandWine.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Love.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/TMagazine.xml",

    # Travel
    "https://rss.nytimes.com/services/xml/rss/nyt/Travel.xml",

    # Marketplace
    "https://rss.nytimes.com/services/xml/rss/nyt/Jobs.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/RealEstate.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Automobiles.xml",

    # Others
    "https://rss.nytimes.com/services/xml/rss/nyt/Opinion.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Food.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Magazine.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/SundayReview.xml"
]

# record and retrieve user likes/dislikes
class Feedback(Document):
    user_id = Keyword()       
    article_id = Keyword()    # article id of the feedback target
    liked = Boolean()         # True: like / False: dislike
    timestamp = Date()        # feedback date

    class Index:
        name = "feedbacks"

# indexing articles
class Article(Document):
    title = Text(analyzer="snowball")
    link = Keyword()
    guid = Keyword()
    description = Text(analyzer="snowball")
    creator = Text()
    pubDate = Date()
    category = Text(analyzer="standard", fields={"raw": Keyword()}, multi=True)
    content = Text(analyzer="snowball")  #  add article content

    class Index:
        name = "articles"

    def save(self, **kwargs):
        self.meta.id = self.guid
        return super().save(**kwargs)



def retrieve():
    print("[DEBUG] retrieve() called")

    for feed_url in RSS_FEEDS:
        try:
            print(f"[INFO] Fetching feed: {feed_url}")
            response = requests.get(feed_url)
            feed = feedparser.parse(response.content)
            # print(f"[INFO] Entries fetched: {len(feed.entries)}")

            for entry in feed.entries:
                try:
                    article_data = parse(entry)
                    index(article_data)
                except Exception as e:
                    print(f"[Error indexing entry] {e}")

        except Exception as e:
            print(f"[Error fetching feed] {feed_url} -> {e}")



def submit_feedback(user_id, article_id, liked=True):
    feedback = Feedback(
        user_id="user1",
        article_id=article_id,
        liked=liked,
        timestamp=datetime.now()
    )
    feedback.meta.id = f"{user_id}-{article_id}"
    feedback.save()
    print(f"[Feedback Saved] user={user_id}, article={article_id}, liked={liked}")




# get all articles
def get_all_articles(max_results=100):
    s = Article.search().sort("-pubDate")[:max_results]
    results = s.execute()

    print(f"Found {len(results)} articles:\n")
    # for hit in results:
    #    print(f"[{hit.pubDate}] {hit.title}\n id:{hit.meta.id}\n{hit.link}\n content:{hit.content} \n")
    return results 


# single word search
def search_articles_by_title(text):
    s = Article.search().query("match", title=text)
    return s

# search by category fuzzy
def search_by_category_fuzzy(keyword):
    q = Q("match", category={"query": keyword, "fuzziness": "AUTO"})
    s = Article.search().query(q).sort("-pubDate")[:50]
    return s


# search in all fields
def search_keywords(keywords, max_results=999, min_match_ratio="60%"):
    if isinstance(keywords, str):
        terms = [w for w in keywords.strip().split() if w]
    else:
        terms = keywords
    print("-------searching query is----- " + " ".join(terms))
    s = Article.search()

    if len(terms) == 1:
        q = Q(
            "multi_match",
            query=terms[0],
            fields=["title", "description", "creator", "category", "content"],
            type="most_fields",
            operator="or"
        )
        results = s.query(q)[:max_results].execute()  
    else:
        should_clauses = [
            Q("multi_match", query=kw, fields=["title", "description", "creator", "category", "content"], type="most_fields", operator="or")
            for kw in terms
        ]
        q = Q("bool", should=should_clauses, minimum_should_match=min_match_ratio)
        results = s.query(q)[:max_results].execute()  

    for hit in results[:10]:  
        print(f"- {hit.title} ({hit.pubDate})\n content_{hit.content} \n{hit.link}\n")

    return results





# STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS = {
    "the", "so","is", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
    "this", "that", "are", "as", "at", "by", "be", "from", "it", "was", "were",
    "has", "had", "but", "not", "have", "will", "would", "can", "could", "you",
    "your", "we", "they", "he", "she", "them", "his", "her", "its", "do", "does",
    "did", "what", "when", "where", "why", "how", "however","i", "me", "my", "mine", "our",
    "us", "their", "theirs", "who", "whom", "also", "if", "then", "than",
        "some", "many", "case", "thing", "called", "new", "old", "today", "said",
    "call", "calls", "use", "used", "one", "two", "three", "more", "other","no","ever"
}
STOP_WORDS.update({chr(c) for c in range(ord('a'), ord('z') + 1)})



def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    filtered = []
    for w in words:
        if w in STOP_WORDS:
            continue
        if w.isdigit():
            year = int(w)
            if 1900 <= year <= 2030:  # keep year number
                filtered.append(w)
        else:
            filtered.append(w)
    return filtered

def get_feedbacks_with_weights(user_id, max_days=30):
    feedbacks = Feedback.search() \
        .filter("term", user_id=user_id) \
        .sort("-timestamp")[:500] \
        .execute()

    weighted_feedbacks = []

    for fb in feedbacks:
        age_days = (datetime.now() - fb.timestamp).days
        if age_days > max_days:
            continue

        base_weight = round(1.0 * (0.9 ** age_days), 3)

        # for dislikes, give negative weights
        if not fb.liked:
            base_weight *= -1 

        weighted_feedbacks.append((fb.article_id, base_weight))

    return weighted_feedbacks


def build_weighted_keywords(user_id):
    weighted_feedbacks = get_feedbacks_with_weights(user_id)
    keyword_score = Counter()
    seen_article_ids = []

    for article_id, weight in weighted_feedbacks:
        try:
            article = Article.get(id=article_id)
            seen_article_ids.append(article.meta.id)

            text = (article.title or "") + " " + (article.description or "")
            for word in extract_keywords(text):
                keyword_score[word] += weight
        except:
            continue

    return keyword_score, seen_article_ids



def recommend_articles(user_id, num_results=15):
    weighted_feedbacks = get_feedbacks_with_weights(user_id)
    if not weighted_feedbacks:
        print("No sufficient feedback found.")
        return []

    # collect user feedbacks
    article_texts = [] #all
    feedback_weights = []
    seen_ids = []

    for article_id, weight in weighted_feedbacks:
        try:
            article = Article.get(id=article_id)
            text = (article.title or "") + " " + (article.description or "")+ (article.content or "")
            article_texts.append(text)
            feedback_weights.append(weight)
            seen_ids.append(article.meta.id)
        except:
            continue

    if not article_texts:
        print("No articles found for feedbacks.")
        return []

    transformer = TfidfTransformer()
    transformer.idf_ = global_idf

    tfidf_matrix, feature_names = get_tfidf_matrix(article_texts)


    keyword_score = Counter()
    for i in range(tfidf_matrix.shape[0]):
        tfidf_vec = tfidf_matrix[i].toarray().flatten()
        for j, tfidf_val in enumerate(tfidf_vec):
            if tfidf_val > 0:
                keyword_score[feature_names[j]] += tfidf_val * feedback_weights[i]

    if not keyword_score:
        print("No keyword scores available.")
        return []

    positive = [(kw, score) for kw, score in keyword_score.items() if score > 0]
    positive.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [kw for kw, _ in positive[:10]]
    should_clauses = [
    Q("match", title=kw) |    
    Q("match", description=kw) |  
    Q("match", content=kw)        
    for kw in top_keywords
]
    q = Q("bool", should=should_clauses, minimum_should_match=1)

    s = Article.search().query(q).exclude("ids", values=seen_ids).sort("-pubDate")
    results = list(s.scan())

    # giving scores
    def weighted_score(article):
        text = (article.title or "") + " " + (article.description or "")
        words = extract_keywords(text)
        return sum(keyword_score[w] for w in top_keywords if w in words)

    sorted_results = sorted(results, key=weighted_score, reverse=True)
    if num_results is not None:
        sorted_results = sorted_results[:num_results]

    print(f"\n[TF-IDF (global IDF) Recommendation] user: {user_id}")
    print(f"Top keywords: {top_keywords}\n")
    for hit in sorted_results[:10]:
        print(f"- {hit.title} ({hit.pubDate})\n  Score: {weighted_score(hit)}\n  {hit.link}\n")

    return sorted_results



def parse(entry):
    link = entry["link"]
    return {
        "title": entry["title"],
        "link": link,
        "guid": entry["guid"],
        "description": entry.get("description", ""),
        "creator": entry.get("author", ""),
        "pubDate": datetime(*entry["published_parsed"][:6]),
        "category": [cat["term"] for cat in entry.get("tags", [])],
        "content": fetch_full_content(link)
    }


def index(article_data):
    Article(**article_data).save()





def search_by_content(text, max_results=999):

    s = Article.search()
    q = Q(
        "multi_match",
        query=text,
        fields=["content"],
    )

    results = s.query(q).sort("-pubDate")[:max_results].execute()
    print(f"\n[Search by Content] found number  '{len(results)}'\n")
    for hit in results[:10]:
        print(f"- {hit.title} ({hit.pubDate})\n  {hit.link}\n")

    return list(results)


def query_articles(field, text, max_results=999, sort_by="relevance"):
    terms = [w for w in text.strip().split() if w]


    if field == "keyword":
        results = search_keywords(text, max_results=max_results, min_match_ratio="60%")
        return sort_results(results, sort_by)


    if field == "title":
        s = search_articles_by_title(text)
    elif field == "category":
        s = search_by_category_fuzzy(text)
    else:
        raise ValueError(f"Unknown field: {field}")

    if sort_by == "time":
        s = s.sort("-pubDate")

    return s.execute()

def sort_results(results, sort_by):
    if sort_by == "time":
        return sorted(results, key=lambda a: a.pubDate, reverse=True)
    return results









if __name__ == "__main__":
 print("Starting indexer...")




 Index("articles").delete(ignore=404)
 Article.init()
 retrieve()
 print("Success! Retrieving new articles once every 5 minutes.")
 schedule.every(5).minutes.do(retrieve)
#  print("Success! Retrieving new articles once every 5 minutes.")
 
 try:
     while True:
         schedule.run_pending()
         time.sleep(1)
 except KeyboardInterrupt:
     print("Exiting...")
