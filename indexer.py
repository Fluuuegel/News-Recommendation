import requests
import feedparser
import schedule
import time
import os
from datetime import datetime
from elasticsearch_dsl import Document, Date, Keyword, Text, Boolean, connections

# Retrieves environment variables ES_HOST, ES_USER, and ES_PASS
# On Windows you can set them with the syntax "$env:ES_USER=user"
# connections.create_connection(
#     hosts=os.environ.get("ES_HOST", "https://localhost:9200"),
#     basic_auth=(os.environ["ES_USER"], os.environ["ES_PASS"]),
#     verify_certs=False,
#     ssl_show_warn=False, 
#     timeout=20
# )

connections.create_connection(
    hosts=["http://localhost:9200"],  # local Docker 
    timeout=20,
    verify_certs=False,
    ssl_show_warn=False
)



# record and retrieve user likes/dislikes
class Feedback(Document):
    user_id = Keyword()       # whose feedback is this
    article_id = Keyword()    # article id of the feedback target
    liked = Boolean()         # True: like / False: dislike
    timestamp = Date()        # feedback date

    class Index:
        name = "feedbacks"

def submit_feedback(user_id, article_id, liked=True):
    feedback = Feedback(
        user_id="user1",
        article_id=article_id,
        liked=liked,
        timestamp=datetime.now()
    )
    # make sure there's only 1 feedback on one article at most
    feedback.meta.id = f"{user_id}-{article_id}"
    feedback.save()
    # print(f"[Feedback Saved] user={user_id}, article={article_id}, liked={liked}")
    return feedback

class Article(Document):
    title = Text(analyzer="snowball")
    link = Keyword()
    guid = Keyword()
    description = Text(analyzer="snowball")
    creator = Text()
    pubDate = Date()
    category = Keyword()

    class Index:
        name = "articles"

    def save(self, **kwargs):
        # Set permalink as ID to avoid fetching duplicate articles
        self.meta.id = self.guid
        return super().save(**kwargs)

# get all articles
# import json

# def get_all_articles(max_results=100):
#     s = Article.search().sort("-pubDate")[:max_results]
#     results = s.execute()

#     print(f"📰 Found {len(results)} articles:\n")

#     for i, hit in enumerate(results, start=1):
#         article_data = {
#             "id": hit.meta.id,
#             "title": hit.title,
#             "link": hit.link,
#             "guid": hit.guid,
#             "description": hit.description,
#             "creator": hit.creator,
#             "pubDate": str(hit.pubDate),
#             "category": list(hit.category) if hit.category else None,
#         }

#         # Pretty print each article as JSON
#         print(f"Article {i}:\n{json.dumps(article_data, indent=2, ensure_ascii=False)}\n")

#     return results



# single word search
def search_articles_by_title(keyword):
    q = Q("match", title=keyword)
    s = Article.search().query(q)
    results = s.execute()

    print(f"Found {len(results)} results for keyword: '{keyword}'\n")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title}\n{hit.link}\n")

# search by categories
def multi_field_search(keyword):
    q = Q("multi_match", query=keyword, fields=["title", "description", "creator"])
    s = Article.search().query(q)
    results = s.execute()

    print(f"Found {len(results)} results for keyword: '{keyword}'\n")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title}\n{hit.link}\n")

# phrase search
def search_by_category(category_value):
    q = Q("term", category=category_value)
    s = Article.search().query(q)
    results = s.execute()

    print(f"Found {len(results)} articles with category = '{category_value}'\n")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title} - {hit.category}\n")



# a combination of many filters: set author/title/..
def search_articles_bool(keyword=None, author=None, category=None, max_results=10):
    must_clauses = []
    filter_clauses = []

    # keyword title / description / creator（使用 multi_match）
    if keyword:
        must_clauses.append(
            Q("multi_match", query=keyword, fields=["title", "description", "creator"])
        )

   
    if author:
        filter_clauses.append(Q("term", creator=author))

    
    if category:
        filter_clauses.append(Q("term", category=category))

    
    q = Q("bool", must=must_clauses, filter=filter_clauses)

    s = Article.search().query(q).sort("-pubDate")[:max_results]
    results = s.execute()

    print(f"Found {len(results)} results with keyword='{keyword}', author='{author}', category='{category}'\n")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title} — {hit.creator} / {hit.category}\n{hit.link}\n")

    return results



# reccomendation search 
def recommend_articles(user_id, num_results=5):
    # get articles user likes
    liked_articles = Feedback.search().filter("term", user_id=user_id).filter("term", liked=True).execute()
    if not liked_articles:
        print("No feedback found.")
        return []

    liked_ids = [f.article_id for f in liked_articles]
    base_articles = [Article.get(id=aid) for aid in liked_ids[:5]]  # 取前3篇文章

    # more like this
    like_clauses = [{"_index": "articles", "_id": a.meta.id} for a in base_articles]

    q = Q("more_like_this", fields=["title", "description"], like=like_clauses, min_term_freq=1, min_doc_freq=1)

    # exclude articles user has seen
    seen_ids = [f.article_id for f in Feedback.search().filter("term", user_id=user_id).execute()]
    s = Article.search().query(q).exclude("ids", values=seen_ids)[:num_results]

    results = s.execute()
    print(f"\n[Recommended for user {user_id}]")
    for hit in results:
        print(f"{hit.title}\n{hit.link}\n")
    return results



# gradually adjusts selection recommendations
from elasticsearch_dsl import Q
from collections import Counter
from datetime import datetime
import re

# STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS = {
    "the", "so","is", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
    "this", "that", "are", "as", "at", "by", "be", "from", "it", "was", "were",
    "has", "had", "but", "not", "have", "will", "would", "can", "could", "you",
    "your", "we", "they", "he", "she", "them", "his", "her", "its", "do", "does",
    "did", "what", "when", "where", "why", "how", "however","i", "me", "my", "mine", "our",
    "us", "their", "theirs", "who", "whom", "also", "if", "then", "than",
        "some", "many", "case", "thing", "called", "new", "old", "today", "said",
    "call", "calls", "use", "used", "one", "two", "three", "more", "other"
}



def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in STOP_WORDS]

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

def recommend_articles(user_id, top_n_keywords=10, num_results=5):
    keyword_score, seen_ids = build_weighted_keywords(user_id)
    if not keyword_score:
        print("No sufficient feedback found.")
        return []

    top_keywords = [kw for kw, _ in keyword_score.most_common(top_n_keywords)]
    should_clauses = [Q("match", title=kw) | Q("match", description=kw) for kw in top_keywords]

    q = Q("bool", should=should_clauses, minimum_should_match=2)

    s = Article.search().query(q).exclude("ids", values=seen_ids).sort("-pubDate")[:num_results]
    results = s.execute()

    print(f"\n recommendation result for user: {user_id}")
    print(f"Top keywords: {top_keywords}\n")
    for hit in results:
        print(f"- {hit.title} ({hit.pubDate})\n  {hit.link}\n")

    return results



def parse(entry):
    return {
        "title": entry["title"],
        "link": entry["link"],
        "guid": entry["guid"],
        "description": entry["description"],
        "creator": entry["author"],

        # Feedparser returns the publication date as a 9-tuple, parse as datetime string
        "pubDate": datetime(*entry["published_parsed"][:6]),

        # Feedparser groups categories into tags, retrieve their string representation
        "category": [cat["term"] for cat in entry["tags"]]
    }


def index_article(article_data):
    Article(**article_data).save()


RSS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml"
]

def retrieve():
    print("[DEBUG] retrieve() called")

    for feed_url in RSS_FEEDS:
        try:
            print(f"[INFO] Fetching feed: {feed_url}")
            response = requests.get(feed_url)
            feed = feedparser.parse(response.content)
            print(f"[INFO] Entries fetched: {len(feed.entries)}")

            for entry in feed.entries:
                try:
                    article_data = parse(entry)
                    index_article(article_data)
                except Exception as e:
                    print(f"[Error indexing entry] {e}")

        except Exception as e:
            print(f"[Error fetching feed] {feed_url} -> {e}")


if __name__ == "__main__":
 print("Starting indexer...")
 Article.init()
 retrieve()
 schedule.every(5).minutes.do(retrieve)
 print("Success! Retrieving new articles once every 5 minutes.")
 
 try:
     while True:
         schedule.run_pending()
         time.sleep(1)
 except KeyboardInterrupt:
     print("Exiting...")

def query_articles(field, text, max_results=30):
    """
    field: "keyword"/"title"/"category"
    text: 查询字符串
    """
    if field == "title":
        s = Article.search().query("match", title=text)
    elif field == "category":
        # term 查询 category 列表中的 exact match
        s = Article.search().query("term", category=text)
    else:  # keyword
        s = Article.search().query(
            "multi_match",
            query=text,
            fields=["title", "description", "creator"]
        )

    s = s.sort("-pubDate")[:max_results]
    return s.execute()