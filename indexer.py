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
    user_id = Keyword()       # whose feedback is this
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
            print(f"[INFO] Entries fetched: {len(feed.entries)}")

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
    # make sure there's only 1 feedback on one article at most
    feedback.meta.id = f"{user_id}-{article_id}"
    feedback.save()
    print(f"[Feedback Saved] user={user_id}, article={article_id}, liked={liked}")




# get all articles
def get_all_articles(max_results=100):
    s = Article.search().sort("-pubDate")[:max_results]
    results = s.execute()

    print(f"Found {len(results)} articles:\n")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title}\n id:{hit.meta.id}\n{hit.link}\n")
    return results 


# single word search
def search_articles_by_title(keyword):
    q = Q("match", title=keyword)
    s = Article.search().query(q)
    results = s.execute()

    print(f"Found {len(results)} results for keyword: '{keyword}'\n")
    # for hit in results:
    #     print(f"[{hit.pubDate}] {hit.title}\n{hit.link}\n")

# search by categories
def multi_field_search(keyword):
    q = Q(
        "multi_match",
        query=keyword,
        fields=["title", "description", "creator", "category","content"],  
        type="best_fields",  
        fuzziness="AUTO",    
        operator="or",       
        minimum_should_match="70%"  
    )

    s = Article.search().query(q).sort("-pubDate")
    results = s.execute()

    print(f"\n[Multi-field Search] keyword: '{keyword}' → {len(results)} results")
    for hit in results:
        print(f"[{hit.pubDate}] {hit.title} — {hit.creator} / {hit.category}\n id:{hit.meta.id}\n{hit.link}\n {hit.content}\n")

    return results

# search by category fuzzy
def search_by_category_fuzzy(keyword):
    q = Q("match", category=keyword.lower())
    s = Article.search().query(q).sort("-pubDate")[:50]
    results = s.execute()

    print(f"\n[Category Match Search] keyword = '{keyword}' → {len(results)} results")
    for hit in results:
        print(f"- {hit.title} ({hit.pubDate}) — {hit.category}\n  {hit.link}\n")

    return results

# a combination of many filters: set author/title/..
def search_articles_bool(keyword=None, author=None, category=None, max_results=10):
    must_clauses = []
    filter_clauses = []

    # keyword title / description / creator
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



def recommend_articles(user_id, num_results=None):
    keyword_score, seen_ids = build_weighted_keywords(user_id)
    if not keyword_score:
        print("No sufficient feedback found.")
        return []

    top_keywords = [kw for kw, _ in keyword_score.most_common(10)]
    should_clauses = [Q("match", title=kw) | Q("match", description=kw) for kw in top_keywords]

    q = Q("bool", should=should_clauses, minimum_should_match=1)
    s = Article.search().query(q).exclude("ids", values=seen_ids).sort("-pubDate")
    results = list(s.scan())

    # calculate weighted scores
    def weighted_score(article):
        text = (article.title or "") + " " + (article.description or "")
        words = re.findall(r'\w+', text.lower())
        return sum(keyword_score[kw] for kw in top_keywords if kw in words)

    # sort articles by scores
    sorted_results = sorted(results, key=weighted_score, reverse=True)

    if num_results is not None:
        sorted_results = sorted_results[:num_results]

    print(f"\n[Recommendation] user: {user_id}")
    print(f"Top keywords: {top_keywords}\n")
    for hit in sorted_results[:10]:
        print(f"- {hit.title} ({hit.pubDate})\n  Score: {weighted_score(hit)}\n  {hit.link}\n")

    return sorted_results




def parse(entry):
    return {
        "title": entry["title"],
        "link": entry["link"],
        "guid": entry["guid"],
        "description": entry.get("description", ""),
        "creator": entry.get("author", ""),
        "pubDate": datetime(*entry["published_parsed"][:6]),
        "category": [cat["term"] for cat in entry.get("tags", [])],
        "content": entry.get("summary", "") 
    }


def index(article_data):
    Article(**article_data).save()


if __name__ == "__main__":
 print("Starting indexer...")

 Index("articles").delete(ignore=404)
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