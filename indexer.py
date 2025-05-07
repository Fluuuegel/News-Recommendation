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
    print("TF-IDF not detecting global_idf.pkl 或 global_vocab.pkl，")



def build_and_save_global_tfidf():
    all_articles = get_all_articles(max_results=10000)
    all_texts = [ (a.title or "") + " " + (a.description or "") for a in all_articles ]

    vectorizer = TfidfVectorizer(tokenizer=extract_keywords)
    vectorizer.fit(all_texts)

    joblib.dump(vectorizer.idf_, idf_path)
    joblib.dump(vectorizer.vocabulary_, vocab_path)
    print("[Success] global_idf.pkl 和 global_vocab.pkl generated！")





def fetch_full_content(url):
    try:
        news = NewsArticle(url)
        news.download()
        news.parse()
        return news.text
    except Exception as e:
        print(f"Failed to fetch full content from {url}: {e}")
        return ""


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
        print(f"[{hit.pubDate}] {hit.title}\n id:{hit.meta.id}\n{hit.link}\n content:{hit.content} \n")
    return results 


# single word search
def search_articles_by_title(text):
    s = Article.search().query("match", title=text)
    return s

# search by category fuzzy
def search_by_category_fuzzy(keyword):
    q = Q("match", category=keyword.lower())
    s = Article.search().query(q).sort("-pubDate")[:50]
    return s

# search in all fields
def search_keywords(keywords, max_results=999, min_match_ratio="40%"):

    if isinstance(keywords, str):
        terms = [w for w in keywords.strip().split() if w]
    else:
        terms = keywords
    print("-------searching query is----- " + " ".join(terms))
 

    s = Article.search()

    if len(terms) == 1:
        print("------------now term is 1 --------------")
        q = Q(
            "multi_match",
            query=terms[0],
            fields=["title", "description", "creator", "category", "content"],
            type="most_fields",
            operator="or"
        )
        results = s.query(q).sort("-pubDate")[:max_results].execute()
    else:
  
        should_clauses = []
        for kw in terms:
            should_clauses.append(
                Q("multi_match", query=kw, fields=["title", "description", "creator", "category", "content"], type="most_fields", operator="or")
            )

        q = Q("bool", should=should_clauses, minimum_should_match=min_match_ratio)
        raw_results = s.query(q).sort("-pubDate")[:max_results].execute()

       # sort by matching counts 
        print("sorting by matching counts")
        def count_matches(article):
            content = (article.title or "") + " " + (article.description or "") + " " + (article.content or "")
            text = content.lower()
            match_count = sum(1 for term in terms if term.lower() in text)
            print(f"------------Match count: {match_count} → {article.title}")
            return match_count
        

        results = sorted(raw_results, key=count_matches, reverse=True)

    print(f"\n[Unified Search] keywords: {terms}, min_match: {min_match_ratio} — Found {len(results)} results\n")
    for hit in results[:10]:
        print(f"- {hit.title} ({hit.pubDate})\n   \n  {hit.link}\n")

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



def recommend_articles(user_id, num_results=None):
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
            text = (article.title or "") + " " + (article.description or "")
            article_texts.append(text)
            feedback_weights.append(weight)
            seen_ids.append(article.meta.id)
        except:
            continue

    if not article_texts:
        print("No articles found for feedbacks.")
        return []

    count_vectorizer = CountVectorizer(tokenizer=extract_keywords, vocabulary=global_vocab)
    tf_matrix = count_vectorizer.fit_transform(article_texts)

    transformer = TfidfTransformer()
    transformer.idf_ = global_idf
    
    
    # feature_names = count_vectorizer.get_feature_names_out()

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


    top_keywords = [kw for kw, _ in keyword_score.most_common(10)]
    should_clauses = [Q("match", title=kw) | Q("match", description=kw) for kw in top_keywords]
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


if __name__ == "__main__":
 print("Starting indexer...")

#  global_idf = joblib.load("idf.pkl")
#  global_vocab = joblib.load("vocab.pkl")
 build_and_save_global_tfidf()
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




def query_articles(field, text, max_results=999):

    terms = [w for w in text.strip().split() if w]

    # if len(terms) > 1:
    #     print("now searching by search_multiple_keywords")
    #     return search_keywords(field, terms, max_results)
    
    if field == "title":
        #s = Article.search().query("match", title=text)
        s = search_articles_by_title(text)
    elif field == "category":
        # term 查询 category 列表中的 exact match
        #s = Article.search().query("term", category=text)
        s = search_by_category_fuzzy(text)
    else:  # keyword
        print("now searching by search_keywords")
        s = search_keywords(text,999,"40%")
        return s


    return s.execute()



def query_articles_with_ranking( field, text, max_results=30):

    
    raw_results = query_articles(field, text, max_results)
    keyword_score, _ = build_weighted_keywords("user1")

    if not keyword_score:
        print(f"No user profile found for user: {"user1"}, returning search results.")
        return raw_results

    top_keywords = [kw for kw, _ in keyword_score.most_common(10)]

 
    def weighted_score(article):
        text = (article.title or "") + " " + (article.description or "")
        words = re.findall(r'\w+', text.lower())
        return sum(keyword_score[kw] for kw in top_keywords if kw in words)

    sorted_results = sorted(raw_results, key=weighted_score, reverse=True)

    print(f"\n[Search+Rank] user: {"user1"}, field: {field}, query: '{text}'")
    print(f"Top keywords: {top_keywords}")
    for hit in sorted_results[:10]:
        print(f"- {hit.title} ({hit.pubDate}) | Score: {weighted_score(hit)}")

    return sorted_results





def get_tfidf_matrix(texts):
    global global_idf, global_vocab

    if global_idf is None or global_vocab is None:
        print("[TF-IDF] 模型尚未加载，尝试加载中...")
        if not os.path.exists(idf_path) or not os.path.exists(vocab_path):
            raise RuntimeError("global_idf.pkl 或 global_vocab.pkl 文件不存在")
        global_idf = joblib.load(idf_path)
        global_vocab = joblib.load(vocab_path)
        print("[TF-IDF] 模型加载完成")

    count_vectorizer = CountVectorizer(tokenizer=extract_keywords, vocabulary=global_vocab)
    tf_matrix = count_vectorizer.fit_transform(texts)

    transformer = TfidfTransformer()
    transformer.idf_ = global_idf  # 设置 IDF

    return transformer.transform(tf_matrix), count_vectorizer.get_feature_names_out()
