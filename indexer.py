import requests
import feedparser
import schedule
import time
import os
from datetime import datetime
from elasticsearch_dsl import Document, Date, Keyword, Text, connections

# Retrieves environment variables ES_HOST, ES_USER, and ES_PASS
# On Windows you can set them with the syntax "$env:ES_USER=user"
connections.create_connection(
    hosts=os.environ.get("ES_HOST", "http://localhost:9200"),
    basic_auth=(os.environ["ES_USER"], os.environ["ES_PASS"]),
    timeout=20
)


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


def index(article_data):
    Article(**article_data).save()


def retrieve():
    response = requests.get(
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml")
    feed = feedparser.parse(response.content)
    for entry in feed.entries:
        index(parse(entry))


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
