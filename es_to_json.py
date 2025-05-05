import os
import json
from elasticsearch import Elasticsearch

client = Elasticsearch(
    os.environ.get("ES_HOST", "https://localhost:9200"),
    basic_auth=(os.environ["ES_USER"], os.environ["ES_PASS"]),
    verify_certs=False,
    ssl_show_warn=False
)


resp = client.search(index="articles", size=1000)
hits = resp["hits"]["hits"]


articles = []
for i, hit in enumerate(hits):
    source = hit["_source"]
    articles.append({
        "id": i + 1,
        "title": source.get("title", "No Title"),
        "content": source.get("description", "No Description")
    })


with open("news.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print(f"Written {len(articles)} news to news.json")
