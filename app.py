from flask import Flask, render_template, request, jsonify
from elasticsearch_dsl import connections
from indexer import Feedback

# import your ES-backed functions
from indexer import query_articles, submit_feedback, recommend_articles,query_articles_with_ranking

# ensure ES connection (indexer.py also does this on import)
connections.create_connection(
    hosts=["http://localhost:9200"],
    timeout=20,
    verify_certs=False,
    ssl_show_warn=False
)

Feedback._index.delete(ignore=[404])
Feedback.init()

app = Flask(__name__)
USER_ID = "user1"  # demo user

@app.route("/", methods=["GET"])
def index():
    # just render the search form + rec panel
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search():
    text  = request.args.get("query", "")
    field = request.args.get("field", "keyword")
    hits  = query_articles(field, text, max_results=50)
    return render_template("results.html",
                           news=hits,
                           query=text,
                           field=field)

@app.route("/feedback", methods=["POST"])
def feedback():
    # try JSON first
    data = request.get_json(silent=True)
    if data:
        article_id = data.get("article_id")
        action     = data.get("action")
    else:
        article_id = request.form.get("article_id")
        action     = request.form.get("action")

    if not article_id or action not in ("like", "dislike"):
        return jsonify({"error": "Missing article_id or invalid action"}), 400

    liked = (action == "like")
    submit_feedback(USER_ID, article_id, liked=liked)
    # no redirect, return empty 204
    return ("", 204)

@app.route("/recommend_json", methods=["GET"])
def recommend_json():
    recs = recommend_articles(USER_ID, num_results=10)
    items = [{
      "title": hit.title,
      "link":  hit.link,
      "category": list(hit.category or [])
    } for hit in recs]
    return jsonify(items)

if __name__ == "__main__":
    app.run(debug=True)
