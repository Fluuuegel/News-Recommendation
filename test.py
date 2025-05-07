from indexer import Feedback,recommend_articles,multi_field_search, submit_feedback, get_all_articles;

# multi_field_search("Trump")

def clear_feedback(user_id):
    feedbacks = Feedback.search().filter("term", user_id=user_id).scan()
    for fb in feedbacks:
        fb.delete()
    print(f"[Clear] All feedbacks deleted for user: {user_id}")


clear_feedback("user1")


submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/business/mattel-barbie-trump-tariffs-prices.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/nyregion/newark-migrants-ras-baraka-ice.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/arts/design/national-african-american-museum-faces-uncertainty-without-its-leader.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/business/uk-india-trade-deal-tariffs.html", liked=True)
# submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/business/mattel-barbie-trump-tariffs-prices.html", liked=True)

recommend_articles("user1")


# Top keywords: ['crypto', 'deals', 'provoke', 'senate', 'backlash', 'investigation', 
# 'democrats', 'supported', 'legislation', 'stablecoins']