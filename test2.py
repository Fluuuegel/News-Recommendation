from indexer import Feedback,recommend_articles,multi_field_search, submit_feedback, get_all_articles;

# multi_field_search("Trump")

def clear_feedback(user_id):
    feedbacks = Feedback.search().filter("term", user_id=user_id).scan()
    for fb in feedbacks:
        fb.delete()
    print(f"[Clear] All feedbacks deleted for user: {user_id}")


clear_feedback("user1")


submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/upshot/medicaid-hospitals-republicans-cuts.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/well/live/eric-topol-longevity-tips.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/06/nyregion/cuomo-mental-health-mayor.html", liked=True)
submit_feedback("user1", article_id="https://www.nytimes.com/2025/05/05/well/teen-wellness-influencer-health-maha.html", liked=True)


recommend_articles("user1")


