import os
from dotenv import load_dotenv

# Create output directory if it doesn't exist
OUTPUT_DIR = "reddit_data"  # Directory to save output files
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load environment variables for API keys
load_dotenv()

# Configuration with subreddit categories
SUBREDDIT_CATEGORIES = {
    "General Discussion & Trends": ["AskReddit", "ExplainLikeImFive", "TodayILearned"],
    "Tech & Programming": ["programming", "MachineLearning", "DataScience"],
    "Finance & Business": ["WallStreetBets", "PersonalFinance", "Entrepreneur"],
    "Health & Psychology": ["mentalhealth", "Fitness", "Psychology"],
    "Entertainment & Pop Culture": ["movies", "gaming", "Music"],
    "News & Politics": ["worldnews", "politics", "news"]
}

# Create a flat list of all subreddits for data collection
SUBREDDITS = [sub for category_subs in SUBREDDIT_CATEGORIES.values() for sub in category_subs]

# API settings
MAX_REQUESTS_PER_MINUTE = 100
POSTS_PER_SUBREDDIT = 1000  # Maximum posts to collect per subreddit

# Time filters for data collection
TIME_FILTERS = {
    "this_week": "week",
    "this_month": "month", 
    "all_time": "all"
}