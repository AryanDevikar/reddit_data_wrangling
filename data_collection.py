import praw
import time
import os
import pandas as pd
from datetime import datetime
from config import SUBREDDIT_CATEGORIES, SUBREDDITS, MAX_REQUESTS_PER_MINUTE, POSTS_PER_SUBREDDIT, TIME_FILTERS
from preprocessing import clean_text
from feature_engineering import extract_features

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT"),
)

def get_reddit_data(subreddit_name, time_filter=None, limit=None):
    """
    Fetch data from Reddit API using PRAW with rate limiting
    
    Parameters:
    -----------
    subreddit_name : str
        Subreddit name
    time_filter : str, optional
        Time filter for top posts (all, year, month, week, day, hour)
    limit : int, optional
        Maximum number of posts to retrieve
        
    Returns:
    --------
    list
        List of post dictionaries
    """
    subreddit = reddit.subreddit(subreddit_name)
    subreddit_category = get_category(subreddit_name)
    posts = []
    request_count = 0
    start_time = time.time()
    
    # Determine which listing to use
    if time_filter:
        submissions = subreddit.top(time_filter=time_filter, limit=limit)
    else:
        submissions = subreddit.new(limit=limit)
    
    # Track the API requests
    post_counter = 0
    
    # Process submissions
    for submission in submissions:
        # Check rate limit
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            elapsed = time.time() - start_time
            if elapsed < 60:
                time.sleep(60 - elapsed)
            request_count = 0
            start_time = time.time()
        
        # Convert PRAW submission to dictionary
        post_dict = {
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "created_utc": submission.created_utc,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "permalink": submission.permalink,
            "url": submission.url,
            "author": str(submission.author),
            "subreddit": submission.subreddit.display_name,
            "upvote_ratio": submission.upvote_ratio,
            "is_self": submission.is_self,
            "over_18": submission.over_18,
            "spoiler": submission.spoiler,
            "stickied": submission.stickied,
            "gilded": submission.gilded,
            "collection_time": datetime.utcnow().timestamp(),
            "category": subreddit_category
        }
        
        # Extract additional features for analysis
        features = extract_features(post_dict)
        post_dict.update(features)
        
        # Get top level comments
        post_dict["top_comments"] = []
        try:
            # Only get a few top comments to avoid excessive API usage
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)  # Don't fetch MoreComments
            for comment in submission.comments[:10]:  # Increased to 10 for better analysis
                comment_text = comment.body
                comment_dict = {
                    "id": comment.id,
                    "body": comment_text,
                    "score": comment.score,
                    "created_utc": comment.created_utc,
                    "author": str(comment.author),
                    "post_id": submission.id,
                    "subreddit": submission.subreddit.display_name,
                    "category": post_dict["category"]  # Add the category to comments too
                }
                
                # Clean comment text
                clean_comment = clean_text(comment_text)
                
                # Add comment sentiment
                if clean_comment:
                    from sentiment_analysis import analyze_sentiment
                    sentiment = analyze_sentiment(clean_comment)
                    comment_dict.update({
                        "clean_body": clean_comment,
                        "comment_length": len(comment_text.split()),
                        "sentiment_compound": sentiment["compound"],
                        "sentiment_pos": sentiment["pos"],
                        "sentiment_neg": sentiment["neg"],
                        "sentiment_neu": sentiment["neu"],
                        "sentiment_category": "positive" if sentiment["compound"] >= 0.05 else 
                                          "negative" if sentiment["compound"] <= -0.05 else "neutral"
                    })
                
                post_dict["top_comments"].append(comment_dict)
            request_count += 1  # Count fetching comments as a request
        except Exception as e:
            print(f"Error fetching comments: {str(e)}")
        
        posts.append(post_dict)
        request_count += 1
        post_counter += 1
        
        # Update progress periodically
        if post_counter % 25 == 0:
            print(f"  - Collected {post_counter} posts from r/{subreddit_name}")
        
        # Break if we have enough posts
        if limit and len(posts) >= limit:
            break
            
    return posts

def collect_comprehensive_data():
    """
    Collect comprehensive data from all specified subreddits
    
    Returns:
    --------
    dict
        Dictionary of DataFrames with different time filters
    """
    all_data = {}
    
    # Process each subreddit
    for subreddit in SUBREDDITS:
        print(f"Collecting data from r/{subreddit}")
        
        # Get latest posts
        print(f"  - Fetching latest posts")
        latest_posts = get_reddit_data(
            subreddit_name=subreddit,
            limit=POSTS_PER_SUBREDDIT // 2  # Split allocation between new and top
        )
        
        # Get top posts with different time filters
        for name, time_filter in TIME_FILTERS.items():
            print(f"  - Fetching {name} top posts")
            posts = get_reddit_data(
                subreddit_name=subreddit,
                time_filter=time_filter,
                limit=POSTS_PER_SUBREDDIT // (len(TIME_FILTERS) * 2)  # Split allocation
            )
            
            # Store in dictionary by time filter
            if name not in all_data:
                all_data[name] = []
            all_data[name].extend(posts)
        
        # Store latest posts
        if "latest" not in all_data:
            all_data["latest"] = []
        all_data["latest"].extend(latest_posts)
    
    # Convert to DataFrames
    dataframes = {}
    for category, posts in all_data.items():
        # Extract all fields except comments
        df = pd.DataFrame([{k: v for k, v in post.items() if k != "top_comments"} for post in posts])
        
        # Convert timestamps to datetime
        if "created_utc" in df.columns and not df.empty:
            df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s")
        
        # Store comments separately
        comments_list = [post.get("top_comments", []) for post in posts]
        comments_flat = [comment for sublist in comments_list for comment in sublist]
        
        if comments_flat:
            comments_df = pd.DataFrame(comments_flat)
            if "created_utc" in comments_df.columns:
                comments_df["created_datetime"] = pd.to_datetime(comments_df["created_utc"], unit="s")
            dataframes[f"{category}_comments"] = comments_df
        
        dataframes[category] = df
    
    return dataframes

def get_category(subreddit_name):
    for category, subs in SUBREDDIT_CATEGORIES.items():
        if subreddit_name in subs:
            return category
    return "Unknown"

dataframes = collect_comprehensive_data()
OUTPUT_DIR = "reddit_data"

# Save each dataframe to CSV
for name, df in dataframes.items():
    filename = os.path.join(OUTPUT_DIR, f"reddit_{name}_data.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")