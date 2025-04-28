from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from preprocessing import get_processed_text
from sentiment_analysis import analyze_sentiment
from config import SUBREDDIT_CATEGORIES

def extract_features(post_dict):
    """
    Extract additional features from post data
    
    Parameters:
    -----------
    post_dict : dict
        Dictionary containing post data
    
    Returns:
    --------
    dict
        Dictionary with additional features
    """
    # Clean title and text and get processed tokens
    full_content, processed_tokens = get_processed_text(
        post_dict.get("title", ""), 
        post_dict.get("selftext", "")
    )
    
    # Word count features
    features = {
        "title_length": len(post_dict.get("title", "").split()),
        "selftext_length": len(post_dict.get("selftext", "").split()) if post_dict.get("selftext") else 0,
        "total_length": len(full_content.split()),
    }
    
    # Extract time features
    if "created_utc" in post_dict:
        created_date = datetime.fromtimestamp(post_dict["created_utc"])
        features.update({
            "day_of_week": created_date.strftime("%A"),
            "hour_of_day": created_date.hour,
            "month": created_date.strftime("%B"),
            "year": created_date.year,
            "is_weekend": created_date.weekday() >= 5,  # 5 = Saturday, 6 = Sunday
        })
    
    # Sentiment features
    if full_content:
        sentiment = analyze_sentiment(full_content)
        features.update({
            "sentiment_compound": sentiment["compound"],
            "sentiment_pos": sentiment["pos"],
            "sentiment_neg": sentiment["neg"],
            "sentiment_neu": sentiment["neu"],
            "sentiment_category": "positive" if sentiment["compound"] >= 0.05 else 
                               "negative" if sentiment["compound"] <= -0.05 else "neutral"
        })
    
    # Add clean text for analysis
    features.update({
        "clean_title": post_dict.get("title", ""),
        "clean_selftext": post_dict.get("selftext", ""),
        "combined_text": full_content
    })
    
    # Common keywords
    if processed_tokens:
        # Get most common words (excluding stopwords)
        word_counter = Counter(processed_tokens)
        top_words = word_counter.most_common(5)
        
        # Add top keywords
        features["top_keywords"] = ", ".join([word for word, count in top_words]) if top_words else ""
    
    # Add category information
    for category, subs in SUBREDDIT_CATEGORIES.items():
        if post_dict["subreddit"] in subs:
            features["category"] = category
            break
    
    return features

def create_analysis_datasets(combined_df, output_dir):
    """
    Create specialized datasets for different analysis purposes
    
    Parameters:
    -----------
    combined_df : DataFrame
        Combined dataset of all posts
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    dict
        Dictionary of specialized DataFrames
    """
    import os
    import pandas as pd
    
    analysis_dfs = {}
    
    # 1. Sentiment analysis dataset
    sentiment_df = combined_df[['id', 'title', 'selftext', 'subreddit', 'category', 'score', 'num_comments', 
                             'clean_title', 'clean_selftext', 'combined_text',
                             'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 
                             'sentiment_category', 'created_datetime']]
    sentiment_df.to_csv(os.path.join(output_dir, "sentiment_analysis_data.csv"), index=False)
    analysis_dfs['sentiment'] = sentiment_df
    
    # 2. Time series dataset
    if 'created_datetime' in combined_df.columns:
        time_df = combined_df.copy()
        time_df['date'] = time_df['created_datetime'].dt.date
        
        # Daily posts by subreddit
        daily_posts_subreddit = time_df.groupby(['date', 'subreddit']).agg({
            'id': 'count',
            'score': 'mean',
            'sentiment_compound': 'mean',
            'num_comments': 'mean'
        }).reset_index()
        daily_posts_subreddit.rename(columns={'id': 'post_count'}, inplace=True)
        daily_posts_subreddit.to_csv(os.path.join(output_dir, "daily_activity_by_subreddit.csv"), index=False)
        analysis_dfs['daily_subreddit'] = daily_posts_subreddit
        
        # Daily posts by category
        daily_posts_category = time_df.groupby(['date', 'category']).agg({
            'id': 'count',
            'score': 'mean',
            'sentiment_compound': 'mean',
            'num_comments': 'mean'
        }).reset_index()
        daily_posts_category.rename(columns={'id': 'post_count'}, inplace=True)
        daily_posts_category.to_csv(os.path.join(output_dir, "daily_activity_by_category.csv"), index=False)
        analysis_dfs['daily_category'] = daily_posts_category
    
    return analysis_dfs