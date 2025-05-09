from datetime import datetime
from collections import Counter
import pandas as pd
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
    
    # Check if required columns exist
    required_columns = ['sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_category']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing sentiment columns: {missing_columns}")
        print("Calculating sentiment for the dataset...")
        # Calculate sentiment if not already present
        if 'full_clean_text' in combined_df.columns:
            sentiment_results = combined_df['full_clean_text'].apply(analyze_sentiment)
            combined_df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
            combined_df['sentiment_pos'] = sentiment_results.apply(lambda x: x['pos'])
            combined_df['sentiment_neg'] = sentiment_results.apply(lambda x: x['neg'])
            combined_df['sentiment_neu'] = sentiment_results.apply(lambda x: x['neu'])
            combined_df['sentiment_category'] = combined_df['sentiment_compound'].apply(
                lambda x: "positive" if x >= 0.05 else "negative" if x <= -0.05 else "neutral"
            )
    
    # Define columns to include, using only those that exist in the dataframe
    sentiment_columns = ['id', 'subreddit', 'score', 'num_comments', 'created_datetime']
    
    # Add optional columns if they exist
    optional_columns = ['title', 'selftext', 'category', 'clean_title', 'clean_selftext', 
                      'full_clean_text', 'combined_text', 'sentiment_compound', 
                      'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_category']
    
    sentiment_cols = sentiment_columns + [col for col in optional_columns if col in combined_df.columns]
    
    # 1. Create sentiment analysis dataset with available columns
    sentiment_df = combined_df[sentiment_cols].copy()
    sentiment_df.to_csv(os.path.join(output_dir, "sentiment_analysis_data.csv"), index=False)
    analysis_dfs['sentiment'] = sentiment_df
    print(f"Created sentiment analysis dataset with {len(sentiment_df)} records and {len(sentiment_cols)} columns")
    
    # 2. Time series dataset
    if 'created_datetime' in combined_df.columns:
        # Make a copy of the dataframe to avoid modifications to the original
        time_df = combined_df.copy()
        
        # Ensure created_datetime is datetime type
        if not pd.api.types.is_datetime64_any_dtype(time_df['created_datetime']):
            print("Converting created_datetime to datetime format...")
            time_df['created_datetime'] = pd.to_datetime(time_df['created_datetime'], errors='coerce')
            
            # Drop rows where conversion failed
            valid_dates = time_df['created_datetime'].notna()
            if not valid_dates.all():
                print(f"Warning: {(~valid_dates).sum()} rows had invalid dates and were dropped")
                time_df = time_df[valid_dates]
        
        # Extract date component
        time_df['date'] = time_df['created_datetime'].dt.date
        
        # Daily posts by subreddit
        agg_columns = {'id': 'count'}
        
        # Add optional aggregations if columns exist
        if 'score' in time_df.columns:
            agg_columns['score'] = 'mean'
        if 'sentiment_compound' in time_df.columns:
            agg_columns['sentiment_compound'] = 'mean'
        if 'num_comments' in time_df.columns:
            agg_columns['num_comments'] = 'mean'
            
        # Group by date and subreddit
        daily_posts_subreddit = time_df.groupby(['date', 'subreddit']).agg(agg_columns).reset_index()
        
        if 'id' in daily_posts_subreddit.columns:
            daily_posts_subreddit.rename(columns={'id': 'post_count'}, inplace=True)
            
        daily_posts_subreddit.to_csv(os.path.join(output_dir, "daily_activity_by_subreddit.csv"), index=False)
        analysis_dfs['daily_subreddit'] = daily_posts_subreddit
        print(f"Created daily subreddit activity dataset with {len(daily_posts_subreddit)} records")
        
        # Daily posts by category if category exists
        if 'category' in time_df.columns:
            daily_posts_category = time_df.groupby(['date', 'category']).agg(agg_columns).reset_index()
            
            if 'id' in daily_posts_category.columns:
                daily_posts_category.rename(columns={'id': 'post_count'}, inplace=True)
                
            daily_posts_category.to_csv(os.path.join(output_dir, "daily_activity_by_category.csv"), index=False)
            analysis_dfs['daily_category'] = daily_posts_category
            print(f"Created daily category activity dataset with {len(daily_posts_category)} records")
    else:
        print("Warning: 'created_datetime' column not found, skipping time series analysis")
    
    return analysis_dfs