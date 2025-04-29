import os
import time
import pandas as pd
from datetime import datetime
# Import modules
from config import OUTPUT_DIR, SUBREDDIT_CATEGORIES
from feature_engineering import create_analysis_datasets
from visualization import generate_analysis_report, generate_category_reports, create_visualizations
# Import preprocessing functions
from preprocessing import clean_text, advanced_preprocess, get_processed_text

def load_existing_data(data_dir):
    """
    Load existing CSV files into dataframes
    
    Args:
        data_dir: Directory containing the CSV files
    
    Returns:
        Dictionary of dataframes
    """
    print(f"Loading existing data from {data_dir}...")
    dataframes = {}
    
    # Find all CSV files that start with "reddit_" in the directory
    for filename in os.listdir(data_dir):
        if filename.startswith("reddit_") and filename.endswith(".csv"):
            # Extract dataframe name from filename
            name = filename.replace("reddit_", "").replace("_data.csv", "")
            filepath = os.path.join(data_dir, filename)
            
            # Load the dataframe
            df = pd.read_csv(filepath)
            dataframes[name] = df
            print(f"Loaded {len(df)} records from {filename}")
    
    return dataframes

def preprocess_dataframes(dataframes):
    """
    Preprocess dataframes to ensure proper data types and clean text fields
    
    Args:
        dataframes: Dictionary of dataframes
    
    Returns:
        Dictionary of preprocessed dataframes
    """
    processed_dfs = {}
    
    for name, df in dataframes.items():
        processed_df = df.copy()
        print(f"Preprocessing {name} dataframe...")
        
        # Convert timestamp to datetime if it exists
        if 'created_utc' in processed_df.columns:
            # Convert created_utc to datetime - ensure it's numeric first
            try:
                # First ensure the column is numeric
                processed_df['created_utc'] = pd.to_numeric(processed_df['created_utc'], errors='coerce')
                # Then convert to datetime
                processed_df['created_datetime'] = pd.to_datetime(processed_df['created_utc'], unit='s', errors='coerce')
                # Drop rows where conversion failed (will be NaT)
                initial_count = len(processed_df)
                processed_df = processed_df.dropna(subset=['created_datetime'])
                if len(processed_df) < initial_count:
                    print(f"  - Dropped {initial_count - len(processed_df)} rows with invalid timestamps")
            except Exception as e:
                print(f"  - Warning: Error converting timestamps: {e}")
                # If the conversion fails completely, create an empty datetime column
                if 'created_datetime' not in processed_df.columns:
                    processed_df['created_datetime'] = pd.NaT
        
        # Apply text preprocessing to title and text content fields
        if 'title' in processed_df.columns:
            print(f"  - Cleaning {len(processed_df)} titles...")
            processed_df['clean_title'] = processed_df['title'].apply(clean_text)
        
        # Apply text preprocessing to selftext if available
        if 'selftext' in processed_df.columns:
            print(f"  - Cleaning {len(processed_df)} selftext entries...")
            processed_df['clean_selftext'] = processed_df['selftext'].apply(clean_text)
            
            # Create combined processed content
            print(f"  - Creating combined processed content...")
            # Apply get_processed_text to pairs of title and selftext
            processed_content = processed_df.apply(
                lambda row: get_processed_text(row.get('title', ''), row.get('selftext', '')), 
                axis=1
            )
            # Unpack the tuples into separate columns
            processed_df['full_clean_text'], processed_df['processed_tokens'] = zip(*processed_content)
            
        # If it's a comments dataframe, process the body
        elif 'body' in processed_df.columns:
            print(f"  - Cleaning {len(processed_df)} comment bodies...")
            processed_df['clean_body'] = processed_df['body'].apply(clean_text)
            # Apply advanced preprocessing and store tokens
            processed_df['processed_tokens'] = processed_df['body'].apply(advanced_preprocess)
        
        processed_dfs[name] = processed_df
    
    return processed_dfs

def main():
    """
    Main execution function for the Reddit data analysis project
    """
    print("Starting Enhanced Reddit data analysis from existing files...")
    start_time = time.time()
   
    # Load data from existing CSV files instead of collecting
    dataframes = load_existing_data(OUTPUT_DIR)
    
    if not dataframes:
        print("No existing data files found. Please check the OUTPUT_DIR path.")
        return
    
    # Preprocess dataframes to ensure proper data types and clean text
    dataframes = preprocess_dataframes(dataframes)
   
    # Create a combined dataset of posts
    post_dfs = [df for name, df in dataframes.items() if not name.endswith("_comments")]
    combined_df = pd.concat(post_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=["id"], inplace=True)  # Remove duplicates
    
    # Save the preprocessed combined data
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "reddit_combined_data.csv"), index=False)
    
    # Also save a clean version with only processed text fields
    clean_columns = ['id', 'subreddit', 'created_datetime', 'full_clean_text']
    if set(clean_columns).issubset(combined_df.columns):
        clean_df = combined_df[clean_columns + ['category'] if 'category' in combined_df.columns else clean_columns]
        clean_df.to_csv(os.path.join(OUTPUT_DIR, "reddit_clean_data.csv"), index=False)
        print(f"Saved cleaned dataset with {len(clean_df)} records")
   
    # Create a combined dataset of comments
    comment_dfs = [df for name, df in dataframes.items() if name.endswith("_comments")]
    if comment_dfs:
        combined_comments = pd.concat(comment_dfs, ignore_index=True)
        combined_comments.drop_duplicates(subset=["id"], inplace=True)  # Remove duplicates
        
        # Save the preprocessed combined comments
        combined_comments.to_csv(os.path.join(OUTPUT_DIR, "reddit_combined_comments.csv"), index=False)
        
        # Also save a clean version with only processed text fields
        if 'clean_body' in combined_comments.columns:
            # Determine which columns exist in the dataframe
            available_columns = ['id', 'subreddit', 'created_datetime', 'clean_body']
            optional_columns = ['parent_id', 'link_id']
            
            # Add optional columns only if they exist
            clean_comment_cols = [col for col in available_columns + optional_columns if col in combined_comments.columns]
            
            # Create and save clean comments dataset
            clean_comments = combined_comments[clean_comment_cols]
            clean_comments.to_csv(os.path.join(OUTPUT_DIR, "reddit_clean_comments.csv"), index=False)
            print(f"Saved cleaned comments dataset with {len(clean_comments)} records and columns: {', '.join(clean_comment_cols)}")
   
    # Create category-specific datasets
    category_dir = os.path.join(OUTPUT_DIR, "categories")
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
       
    for category in SUBREDDIT_CATEGORIES.keys():
        if 'category' in combined_df.columns:
            cat_posts = combined_df[combined_df['category'] == category]
            if not cat_posts.empty:
                cat_filename = os.path.join(category_dir, f"{category.lower().replace(' & ', '_').replace(' ', '_')}_data.csv")
                cat_posts.to_csv(cat_filename, index=False)
                
                # Also save a clean version with only processed text fields
                if 'full_clean_text' in cat_posts.columns:
                    clean_cat_cols = ['id', 'subreddit', 'created_datetime', 'full_clean_text']
                    clean_cat_posts = cat_posts[clean_cat_cols]
                    clean_cat_filename = os.path.join(category_dir, f"{category.lower().replace(' & ', '_').replace(' ', '_')}_clean_data.csv")
                    clean_cat_posts.to_csv(clean_cat_filename, index=False)
                    print(f"Saved {len(clean_cat_posts)} cleaned records to {clean_cat_filename}")
                else:
                    print(f"Saved {len(cat_posts)} records to {cat_filename}")
   
    # Generate analysis datasets optimized for specific purposes
    analysis_dfs = create_analysis_datasets(combined_df, OUTPUT_DIR)
   
    # Generate basic analysis report
    generate_analysis_report(dataframes)
   
    # Generate category-specific reports
    generate_category_reports(dataframes)
   
    # Create visualizations
    create_visualizations(dataframes)
   
    total_time = (time.time() - start_time) / 60
    print(f"\nData analysis complete in {total_time:.2f} minutes")
    print(f"Total posts processed: {len(combined_df)}")
   
    # Report by category
    if 'category' in combined_df.columns:
        category_counts = combined_df['category'].value_counts()
        print("\nPosts by category:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count} posts")
   
    print(f"\nAll data saved to {OUTPUT_DIR} directory")
    print(f"Special analysis datasets created:")
    print(f"  - reddit_clean_data.csv: Dataset with all cleaned text")
    print(f"  - reddit_clean_comments.csv: Comments with cleaned text")
    print(f"  - sentiment_analysis_data.csv: Optimized for sentiment analysis")
    print(f"  - daily_activity_by_subreddit.csv: Time series data by subreddit")
    print(f"  - daily_activity_by_category.csv: Time series data by category")
    print(f"  - /categories/: Category-specific datasets and reports")
    print(f"  - analysis_summary.md: Comprehensive analysis report")
    print(f"  - /visualizations/: Data visualizations and charts")

if __name__ == "__main__":
    main()