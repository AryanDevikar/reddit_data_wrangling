import os
import time
import pandas as pd
from datetime import datetime
# Import modules
from config import OUTPUT_DIR, SUBREDDIT_CATEGORIES
from feature_engineering import create_analysis_datasets
from visualization import generate_analysis_report, generate_category_reports, create_visualizations

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
    Preprocess dataframes to ensure proper data types
    
    Args:
        dataframes: Dictionary of dataframes
    
    Returns:
        Dictionary of preprocessed dataframes
    """
    processed_dfs = {}
    
    for name, df in dataframes.items():
        processed_df = df.copy()
        
        # Convert timestamp to datetime if it exists
        if 'created_utc' in processed_df.columns:
            # Convert created_utc to datetime
            processed_df['created_datetime'] = pd.to_datetime(processed_df['created_utc'], unit='s')
        
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
    
    # Preprocess dataframes to ensure proper data types
    dataframes = preprocess_dataframes(dataframes)
   
    # Create a combined dataset of posts
    post_dfs = [df for name, df in dataframes.items() if not name.endswith("_comments")]
    combined_df = pd.concat(post_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=["id"], inplace=True)  # Remove duplicates
    
    # Ensure we have 'created_datetime' in the combined dataframe
    if 'created_utc' in combined_df.columns and 'created_datetime' not in combined_df.columns:
        combined_df['created_datetime'] = pd.to_datetime(combined_df['created_utc'], unit='s')
    
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "reddit_combined_data.csv"), index=False)
   
    # Create a combined dataset of comments
    comment_dfs = [df for name, df in dataframes.items() if name.endswith("_comments")]
    if comment_dfs:
        combined_comments = pd.concat(comment_dfs, ignore_index=True)
        combined_comments.drop_duplicates(subset=["id"], inplace=True)  # Remove duplicates
        
        # Ensure we have 'created_datetime' in the combined comments dataframe
        if 'created_utc' in combined_comments.columns and 'created_datetime' not in combined_comments.columns:
            combined_comments['created_datetime'] = pd.to_datetime(combined_comments['created_utc'], unit='s')
            
        combined_comments.to_csv(os.path.join(OUTPUT_DIR, "reddit_combined_comments.csv"), index=False)
   
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
    print(f"  - sentiment_analysis_data.csv: Optimized for sentiment analysis")
    print(f"  - daily_activity_by_subreddit.csv: Time series data by subreddit")
    print(f"  - daily_activity_by_category.csv: Time series data by category")
    print(f"  - /categories/: Category-specific datasets and reports")
    print(f"  - analysis_summary.md: Comprehensive analysis report")
    print(f"  - /visualizations/: Data visualizations and charts")

if __name__ == "__main__":
    main()