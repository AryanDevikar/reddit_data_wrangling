import os
from datetime import datetime
import pandas as pd
from collections import Counter
from config import SUBREDDIT_CATEGORIES, OUTPUT_DIR

def generate_analysis_report(dataframes):
    """
    Generate basic analysis report from collected data
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary of DataFrames
    """
    # Combine all post DataFrames
    post_dfs = [df for name, df in dataframes.items() if not name.endswith("_comments")]
    all_posts = pd.concat(post_dfs, ignore_index=True)
    all_posts.drop_duplicates(subset=["id"], inplace=True)
    
    # Combine all comment DataFrames
    comment_dfs = [df for name, df in dataframes.items() if name.endswith("_comments")]
    all_comments = pd.concat(comment_dfs, ignore_index=True) if comment_dfs else None
    
    if all_comments is not None:
        all_comments.drop_duplicates(subset=["id"], inplace=True)
    
    # Generate report
    report = []
    report.append("# Reddit Data Analysis Summary Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall statistics
    report.append("\n## Overall Statistics")
    report.append(f"- Total Posts: {len(all_posts)}")
    report.append(f"- Total Comments: {len(all_comments) if all_comments is not None else 0}")
    report.append(f"- Subreddits: {', '.join(all_posts['subreddit'].unique())}")
    
    # Safely get date range
    try:
        # Make sure created_datetime has only datetime objects by dropping NaT values
        valid_dates = all_posts['created_datetime'].dropna()
        if len(valid_dates) > 0:
            report.append(f"- Date Range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
        else:
            report.append("- Date Range: Not available (no valid dates)")
    except (TypeError, AttributeError) as e:
        report.append(f"- Date Range: Not available (error: {str(e)})")
    
    # Sentiment distribution (if available)
    if 'sentiment_category' in all_posts.columns:
        report.append("\n## Sentiment Analysis")
        sentiment_dist = all_posts['sentiment_category'].value_counts()
        report.append("### Post Sentiment Distribution")
        for category, count in sentiment_dist.items():
            report.append(f"- {category.capitalize()}: {count} posts ({count/len(all_posts)*100:.1f}%)")
    else:
        report.append("\n## Sentiment Analysis not available (missing sentiment_category column)")
    
    # Category statistics
    report.append("\n## Category Statistics")
    if 'category' in all_posts.columns:
        for category in SUBREDDIT_CATEGORIES.keys():
            cat_posts = all_posts[all_posts['category'] == category]
            if len(cat_posts) > 0:
                report.append(f"\n### {category}")
                report.append(f"- Total Posts: {len(cat_posts)}")
                report.append(f"- Average Score: {cat_posts['score'].mean():.1f}")
                report.append(f"- Average Comments: {cat_posts['num_comments'].mean():.1f}")
                
                # Sentiment distribution within category (if available)
                if 'sentiment_category' in cat_posts.columns:
                    cat_sentiment = cat_posts['sentiment_category'].value_counts()
                    report.append("- Sentiment Distribution:")
                    for sentiment, count in cat_sentiment.items():
                        report.append(f"  - {sentiment.capitalize()}: {count} posts ({count/len(cat_posts)*100:.1f}%)")
                
                # Most upvoted post in category
                if not cat_posts.empty:
                    try:
                        top_post = cat_posts.loc[cat_posts['score'].idxmax()]
                        title = top_post.get('title', 'Untitled')
                        # Replace or remove problematic characters
                        title = ''.join(c if ord(c) < 128 else '?' for c in title)
                        report.append(f"- Top Post: '{title}' in r/{top_post['subreddit']} (Score: {top_post['score']})")
                    except Exception as e:
                        report.append(f"- Top Post: Unable to determine (error: {str(e)})")
    
    # Subreddit statistics within categories
    report.append("\n## Subreddit Statistics")
    for category, subreddits in SUBREDDIT_CATEGORIES.items():
        report.append(f"\n### {category}")
        for subreddit in subreddits:
            sub_posts = all_posts[all_posts['subreddit'] == subreddit]
            if len(sub_posts) > 0:
                report.append(f"\n#### r/{subreddit}")
                report.append(f"- Total Posts: {len(sub_posts)}")
                report.append(f"- Average Score: {sub_posts['score'].mean():.1f}")
                report.append(f"- Average Comments: {sub_posts['num_comments'].mean():.1f}")
                
                # Most upvoted post
                if not sub_posts.empty:
                    try:
                        top_post = sub_posts.loc[sub_posts['score'].idxmax()]
                        title = top_post.get('title', 'Untitled')
                        # Replace or remove problematic characters
                        title = ''.join(c if ord(c) < 128 else '?' for c in title)
                        report.append(f"- Top Post: '{title}' (Score: {top_post['score']})")
                    except Exception as e:
                        report.append(f"- Top Post: Unable to determine (error: {str(e)})")
    
    # Temporal patterns
    if 'hour_of_day' in all_posts.columns:
        report.append("\n## Temporal Patterns")
        hour_counts = all_posts['hour_of_day'].value_counts().sort_index()
        if not hour_counts.empty:
            peak_hour = hour_counts.idxmax()
            report.append(f"- Peak Posting Hour: {peak_hour}:00 ({hour_counts[peak_hour]} posts)")
        
        if 'day_of_week' in all_posts.columns:
            day_counts = all_posts['day_of_week'].value_counts()
            report.append("- Posts by Day of Week:")
            for day, count in day_counts.items():
                report.append(f"  - {day}: {count} posts")
            
        # Temporal patterns by category
        if 'category' in all_posts.columns:
            report.append("\n### Posting Patterns by Category")
            for category in SUBREDDIT_CATEGORIES.keys():
                cat_posts = all_posts[all_posts['category'] == category]
                if len(cat_posts) > 0:
                    cat_hour_counts = cat_posts['hour_of_day'].value_counts().sort_index()
                    cat_peak_hour = cat_hour_counts.idxmax() if not cat_hour_counts.empty else 0
                    report.append(f"\n#### {category}")
                    report.append(f"- Peak Posting Hour: {cat_peak_hour}:00 ({cat_hour_counts.get(cat_peak_hour, 0)} posts)")
                    if 'day_of_week' in cat_posts.columns:
                        cat_day_counts = cat_posts['day_of_week'].value_counts()
                        report.append("- Most Active Day: " + (cat_day_counts.idxmax() if not cat_day_counts.empty else "N/A"))
    
    # Save report
    try:
        with open(os.path.join(OUTPUT_DIR, "analysis_summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(report))
    except UnicodeEncodeError:
        # Fallback if utf-8 encoding fails
        clean_report = [sanitize_text(line) for line in report]
        with open(os.path.join(OUTPUT_DIR, "analysis_summary.md"), "w", encoding="ascii", errors="replace") as f:
            f.write("\n".join(clean_report))

def sanitize_text(text):
    """Remove or replace non-ASCII characters in text"""
    return ''.join(c if ord(c) < 128 else '?' for c in text)

def generate_category_reports(dataframes):
    """
    Generate separate analysis reports for each category
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary of DataFrames
    """
    # Combine all post DataFrames
    post_dfs = [df for name, df in dataframes.items() if not name.endswith("_comments")]
    all_posts = pd.concat(post_dfs, ignore_index=True)
    all_posts.drop_duplicates(subset=["id"], inplace=True)
    
    # Create category directory if it doesn't exist
    category_dir = os.path.join(OUTPUT_DIR, "categories")
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    # Generate a report for each category
    for category in SUBREDDIT_CATEGORIES.keys():
        cat_posts = all_posts[all_posts['category'] == category]
        
        if len(cat_posts) == 0:
            continue
            
        # Generate report
        report = []
        report.append(f"# {category} - Reddit Analysis Report")
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall category statistics
        report.append("\n## Category Overview")
        report.append(f"- Total Posts: {len(cat_posts)}")
        report.append(f"- Average Score: {cat_posts['score'].mean():.1f}")
        report.append(f"- Average Comments: {cat_posts['num_comments'].mean():.1f}")
        
        # Sentiment distribution
        sentiment_dist = cat_posts['sentiment_category'].value_counts()
        report.append("\n## Sentiment Analysis")
        for sentiment, count in sentiment_dist.items():
            report.append(f"- {sentiment.capitalize()}: {count} posts ({count/len(cat_posts)*100:.1f}%)")
        
        # Subreddit statistics within this category
        report.append("\n## Subreddit Statistics")
        for subreddit in SUBREDDIT_CATEGORIES[category]:
            sub_posts = cat_posts[cat_posts['subreddit'] == subreddit]
            if len(sub_posts) > 0:
                report.append(f"\n### r/{subreddit}")
                report.append(f"- Total Posts: {len(sub_posts)}")
                report.append(f"- Average Score: {sub_posts['score'].mean():.1f}")
                report.append(f"- Average Comments: {sub_posts['num_comments'].mean():.1f}")
                
                # Top keywords
                if 'top_keywords' in sub_posts.columns:
                    # Aggregate keywords across posts
                    all_keywords = []
                    for keywords in sub_posts['top_keywords'].dropna():
                        if keywords:
                            all_keywords.extend(keywords.split(', '))
                            keyword_counts = Counter(all_keywords)
                    top_keywords = keyword_counts.most_common(10)
                    
                    if top_keywords:
                        report.append("- Most Common Keywords:")
                        for keyword, count in top_keywords:
                            # Sanitize keywords
                            safe_keyword = sanitize_text(keyword)
                            report.append(f"  - {safe_keyword}: {count} occurrences")
                
                # Most upvoted posts
                top_posts = sub_posts.sort_values('score', ascending=False).head(3)
                if not top_posts.empty:
                    report.append("- Top Posts:")
                    for _, post in top_posts.iterrows():
                        # Sanitize title
                        title = post.get('title', 'Untitled')
                        safe_title = sanitize_text(title)
                        report.append(f"  - '{safe_title}' (Score: {post['score']}, Comments: {post['num_comments']})")
        
        # Temporal patterns
        if 'hour_of_day' in cat_posts.columns:
            report.append("\n## Temporal Patterns")
            hour_counts = cat_posts['hour_of_day'].value_counts().sort_index()
            peak_hour = hour_counts.idxmax() if not hour_counts.empty else 0
            report.append(f"- Peak Posting Hour: {peak_hour}:00 ({hour_counts.get(peak_hour, 0)} posts)")
            
            if 'day_of_week' in cat_posts.columns:
                day_counts = cat_posts['day_of_week'].value_counts()
                report.append("- Posts by Day of Week:")
                for day, count in sorted(day_counts.items(), key=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(x[0]) if x[0] in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] else 7):
                    report.append(f"  - {day}: {count} posts")
        
        # Save report
        try:
            with open(os.path.join(category_dir, f"{category.lower().replace(' & ', '_').replace(' ', '_')}_report.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(report))
        except UnicodeEncodeError:
            # Fallback to ASCII with replacement if UTF-8 fails
            clean_report = [sanitize_text(line) for line in report]
            with open(os.path.join(category_dir, f"{category.lower().replace(' & ', '_').replace(' ', '_')}_report.md"), "w", encoding="ascii", errors="replace") as f:
                f.write("\n".join(clean_report))

def create_visualizations(dataframes):
    """
    Create visualizations from Reddit data
    Note: This is a placeholder function that would implement data visualization
          with libraries like matplotlib, seaborn, plotly, etc.
    """
    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        has_viz_libs = True
    except ImportError:
        print("Warning: Visualization libraries not found. Install matplotlib and seaborn to enable visualizations.")
        has_viz_libs = False
        return
    
    if not has_viz_libs:
        return
    
    # Create a visualization directory
    viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Combine all post DataFrames
    post_dfs = [df for name, df in dataframes.items() if not name.endswith("_comments")]
    all_posts = pd.concat(post_dfs, ignore_index=True)
    all_posts.drop_duplicates(subset=["id"], inplace=True)
    
    # 1. Sentiment Distribution by Category
    if 'category' in all_posts.columns and 'sentiment_category' in all_posts.columns:
        plt.figure(figsize=(12, 8))
        sentiment_by_cat = pd.crosstab(all_posts['category'], all_posts['sentiment_category'])
        sentiment_by_cat_pct = sentiment_by_cat.div(sentiment_by_cat.sum(axis=1), axis=0)
        
        ax = sentiment_by_cat_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Sentiment Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Percentage')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_by_category.png'))
        plt.close()
    
    # 2. Post Volume by Hour of Day
    if 'hour_of_day' in all_posts.columns:
        plt.figure(figsize=(10, 6))
        hour_counts = all_posts['hour_of_day'].value_counts().sort_index()
        sns.lineplot(x=hour_counts.index, y=hour_counts.values)
        plt.title('Post Volume by Hour of Day')
        plt.xlabel('Hour (UTC)')
        plt.ylabel('Number of Posts')
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'post_volume_by_hour.png'))
        plt.close()
    
    # 3. Average Score by Subreddit
    plt.figure(figsize=(12, 8))
    sub_scores = all_posts.groupby('subreddit')['score'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=sub_scores.index, y=sub_scores.values)
    plt.title('Average Post Score by Subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'avg_score_by_subreddit.png'))
    plt.close()
    
    # 4. Correlation Heatmap
    numeric_cols = ['score', 'num_comments']
    # Add optional columns if they exist
    for col in ['upvote_ratio', 'title_length', 'selftext_length', 'total_length', 'sentiment_compound']:
        if col in all_posts.columns:
            numeric_cols.append(col)
    
    numeric_data = all_posts[numeric_cols].copy()
    
    plt.figure(figsize=(10, 8))
    correlation = numeric_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"Visualizations saved to {viz_dir} directory")