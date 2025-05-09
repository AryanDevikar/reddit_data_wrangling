import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with sentiment scores
    """
    if not text:
        return {
            "compound": 0,
            "pos": 0,
            "neg": 0,
            "neu": 1
        }
    
    return sia.polarity_scores(text)

def get_sentiment_category(compound_score):
    """
    Convert compound sentiment score to category
    
    Parameters:
    -----------
    compound_score : float
        VADER compound sentiment score
    
    Returns:
    --------
    str
        Sentiment category (positive, negative, neutral)
    """
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_comment_sentiment(comments):
    """
    Analyze sentiment for a list of comments
    
    Parameters:
    -----------
    comments : list
        List of comment dictionaries
    
    Returns:
    --------
    dict
        Dictionary with comment sentiment statistics
    """
    if not comments:
        return {
            "avg_sentiment": 0,
            "pos_percentage": 0,
            "neg_percentage": 0,
            "neu_percentage": 0
        }
    
    sentiments = [comment.get("sentiment_compound", 0) for comment in comments]
    categories = [comment.get("sentiment_category", "neutral") for comment in comments]
    
    pos_count = categories.count("positive")
    neg_count = categories.count("negative")
    neu_count = categories.count("neutral")
    total = len(categories)
    
    return {
        "avg_sentiment": sum(sentiments) / len(sentiments) if sentiments else 0,
        "pos_percentage": (pos_count / total) * 100 if total > 0 else 0,
        "neg_percentage": (neg_count / total) * 100 if total > 0 else 0,
        "neu_percentage": (neu_count / total) * 100 if total > 0 else 0
    }

# You can add more advanced sentiment analysis functions here
# For a future extension using a more sophisticated model like BERT:

def initialize_advanced_sentiment_model():
    """
    Placeholder for initializing a more sophisticated sentiment model
    
    This function would load and prepare a pre-trained model like BERT
    for sentiment analysis.
    """
    # In a real implementation, this would load a pretrained model
    pass

def analyze_sentiment_advanced(text):
    """
    Placeholder for analyzing sentiment using a more sophisticated model
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with sentiment scores
    """
    # For now, this just calls the VADER analyzer
    # In a real implementation, this would use the loaded advanced model
    return analyze_sentiment(text)