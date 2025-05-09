import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text for analysis
    
    Parameters:
    -----------
    text : str
        Text to clean
    
    Returns:
    --------
    str
        Cleaned text
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_preprocess(text):
    """
    Advanced preprocessing including tokenization, stopword removal,
    and lemmatization
    
    Parameters:
    -----------
    text : str
        Text to preprocess
    
    Returns:
    --------
    list
        List of processed tokens
    """
    if not text:
        return []
    
    # Basic cleaning
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized

def get_processed_text(title, selftext):
    """
    Process title and selftext together
    
    Parameters:
    -----------
    title : str
        Post title
    selftext : str
        Post content
    
    Returns:
    --------
    tuple
        (cleaned_combined_text, processed_tokens)
    """
    # Clean title and text
    clean_title = clean_text(title or "")
    clean_selftext = clean_text(selftext or "")
    
    # Combine for content analysis
    full_content = f"{clean_title} {clean_selftext}"
    
    # Advanced processing for modeling
    processed_tokens = advanced_preprocess(full_content)
    
    return full_content, processed_tokens