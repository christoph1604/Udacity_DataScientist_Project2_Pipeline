# Imports
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Regular expression for finding URLs in messages
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
"""
Tokenize method. Executes the following tasks:
- Replacement of URLs in messages
- Removal of punctuation in phrases
- Conversion to lower case
- Word tokenization
- Lemmatization
"""
    
    found_urls=re.findall(url_regex, text)
    for pos in found_urls:
        text=text.replace(pos, "urlplaceholder")
    text=re.sub(r"[^a-zA-Z0-9]", " ", text) 
    text=text.lower()
    words=nltk.word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in words:
        tok=lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(tok)
        
    return clean_tokens