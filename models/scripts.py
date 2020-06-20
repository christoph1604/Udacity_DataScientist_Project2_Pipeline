#Imports
import re
import nltk
import pandas as pd

from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

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
    
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Class for determining whether the starting word of a phrase is a verb.
    """
    
    def starting_verb(self, text):
    """
    Determines whether the starting word of a phrase is a verb.
    
    Tokenizes the phrase, tags the words and checks whether the starting word is a verb.    
    """        
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
            else:
                return False
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
    """
    Dummy fit method.
    """
        return self

    def transform(self, X):
    """
    Transform method. Executes the starting_verb method for all phrases. 
    """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
        
        
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Text length extractor.
    
    Determines the length of a phrase.
    """
    
    def fit(self, X, y=None):
    """
    Dummy fit method.
    """
        return self

    def transform(self, X):
    """
    Determines the string length for all given message phrases.
    """
        X_len = X.str.len()
        return pd.DataFrame(X_len)