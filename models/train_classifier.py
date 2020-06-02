# import libraries
import sys
import pandas as pd
import nltk
import re
import sklearn as sk
import pickle

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from scripts import tokenize

nltk.download(["punkt", "wordnet"])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath )
    df = pd.read_sql_table("MessageData", engine)
    
    X = df.message
    Y = df.loc[:, "related":"direct_report"]
    category_names=Y.columns
    
    return X, Y, category_names


def tokenize(text):
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


def build_model():
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ('moclf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        "moclf__estimator__criterion": ["gini", "entropy"],
        "moclf__estimator__max_depth": [5, 10, None]    
    }
    
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=pd.DataFrame(model.predict(X_test))
    Y_pred.columns=category_names
    
    #print("Best parameters: ", model.best_params_)
    
    Y_test=Y_test.reset_index(drop=True)
    acc_avg=0
    for col in category_names:    
        accuracy_col=(Y_test[col]==Y_pred[col]).mean()
        acc_avg=acc_avg+accuracy_col
    print("Avg. accuracy: ", acc_avg/len(category_names)) 
    
    for col in category_names:
        print(col)
        report=classification_report(Y_test[col], Y_pred[col])
        print(report)


def save_model(model, model_filepath):
    file = open(model_filepath, "wb")
    s = pickle.dump(model, file)
    file.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()