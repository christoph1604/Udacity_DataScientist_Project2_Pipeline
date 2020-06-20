# import libraries
import sys
import pandas as pd
import nltk
import re
import sklearn as sk
import math
import pickle
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

from scripts import tokenize
from scripts import StartingVerbExtractor
from scripts import TextLengthExtractor

nltk.download(["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"])

# Regular expression for finding URLs in messages
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
    Loads dataset from sqllite database and splits it into training and test data.
    """
    engine = create_engine('sqlite:///'+database_filepath )
    df = pd.read_sql_table("MessageData", engine)
    
    X = df.message
    Y = df.loc[:, "related":"direct_report"]
    category_names=Y.columns
    
    return X, Y, category_names

def build_model():
    """
    Builds a pipeline to create a classification model. 
    The pipeline
    - Tokenizes the phrases and counts the single tokens.
    - Calculates the TF-IDF value for each token
    - Calculates the length of each message phrase (number of letters)
    - Determines whether the starting token of a phrase is a verb
    - Uses multilabel classification to determine the categories of the phrases (using the RandomForestClassifier)
    - Uses grid search to optimize the classifier. 
    """
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("nlp_pipeline", Pipeline([
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer())
            ])),
            ("txt_len", TextLengthExtractor()),
            ("sve", StartingVerbExtractor())        
        ])),        
        ('moclf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        "moclf__estimator__n_estimators": [10,100]    
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Tests the trained model on test data and compares the predicted result on the real category labelling.
    Prints out the optimal grid search parameters of the training.
    Calculates the overall accuracy on all categories. 
    Prints out a classification report for every category. 
    """
    Y_pred=pd.DataFrame(model.predict(X_test))
    Y_pred.columns=category_names
    
    print("Best parameters: ", model.best_params_)
    
    Y_test=Y_test.reset_index(drop=True)
    print("Avg. accuracy: ", (Y_pred==Y_test).mean().mean()) 
    
    for col in category_names:
        print(col)
        report=classification_report(Y_test[col], Y_pred[col])
        print(report)


def save_model(model, model_filepath):
    """
    Saves the created model to a .pkl file. 
    """
    file = open(model_filepath, "wb")
    s = pickle.dump(model, file)
    file.close()

def main():
    """
    Executes the training, testing and saving of the classification model. 
    """
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