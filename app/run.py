# Imports
import json
import plotly
import pandas as pd
import math
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sqlalchemy import create_engine

from scripts import tokenize

# Create Flask app
app = Flask(__name__)

# load data

# Version for Heroku (starts skript from top folder)
# engine = create_engine('sqlite:///data/DisasterResponse.db')

# Version for local execution (starts script from app folder)
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageData', engine)

# load model

# Version for Heroku (starts skript from top folder)
#model = joblib.load("models/classifier.pkl")

# Version for local execution (starts script from app folder)
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Index webpage. Displays visualizations and receives user input text for model
    """
    
    # extract data needed for visuals    
    # Visualization 1: Frequency of categories
    cat_number=df.loc[:, "related":"direct_report"].sum(axis=0).reset_index()
    cat_number.columns=["category", "count"]
    cat_number=cat_number.sort_values(by="count")
    
    # Visualization 2: Average length of message string by category
    df["letter_count"]=df.message.str.len()
    cat_strlen_l=[]
    categories=df.loc[:, "related":"direct_report"].columns
    for cat in categories:
        avg_strlen=df[df[cat]==1].letter_count.mean()
        if(math.isnan(avg_strlen)):
            avg_strlen=0
        cat_strlen_l.append({"category": cat, "avg_strlen": avg_strlen})
    cat_strlen=pd.DataFrame(cat_strlen_l)
    cat_strlen=cat_strlen.sort_values("avg_strlen")    
    
    # Visualization 3: Frequency of genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
        
    # Create visualizations
    graphs = [
        # Visualization 1: Frequency of categories in a pie chart
        {
            "data": [
                Pie(
                    values=cat_number["count"],
                    labels=cat_number["category"]
                )            
            ],
            
            "layout": {
                "title": "Distribution of Message Categories in Training Set",
                "width": "1000",
                "height": "625",
                "autosize": False,                
            
            }
        },
        
        # Visualization 2: Average length of message string by category in a horizontal bar chart
        {
            "data": [
                Bar(
                    x=cat_strlen.avg_strlen,
                    y=cat_strlen.category,
                    orientation="h"
                )            
            ],
            
            "layout": {
                "title": "Average message length by category",
                "width": "1000",
                "height": "1000",
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Average message length (in letters)"
                }, 
                "margin": {
                    "l": 200
                }
            
            }
        },        
        
        # Visualization 3: Frequency of genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
 
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    Web page that handles user query and displays model results.
    """    
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main method. Runs application on predefined IP / port
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()