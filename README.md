# Disaster Response Pipeline Project

### Summary

This project contains an implementation of a disaster response pipeline, i.e. a classification tool for social media messages written during disaster events. 
The project has been developed as part of the Udacity nanodegree "Data Scientist" (course project 2). It is an example application using the Python scikit-learn framework for the preprocessing and classification of text messages into 36 different disaster classes (e.g. storm, earthquake, refugees, ...)

The project includes:
* An ETL pipeline to clean the message data and store them in an SQLlite database
* A machine learning pipeline which trains a classifier based on the labelled training messages
* A web application which provides a possibility for message classification as well as a statistical overview of the training dataset

### File content

* /app: Implementation of the web application
	* /app/run.py: Python script containing the main method for running the web application.
* /data: Folder containing the data source 
	* /data/disaster_categories.csv: .csv file containing training data
	* /data/disaster_messages.csv: .csv file containing training data
	* /data/process_data.py: ETL script to clean/transform the .csv files to an SQLlite database
	* /data/DisasterResponse.db: Resulting SQLlite database
* /models: Folder containing the ML model
	* /models/train_classifier.py: Python script containing the ML pipeline to read in the source data, transform them and train a machine learning model
	* /models/classifier.pkl: Resulting ML model used in the web app
* /notebooks: Jupyter notebooks used to develop/test/debug the source code of the ETL pipeline, the machine learning pipeline and the web application

### Usage instructions

Run the following commands in the project's root directory to set up your database and model.

1. Reading in the data
To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Building the machine learning model
To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Starting the web application
Run the following command in the app's directory to run your web app.
`python run.py`

4. Opening the web application
Go to http://127.0.0.1:3001/

### Access via cloud

The web application has been deployed to the Heroku Cloud Application Platform. 
It can be called via the following URL:
https://my-drp-app.herokuapp.com/


