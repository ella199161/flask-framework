# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request
#from boto.s3.connection import S3Connection
import requests
import pandas as pd
import datetime
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from math import pi
from bokeh.models import DatetimeTickFormatter
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# create the application object
app = Flask(__name__)


# use decorators to link the function to a url
@app.route('/')
def main():
	return redirect(url_for('index'))

@app.route('/index')
def index():
	return render_template('index.html') 

@app.route('/model')
def model():
	return render_template('model.html') 


@app.route('/project')
def project():

	company = request.args.get('company')

	if company == "Prosper":
		clf = pickle.load(open("Prosper_toy3.pickle", "rb"))
	Lending_club_clf =1 
	term = request.args.get('term')
	home = request.args.get('home')
	fico = request.args.get('fico')
	amount = request.args.get('amount')
	total_acc = request.args.get('total_acc')
	#income = int(request.args.get('income'))
	balance = request.args.get('balance')
	Credit_L = request.args.get('Credit_L')
	interest = request.args.get('int_rate') 
	days = request.args.get('days') 
	
	#features = pd.Series([int(Credit_L), int(term), int(interest)*1.0/100, int(home), int(fico), int(total_acc)]).reshape(1,6)
	grade = request.args.get('grade') 
	if company and fico:
		features = pd.Series([int(Credit_L), int(term), int(interest)/100*0.1, int(home), int(fico), int(total_acc)])

		ypred = clf.predict(features.reshape(1,6))
		print(ypred)
	risk_dict = {"A": 3.983147797773583,"B": 8.3016729850008204,"C": 13.242679912878174, "D": 20.338340799622756, "E": 26.98618268861479, "F": 33.391248563256795, "G" :34.32795211318701}

	if fico:
		if ypred == 1:
			return render_template('project.html',status = {'code': 1, 'msg': "GOOD" } ) 
			
		else:
			return render_template('project.html',status = {'code': 1, 'msg': "BAD"} ) 
		
	elif days:
		#param = pickle.load(open("Lending_club_fit_param.pickle", "rb"))
		print(grade)
		safe = 100 -  risk_dict[grade]*int(days)/1095 *1.0

		return render_template('project.html',status = {'code': 3, 'msg': safe} ) 
	else:
		return render_template('project.html',status = {'code': 2, 'msg': 'Oops'} ) 

@app.route('/about')
def about():
    return render_template('about.html')  # render a template



# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)






