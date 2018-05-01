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
from urllib.request import urlopen, Request
import time
from transformers import *
from datetime import date
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

#headers = {'Authorization' : 'p96JwpSHaxLH5U4PFPwcYWv9ZhY='}
#result = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', headers = headers)
#Listing = result.json()
#Listing = pd.DataFrame(Listing['loans'])

	pipeA = pickle.load(open("pipeA.pickle", "rb"))
	pipeB = pickle.load(open("pipeB.pickle", "rb"))
	pipeC = pickle.load(open("pipeC.pickle", "rb"))
	pipeD = pickle.load(open("pipeD.pickle", "rb"))
	pipeE = pickle.load(open("pipeE.pickle", "rb"))
	pipeF = pickle.load(open("pipeF.pickle", "rb"))
	pipeG = pickle.load(open("pipeG.pickle", "rb"))
	agree = pickle.load(open("agree.pickle", "rb"))
	Listing = pd.read_csv('Listing.csv')
	Listing['issued'] = date.today()
	List_col = [i.lower() for i in Listing.columns.values]
	Listing.columns = List_col
	cols_to_show = ['id', 'grade', 'annualinc', 'dti', 'delinq2yrs','earliestcrline','ficorangehigh','loanamount', 'intrate']
	Listing_show = Listing[cols_to_show]

	Listing = Listing[agree]
	ListingA = Listing[Listing['grade'] == 'A']
	ListingB = Listing[Listing['grade'] == 'B']
	ListingC = Listing[Listing['grade'] == 'C']
	ListingD = Listing[Listing['grade'] == 'D']
	ListingE = Listing[Listing['grade'] == 'E']
	ListingF = Listing[Listing['grade'] == 'F']
	ListingG = Listing[Listing['grade'] == 'G']




	company = request.args.get('company')

	if company == "Prosper":
		clf = pickle.load(open("Prosper_toy3.pickle", "rb"))
	grade = request.args.get('grade') 
	days = request.args.get('days') 

	filename = 'Listing ' + str(date.today())
	new_loan = request.args.get('new_loan')
	print(new_loan)
	if new_loan:
		return render_template('project.html',status = {'code': 1, 'msg': 'New Listing'}, name=filename, data=Listing.to_html() ) 


	"""Lending_club_clf =1 
	term = request.args.get('term')
	home = request.args.get('home')
	fico = request.args.get('fico')
	amount = request.args.get('amount')
	total_acc = request.args.get('total_acc')
	#income = int(request.args.get('income'))
	balance = request.args.get('balance')
	Credit_L = request.args.get('Credit_L')
	interest = request.args.get('int_rate') 

	
	#features = pd.Series([int(Credit_L), int(term), int(interest)*1.0/100, int(home), int(fico), int(total_acc)]).reshape(1,6)

	if company and fico:
		features = pd.Series([int(Credit_L), int(term), int(interest)/100*0.1, int(home), int(fico), int(total_acc)])

		ypred = clf.predict(features.reshape(1,6))
		print(ypred)
	
	if fico:
		if ypred == 1:
			return render_template('project.html',status = {'code': 1, 'msg': "GOOD" } ) 
			
		else:
			return render_template('project.html',status = {'code': 1, 'msg': "BAD"} ) 
	"""	

	risk_dict = {"A": 3.983147797773583,"B": 8.3016729850008204,"C": 13.242679912878174, "D": 20.338340799622756, "E": 26.98618268861479, "F": 33.391248563256795, "G" :34.32795211318701}

	if days:
		#param = pickle.load(open("Lending_club_fit_param.pickle", "rb"))
		print(grade)
		safe = 100 -  risk_dict[grade]*(1095-int(days))/1095 *1.0

		return render_template('project.html',status = {'code': 3, 'msg': safe} ) 
	else:
		return render_template('project.html',status = {'code': 2, 'msg': 'Oops'} ) 

@app.route('/about')
def about():
    return render_template('about.html')  # render a template

@app.route('/listing')
def listing():
	risk = request.args.get('risk')

	pipeB = pickle.load(open("pipeB.pickle", "rb"))
	pipeC = pickle.load(open("pipeC.pickle", "rb"))
	pipeD = pickle.load(open("pipeD.pickle", "rb"))
	pipeE = pickle.load(open("pipeE.pickle", "rb"))
	pipeF = pickle.load(open("pipeF.pickle", "rb"))
	pipeG = pickle.load(open("pipeG.pickle", "rb"))
	agree = pickle.load(open("agree.pickle", "rb"))
	Listing = pd.read_csv('Listing.csv')
	Listing['issued'] = date.today()
	List_col = [i.lower() for i in Listing.columns.values]
	Listing.columns = List_col
	cols_to_show = ['id', 'grade', 'annualinc', 'dti', 'delinq2yrs','earliestcrline','ficorangehigh','loanamount', 'intrate']
	Listing_show = Listing[cols_to_show]

	Listing = Listing[agree]
	ListingA = Listing[Listing['grade'] == 'A']
	ListingB = Listing[Listing['grade'] == 'B']
	ListingC = Listing[Listing['grade'] == 'C']
	ListingD = Listing[Listing['grade'] == 'D']
	ListingE = Listing[Listing['grade'] == 'E']
	ListingF = Listing[Listing['grade'] == 'F']
	ListingG = Listing[Listing['grade'] == 'G']
	filename = 'Listing ' + str(date.today())
	Listing_All = Listing[cols_to_show]
	
	if risk in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		to_load = 'pipe' + risk +'.pickle'
		pipeX = pickle.load(open(to_load, "rb"))		
		ListingX = Listing[Listing['grade'] == risk]
		y_pred = pipeX.predict(ListingX)
		showX = Listing_All[Listing_All['grade'] == risk]
		goodX = showX[[bool(i) for i in y_pred]]
		badX = showX[[not bool(i) for i in y_pred]]
		filename = 'Listing ' + str(date.today())
		return render_template('listing.html',status = {'code': 1, 'msg': 'A Listing'} ,  name=filename, data_all=showX, data_good = goodX.to_html(),data_bad = badX.to_html())

	else:





		return render_template('listing.html',status = {'code': 0, 'msg': 'All Listing'} ,  name=filename, data=Listing_show.to_html() )



# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)



"""pipeA = pickle.load(open("pipeA.pickle", "rb"))		
		ListingA = Listing[Listing['grade'] == 'A']
		y_predA = pipeA.predict(ListingA)
		showA = Listing_All[Listing_All['grade'] == 'A']
		goodA = showA[[bool(i) for i in y_predA]]
		badA = showA[[not bool(i) for i in y_predA]]
		filename = 'Listing ' + str(date.today())
		return render_template('listing.html',status = {'code': 1, 'msg': 'A Listing'} ,  name=filename, data_all=showA.to_html(), data_good = goodA.to_html(),data_bad = badA.to_html())
"""


