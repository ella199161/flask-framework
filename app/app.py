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
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from math import pi
from bokeh.models import DatetimeTickFormatter
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from urllib.request import urlopen, Request
import time
#from transformers import ColumnSelectTransformer, CreditTimeTransformer, NaInputeTransformer, EstimatorTransformer, ColumnUnSelectTransformer,OneColumnSelectTransformer,ReshapeTransformer,MyLabelEncoder

from datetime import date
import pandas as pd
import numpy as np
import os
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import timedelta, datetime
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn import base
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):   
            # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X[self.col_names]

class CreditTimeTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, credit_L, issued):
        self.credit_L = credit_L  # We will need these in transform()
        self.issued = issued
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):   
            # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        X[self.credit_L] = pd.to_datetime(X[self.credit_L])
        X[self.issued] = pd.to_datetime(X[self.issued])
        X[self.credit_L] = (X[self.issued] - X[self.credit_L]).apply(lambda x: x.days)      
        return X

class NaInputeTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, zerox_inp, mean_inp, maj_inp):
        self.zerox_inp = zerox_inp
        self.mean_inp = mean_inp
        self.maj_inp = maj_inp
        pass  
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):   
        for i in self.zerox_inp:
            X[i].fillna(0, inplace = True)
        for i in self.mean_inp:
            X[i].fillna(X[i].mean(skipna = True), inplace = True)
            
        for i in self.maj_inp:
            
            X_group = X.groupby(i).size()
            maj = X_group[X_group == X_group.max()].index[0]
            X[i].fillna(maj, inplace = True)
        return X
cata = ['addrstate', 'purpose', 'homeownership']
columns_keep = ['addrstate','annualinc','chargeoffwithin12mths','delinq2yrs', 'purpose',
                'delinqamnt', 'dti','earliestcrline','emplength','ficorangehigh','pubrec',
               'installment','intrate','loanamount', 'numoprevtl', 'numrevaccts',
       'homeownership','numbcsats','revolbal','revolutil','taxliens','totcurbal'
               ]
zerox_inp = ['chargeoffwithin12mths','delinq2yrs', 'delinqamnt','emplength',
             'numoprevtl', 'numrevaccts','numbcsats','pubrec','revolbal',
             'revolutil','taxliens','totcurbal'
            ]
mean_inp = ['ficorangehigh','annualinc','dti']
maj_inp = ['addrstate','purpose', 'homeownership', 'earliestcrline']

pipe = Pipeline([
    ('ctt', CreditTimeTransformer('earliestcrline', 'issued')),
    ('cst', ColumnSelectTransformer(columns_keep)),
    ('nit', NaInputeTransformer(zerox_inp, mean_inp, maj_inp))
])

class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator
        # What needs to be done here?
    
    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self
        # Fit the stored estimator.
        # Question: what should be returned?
    
    def transform(self, X):
        
        return self.X

class ColumnUnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):   
            # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X.drop(self.col_names, axis = 1)

class OneColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_name):
        self.col_name = col_name  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):   
            # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X[self.col_name].values

class MyLabelEncoder(LabelEncoder):
    def fit(self, X, y=None):
        super( MyLabelEncoder, self).fit(X)
        return self #
    def fit_transform(self, X, y=None):
        return super( MyLabelEncoder, self).fit_transform(X)

class ReshapeTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
         # We will need these in transform()
        pass
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):   
            # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X.reshape(-1, 1)

purpose_pipe = Pipeline([('cst1', OneColumnSelectTransformer('purpose')),
    ('lab', MyLabelEncoder()),
    ('rft', ReshapeTransformer()),                     
    ('ohe', OneHotEncoder(sparse=False))
                        ])

homeownership_pipe = Pipeline([('cst1', OneColumnSelectTransformer('homeownership')),
    ('lab', MyLabelEncoder()),
    ('rft', ReshapeTransformer()),                     
    ('ohe', OneHotEncoder(sparse=False))
                        ])


union =  FeatureUnion([
    ('cust', ColumnUnSelectTransformer(cata)),
    ('purpose_pipe', purpose_pipe),
    ('homeownership_pipe', homeownership_pipe)
   # ('addrstate_pipe', addrstate_pipe)
    ])

full_pipe = Pipeline([
    ('pipe_clean', pipe),
    ('pipe_enc', union),
    ('clf', LogisticRegression(class_weight='balanced'))
])






# create the application object
app = Flask(__name__)
pipeA = pickle.load(open("pipeA.pickle", "rb"))
pipeB = pickle.load(open("pipeB.pickle", "rb"))
pipeC = pickle.load(open("pipeC.pickle", "rb"))
pipeD = pickle.load(open("pipeD.pickle", "rb"))
pipeE = pickle.load(open("pipeE.pickle", "rb"))
pipeF = pickle.load(open("pipeF.pickle", "rb"))
pipeG = pickle.load(open("pipeG.pickle", "rb"))
agree = pickle.load(open("agree.pickle", "rb"))

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


	Listing = pd.read_csv('Listing_2d.csv')
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



	grade = request.args.get('grade')
	company = request.args.get('company')
	days = request.args.get('days') 
	#company = "Lending Club"
	if company == "Lending Club" or company == "Lending club":
		Lending_p = pickle.load(open('Lending_club_fit_param.pickle', 'rb'))
		para_map = {'A':0 , 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
		xx = np.arange(3*365)
		a,b,c = Lending_p[para_map[grade]]
		yfit = a * np.exp(-b * xx + c)
		yy = [min(1, i) for i in yfit]
		y_end = a * np.exp(-b * 3*365 + c) 
		y_safe = y_end/yy*100
		x_pos = int(days)
		y_pos = a * np.exp(-b * x_pos + c)
		py_pos = y_end/y_pos*100
		data = {'days': xx,
		       'Pay_Off_Prob': y_safe,
		       'yy': yy}
		source = ColumnDataSource(data=data)
		hover = HoverTool(tooltips = [
		    ("days", "@days"),
		    ("Pay Off Prob", "@Pay_Off_Prob %"),
		] )
		p = figure(plot_width=600,x_range=(0, 1095),y_range=(0, 1.05), plot_height=400, tools= [hover], toolbar_location='left',title="Loan Survival Prob: {}%".format(py_pos))
		p.circle(x_pos, y_pos, size=20, color="navy", alpha=0.5)
		p.line("days",'yy',line_width = 2,source = source)
		p.xaxis.axis_label_text_font_size = '16pt'
		p.yaxis.axis_label_text_font_size = '16pt'
		p.xaxis.major_label_text_font_size="14pt"
		p.xaxis.axis_label = 'Days'
		p.yaxis.axis_label = 'Pay Off Prob%'		
		script, div = components(p, INLINE)
		return render_template('project.html',status = {'code': 1, 'msg': 'good'},  plot = {'script':script, 'div':div})

	return render_template('project.html',status = {'code': 3, 'msg': 'Please Enter Valid info'} ) 


	

@app.route('/about')
def about():
    return render_template('about.html')  # render a template

@app.route('/listing')
def listing():
	risk = request.args.get('risk')

	Listing = pd.read_csv('Listing.csv')
	Listing['issued'] = date.today()
	List_col = [i.lower() for i in Listing.columns.values]
	Listing.columns = List_col
	cols_to_show = ['id', 'grade', 'intrate','loanamount', 'annualinc', 'dti', 'ficorangehigh']
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
		if len(ListingX) == 0:
			return render_template('listing.html', status = {'code': 2, 'msg': 'No current listing at this risk! Try again later!!'} )
		y_pred = pipeX.predict(ListingX)
		showX = Listing_All[Listing_All['grade'] == risk]
		goodX = showX[[bool(i) for i in y_pred]]
		badX = showX[[not bool(i) for i in y_pred]]
		filename = 'Listing ' + str(date.today())
		return render_template('listing.html',status = {'code': 1, 'msg': 'A Listing'} ,  name=filename, data_all=showX, data_good = goodX,data_bad = badX)

	else:





		return render_template('listing.html',status = {'code': 0, 'msg': 'All Listing'}  )




# start the server with the 'run()' method
if __name__ == '__main__':
	import os
	remote = os.environ.get('REMOTE', False)
	if not remote:
		app.run(debug=True)
	else:
		app.run(debug=True, host='0.0.0.0')

"""pipeA = pickle.load(open("pipeA.pickle", "rb"))		
		ListingA = Listing[Listing['grade'] == 'A']
		y_predA = pipeA.predict(ListingA)
		showA = Listing_All[Listing_All['grade'] == 'A']
		goodA = showA[[bool(i) for i in y_predA]]
		badA = showA[[not bool(i) for i in y_predA]]
		filename = 'Listing ' + str(date.today())
		return render_template('listing.html',status = {'code': 1, 'msg': 'A Listing'} ,  name=filename, data_all=showA.to_html(), data_good = goodA.to_html(),data_bad = badA.to_html())
"""


