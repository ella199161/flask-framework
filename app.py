# import the Flask class from the flask module
from flask import Flask, render_template, redirect, \
    url_for, request, session, flash, g
from functools import wraps
import sqlite3
import requests
import pandas as pd
import datetime
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from math import pi
from bokeh.models import DatetimeTickFormatter

# create the application object
app = Flask(__name__)


# use decorators to link the function to a url
@app.route('/', methods = ['GET', 'POST'])
def main():
	return redirect(url_for('stock'))


#@login_required
@app.route('/stock', methods = ['GET', 'POST'])
def stock():
	stockticker = request.args.get('ticker')
	stockclose = request.args.get('close')
	stockopen = request.args.get('open')
	stockAclose = request.args.get('Aclose')
	stockAopen = request.args.get('Aopen')
	print('he is', stockticker, stockclose, stockopen ,stockAclose,stockAopen,'here')
	script = 0
	div = 0
	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()


	if stockticker:
		quandl = 'https://www.quandl.com/api/v3/datasets/WIKI/'
		timeURL = '&start_date=' + (datetime.datetime.now() - datetime.timedelta(days=30) ).strftime('%Y-%m-%d')+ '&end_date=' + (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
		my_token = '.json?api_key=z3Bsb_-YBr-VQVNsGcNn'
		all_url = quandl + stockticker + my_token + timeURL
		Respond = requests.get(all_url)  #data from web
		HTTP_status = Respond.status_code
		if HTTP_status == 200:
			rjson = Respond.json()['dataset']
			df = pd.DataFrame(rjson['data'], columns = rjson['column_names'])
			df['Date'] = pd.to_datetime(df['Date'])
			df['avg'] = (df['Open'] + df['Close'])/2
			df['range'] = abs(df['Close']-df['Open'])      		

			print(df.head())
			w = 12*60*60*1000 
#plot
			fig = figure(plot_width = 800, plot_height = 600, x_axis_type = "datetime", toolbar_location = "below", tools = "crosshair, pan, wheel_zoom, box_zoom, reset", title = stockticker + ' Graph')
			

			if stockclose == 'close':
				fig.line(df['Date'],df['Close'],color = 'black', legend = 'Close')

			if stockopen == 'open':
				fig.line(df['Date'], df['Open'], color = 'red', legend = 'Open')

			if stockAopen == 'Aopen':
				fig.circle(df['Date'],df['Adj. Open'],color = 'blue', legend = 'Adj. Open')

			if stockAclose == 'Aclose':
				fig.circle(df['Date'],df['Adj. Close'],color = 'green', legend = 'Adj. Close')

			fig.xaxis.formatter=DatetimeTickFormatter(hours=["%Y-%B-%d"], days=["%Y-%B-%d"], months=["%Y-%B-%d"], years=["%Y-%B-%d"])
			fig.xaxis.major_label_orientation = pi/4 
			js_resources = INLINE.render_js()
			css_resources = INLINE.render_css()  
			script, div = components(fig, INLINE)
		else:
		    return render_template('stock.html', status = {'code':2, 'msg':'Server Error'})
	
	else: 
		return render_template('stock.html', status = {'code': 3, 'message': 'Please input ticker symbol'}, js_resources=js_resources, css_resources=css_resources)  # render a template



	return render_template('stock.html', status = {'code': 1, 'message': 'Here is the plot'}, stock = {'ticker':stockticker}, plot={'script':script, 'div':div}, js_resources=js_resources, css_resources=css_resources)  # render a template


@app.route('/about')
def about():
    return render_template('about.html')  # render a template



# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)






