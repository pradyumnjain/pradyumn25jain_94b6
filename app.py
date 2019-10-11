from flask import Flask, render_template, request, flash, redirect
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
import csv

#Importing libraries for the algorithm
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('defaults/layout.html', content=render_template('pages/about.html'))

@app.route('/explore')
def explore():
	return render_template('defaults/layout.html', content=render_template('pages/explore.html'))

@app.route('/contact')
def contact():
	return render_template('defaults/layout.html', content=render_template('pages/contact.html'))

if __name__=="__main__":
	app.run(debug=True, use_reloader=True)
