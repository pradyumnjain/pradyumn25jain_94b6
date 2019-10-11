from flask import Flask, render_template, request, flash, redirect
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
import csv

#Importing libraries for the algorithm
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/inspireme')
def inspireme():
	return render_template('inspireme.html')



if __name__=="__main__":
	app.run(debug=True)


