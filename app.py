from flask import Flask, render_template, request, flash, redirect
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
import csv

#Importing libraries for the algorithm
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db= SQLAlchemy(app)
# use DB-Browser to modify the database


class questions_table(db.Model):
    question_number = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500), unique=True, nullable=False)
    #answer type= 'm' for mcq questions, 's' for slider input
    answer_type = db.Column(db.String(1), nullable=False)
    #options 1 and 2 can also act as the arguments for slider's min and max values
    option_1_range= db.Column(db.String(50),nullable=False)
    option_2_range= db.Column(db.String(50),nullable=False)
    option_3= db.Column(db.String(50))
    option_1_value=db.Column(db.Integer,nullable=False)
    option_2_value=db.Column(db.Integer,nullable=False)
    option_3_value=db.Column(db.Integer,nullable=False)



@app.route('/')
def index():
	return render_template('index.html')



if __name__=="__main__":
	app.run(debug=True)

#trial pull
