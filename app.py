from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms.fields.html5 import DecimalRangeField

from wtforms import TextField, IntegerField, TextAreaField, SubmitField, RadioField, SelectField
from wtforms import validators, ValidationError
from flask_sqlalchemy import SQLAlchemy
import csv

#Importing libraries for the algorithm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
#Importing libraries for the algorithm
import numpy as np
import pandas as pd
import config as cfg

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.secret_key = 'development key'
db= SQLAlchemy(app)
# use DB-Browser to modify the database

final_output_array=[]

def all_questions_answered(no_of_rows):
    if(no_of_rows==cfg.qno):
        return True
    else:
        return False

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

class QuestionForm(FlaskForm):
    a=questions_table.query.get(cfg.qno)
    slider = DecimalRangeField('Slide the slider!', default=0)
    Options = RadioField('Options', choices = [(float(a.option_1_value) ,a.option_1_range),(float(a.option_2_value) ,a.option_2_range),(float(a.option_3_value) , a.option_3)])
    # Options = RadioField('Options', choices = [('1' ,all_options[qno][0]),('2' ,all_options[qno][1]),('3' ,all_options[qno][2])] )
    submit = SubmitField("Send")



@app.route('/')
def home():
	return render_template('index.html', content=render_template('pages/home.html'))

@app.route('/about')
def about():
	return render_template('pages/about.html')

@app.route('/explore')
def explore():
	return render_template('index.html', content=render_template('pages/explore.html'))

@app.route('/contact')
def contact():
	return render_template('index.html', content=render_template('pages/contact.html'))

@app.route('/inspire')
def inspire():
	return render_template('pages/inspire.html')

@app.route('/output')
def output():
    cfg.traits=cfg.answers[0:cfg.no_of_trait_questions]
    del cfg.answers[0:cfg.no_of_trait_questions]
    final_output_array= recommendation_algorithm(cfg.traits, cfg.answers)
    return render_template('output.html', p1= final_output_array[0], p2=final_output_array[1], p3=final_output_array[2])

@app.route('/tv',  methods=['GET','POST'])
def tv():
    form = QuestionForm()
    cfg.no_of_rows= questions_table.query.count()
    question= questions_table.query.get(cfg.qno).question
    a=questions_table.query.get(cfg.qno)
    min=a.option_1_range
    max=a.option_2_range

    if request.method == 'POST':
 #known issue- no validation if no rb is selected
        # Render the same screen when no radio button is selected
        # if form.validate() == False:
        if(False):
            print("can't validate")
            return render_template('index.html', content=render_template('pages/tv.html', form=form, question=question, flag=a.answer_type, min_=min, max_=max))


        #Render output page if all questions are answered
        elif(all_questions_answered(cfg.no_of_rows)):
        	return redirect('/output')


        else:
            #re-initializing the form everytime to update options
            if(a.answer_type=='m'):
                cfg.answers.append(float(form.Options.data))
            elif(a.answer_type=='s'):
                cfg.answers.append(float(form.slider.data))
            else:
                print("error")
            if(all_questions_answered(cfg.no_of_rows)):
                return redirect('/output')

            cfg.qno=cfg.qno+1
            #question refresh, code to be optimized
            question= questions_table.query.get(cfg.qno).question


            #render the mcq choices here
            a=questions_table.query.get(cfg.qno)
            if(a.answer_type=='m'):

                form.Options.choices = [(float(a.option_1_value) ,a.option_1_range),(float(a.option_2_value) ,a.option_2_range),(float(a.option_3_value) , a.option_3)]
            elif(a.answer_type=='s'):
                min=a.option_1_range
                max=a.option_2_range



            # form.Options.choices=[('1',all_options[qno][0]),('2' ,all_options[qno][1]),('3',all_options[qno][2])]
            return render_template('index.html', content=render_template('pages/tv.html', form=form, question=question, flag=a.answer_type, min_=min, max_=max))

    elif request.method == 'GET':
        # this part runs for the first question
        return render_template('index.html', content=render_template('pages/tv.html', form = form, question=question, flag=a.answer_type,min_=min, max_=max))


@app.route('/buy')
def buy():
    return render_template('pages/buynow.html')

if __name__=="__main__":
	app.run(debug=True, use_reloader=True)
