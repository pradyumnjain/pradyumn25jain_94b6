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
import multiple_page_web_crawler as crawler

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
	return render_template('pages/home.html')

@app.route('/about')
def about():
	return render_template('pages/about.html')

@app.route('/explore')
def explore():
	return redirect(url_for('buy'))

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
    final_output_array= recommendation_algorithm(df, cfg.traits, cfg.answers)
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

"""
ALL THE DATA WORK IS BELOW THIS.
FEEL FREE TO ADJUST ACCORDING TO THE WEBSITE REQUIREMENTS
TODO: BETTER VARIABLE NAMES, OPTIMIZATION, INCLUDE ALL DATASETS, REFACTOR CHUNKS OF CODE
"""

NUMBER_OF_PERSONALITY_QUESTIONS = 5
#Let me test for tv dataset first
df = pd.read_csv("data/final_phone.csv", skipinitialspace=True)

df_real_time = crawler.run()

#Data preprocessing
df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop("weight_oz", axis=1, inplace=True)
df.drop('GPS', axis=1, inplace=True)
df.drop('memory_card', axis=1, inplace=True)
df.drop('battery_removable', axis=1, inplace=True)
df.drop('length_mm', axis=1, inplace=True)
df.drop('width_mm', axis=1, inplace=True)
df.drop('gprs', axis=1, inplace=True)
df.drop('edge', axis=1, inplace=True)
df.dropna(inplace=True)

PRIMARY_COL_NAME = "model"
X = df[[x for x in df.columns if x!=PRIMARY_COL_NAME]]

#Convert all the dtypes into float
X = X.astype('float64')

#normalize the dataset
X = (X - X.max())/(X.max() - X.min())

#Create the kmeans model
kmeans_model = KMeans(n_clusters=NUMBER_OF_PERSONALITY_QUESTIONS)

#Train
kmeans_model.fit(X)

#Predict on the same data i.e basically make clusters for the dataset
y_kmeans = kmeans_model.predict(X)

#Make centroids
centroid_positions = kmeans_model.cluster_centers_
new_X_centroids = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
#Setting values randomly for now but has to be evaluated by seeing the products
new_X_centroids[0,0] = 1
new_X_centroids[1,1] = 1
new_X_centroids[2,2] = 1
new_X_centroids[3,3] = 1
new_X_centroids[4,4] = 1

#Calculating P value for centroids
P_centroids = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
for i in range(len(centroid_positions)):
    disti = []
    for j in range(len(centroid_positions)):
        disti.append(sum(np.sqrt((centroid_positions[i] - centroid_positions[j])**2)))
    disti /= (max(disti)+np.average(disti))
    disti = 1 - disti
    P_centroids[i] = np.array(disti)

#Calculate distance for all the data points to calculate P value
P_dist = np.zeros(len(y_kmeans))
for ind, val in enumerate(y_kmeans):
    d = sum(np.sqrt((centroid_positions[val] - X.iloc[ind].values)**2))
    P_dist[ind] = d

#I am sure there is a better way to do it
def get_indices(arr, x):
    indices = []
    for i in range(len(arr)):
        if arr[i]==x:
            indices.append(i)
    return indices
P_X = np.zeros((len(y_kmeans), NUMBER_OF_PERSONALITY_QUESTIONS))

#Calculate P value for all the data points
#i is the personality
#j is the centroid's personality
for i in range(NUMBER_OF_PERSONALITY_QUESTIONS):
    for j in range(NUMBER_OF_PERSONALITY_QUESTIONS):
        P_X[get_indices(y_kmeans, i), j] = P_centroids[i,j] - (P_centroids[i,j]/max(P_dist[get_indices(y_kmeans, i)]))*P_dist[get_indices(y_kmeans, i)]

#Creating random forest regressor to perform supervised learning
#This will basically generate mapping between the features and personality
rfr_model = RandomForestRegressor(max_depth=30)
rfr_model.fit(P_X, X)

def recommendation_algorithm(dataframe, personality_answers_array, tech_answers_array):
    # Make sure to return a list/array object or anything else and changes on the @output route accordingly
    predictedX_features = rfr_model.predict([personality_answers_array])

    #This is the rule for deciding on how to transition to feature question set
    #For now I am choosing closest 20 points
    closest = np.argsort(np.sum((X.values - predictedX_features)**2, axis=1))[:20]

    #After getting the predicted feature vector we can rank the things
    predicted_products = []
    closest_for_fx = np.argsort(np.sum((X.values - tech_answers_array)**2, axis=1))[:3]

    #The datatype is the series, FINAL OUTPUT
    recommended_products = dataframe.iloc[closest_for_fx][PRIMARY_COL_NAME]

    return recommended_products.to_list()



def all_questions_answered(no_of_rows):
    if(no_of_rows==cfg.qno):
        return True
    else:
        return False

if __name__=="__main__":
	app.run(debug=True, use_reloader=True)
