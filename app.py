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
    option_1_range= db.Column(db.String(300),nullable=False)
    option_2_range= db.Column(db.String(300),nullable=False)
    option_3= db.Column(db.String(300))
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
    clear_all_selections()
    return render_template('pages/home.html')

@app.route('/about')
def about():
    clear_all_selections()
    return render_template('pages/about.html')

@app.route('/explore')
def explore():
    clear_all_selections()
    return redirect(url_for('buy'))

@app.route('/contact')
def contact():
    clear_all_selections()
    return render_template('index.html', content=render_template('pages/contact.html'))

@app.route('/inspire')
def inspire():
    clear_all_selections()
    return render_template('pages/inspire.html')

@app.route('/output')
def output():
    cfg.traits=cfg.answers[0:cfg.no_of_trait_questions]
    del cfg.answers[0:cfg.no_of_trait_questions]
    print(cfg.traits, cfg.answers)
    final_output_array= recommendation_algorithm(df_tv, cfg.traits, cfg.answers)
    return render_template('output.html', p1= final_output_array[0], p2=final_output_array[1], p3=final_output_array[2])

@app.route('/tv',  methods=['GET','POST'])
def tv():
    form = QuestionForm()
    cfg.no_of_rows= questions_table.query.count()
    question= questions_table.query.get(cfg.qno).question
    a=questions_table.query.get(cfg.qno)
    min=a.option_1_value
    max=a.option_2_value
    if request.method == 'POST':
 #known issue- no validation if no rb is selected
        # Render the same screen when no radio button is selected
        # if form.validate() == False:
        if(False):
            print("can't validate")
            return render_template('pages/tv.html', form=form, question=question, flag=a.answer_type, min_=min, max_=max)

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
            question = questions_table.query.get(cfg.qno).question


            #render the mcq choices here
            a=questions_table.query.get(cfg.qno)
            if(a.answer_type=='m'):

                form.Options.choices = [(float(a.option_1_value) ,a.option_1_range),(float(a.option_2_value) ,a.option_2_range),(float(a.option_3_value) , a.option_3)]
            elif(a.answer_type=='s'):
                min=a.option_1_value
                max=a.option_2_value
            # form.Options.choices=[('1',all_options[qno][0]),('2' ,all_options[qno][1]),('3',all_options[qno][2])]
            return render_template('pages/tv.html', form=form, question=question, flag=a.answer_type, min_=min, max_=max)

    elif request.method == 'GET':
        # this part runs for the first question
        return render_template('pages/tv.html', form = form, question=question, flag=a.answer_type,min_=min, max_=max)


@app.route('/buy')
def buy():
    return render_template('pages/buynow.html')

"""
ALL THE DATA WORK IS BELOW THIS.
FEEL FREE TO ADJUST ACCORDING TO THE WEBSITE REQUIREMENTS
TODO: BETTER VARIABLE NAMES, OPTIMIZATION, INCLUDE ALL DATASETS, REFACTOR CHUNKS OF CODE
"""
#I am sure there is a better way to do it
def get_indices(arr, x):
    indices = []
    for i in range(len(arr)):
        if arr[i]==x:
            indices.append(i)
    return indices

NUMBER_OF_PERSONALITY_QUESTIONS = 3
#Let me test for tv dataset first
df_phone = pd.read_csv("data/final_phone.csv", skipinitialspace=True)
df_tv = crawler.run()

#Data preprocessing
df_phone.drop("Unnamed: 0", axis=1, inplace=True)
df_phone.drop("weight_oz", axis=1, inplace=True)
df_phone.drop('GPS', axis=1, inplace=True)
df_phone.drop('memory_card', axis=1, inplace=True)
df_phone.drop('battery_removable', axis=1, inplace=True)
df_phone.drop('length_mm', axis=1, inplace=True)
df_phone.drop('width_mm', axis=1, inplace=True)
df_phone.drop('gprs', axis=1, inplace=True)
df_phone.drop('edge', axis=1, inplace=True)
df_phone.dropna(inplace=True)

df_tv.drop("usb_port", axis=1, inplace=True)
df_tv.drop("hdmi_port", axis=1, inplace=True)
df_tv.drop("type", axis=1, inplace=True)
df_tv.drop("refresh_rate ", axis=1, inplace=True)
df_tv.drop("Rating", axis=1, inplace=True)
df_tv.drop("res1", axis=1, inplace=True)
df_tv.drop("review", axis=1, inplace=True)

df_tv.dropna(inplace=True)

PRIMARY_COL_NAME_PHONE = "model"
PRIMARY_COL_NAME_TV = "product_name"
X_phone = df_phone[[x for x in df_phone.columns if x!=PRIMARY_COL_NAME_PHONE]]
X_tv = df_tv[[x for x in df_tv.columns if x!=PRIMARY_COL_NAME_TV]]

#Convert all the dtypes into float
X_phone = X_phone.astype('float64')
X_tv = X_tv.astype('float64')

#normalize the dataset
X_phone = (X_phone - X_phone.max())/(X_phone.max() - X_phone.min())
X_tv = (X_tv - X_tv.max())/(X_tv.max() - X_tv.min())

#Create the kmeans model
kmeans_model_phone = KMeans(n_clusters=NUMBER_OF_PERSONALITY_QUESTIONS)
kmeans_model_tv = KMeans(n_clusters=NUMBER_OF_PERSONALITY_QUESTIONS)

#Train
kmeans_model_phone.fit(X_phone)
kmeans_model_tv.fit(X_tv)

#Predict on the same data i.e basically make clusters for the dataset
y_kmeans_phone = kmeans_model_phone.predict(X_phone)
y_kmeans_tv = kmeans_model_tv.predict(X_tv)
#Make centroids
centroid_positions_phone = kmeans_model_phone.cluster_centers_
new_X_centroids_phone = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
#Setting values randomly for now but has to be evaluated by seeing the products
# new_X_centroids_phone[0,0] = 1
# new_X_centroids_phone[1,1] = 1
# new_X_centroids_phone[2,2] = 1
# new_X_centroids_phone[3,3] = 1
# new_X_centroids_phone[4,4] = 1

centroid_positions_tv = kmeans_model_tv.cluster_centers_
# new_X_centroids_tv = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
# new_X_centroids_tv[0,0] = 1
# new_X_centroids_tv[1,1] = 1
# new_X_centroids_tv[2,2] = 1
# new_X_centroids_tv[3,3] = 1
# new_X_centroids_tv[4,4] = 1

#Calculating P value for centroids
P_centroids_phone = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
for i in range(len(centroid_positions_phone)):
    disti = []
    for j in range(len(centroid_positions_phone)):
        disti.append(sum(np.sqrt((centroid_positions_phone[i] - centroid_positions_phone[j])**2)))
    disti /= (max(disti)+np.average(disti))
    disti = 1 - disti
    P_centroids_phone[i] = np.array(disti)
P_centroids_tv = np.zeros((NUMBER_OF_PERSONALITY_QUESTIONS,NUMBER_OF_PERSONALITY_QUESTIONS))
for i in range(len(centroid_positions_tv)):
    disti = []
    for j in range(len(centroid_positions_tv)):
        disti.append(sum(np.sqrt((centroid_positions_tv[i] - centroid_positions_tv[j])**2)))
    disti /= (max(disti)+np.average(disti))
    disti = 1 - disti
    P_centroids_tv[i] = np.array(disti)

#Calculate distance for all the data points to calculate P value
P_dist_phone = np.zeros(len(y_kmeans_phone))
for ind, val in enumerate(y_kmeans_phone):
    d = sum(np.sqrt((centroid_positions_phone[val] - X_phone.iloc[ind].values)**2))
    P_dist_phone[ind] = d
P_dist_tv = np.zeros(len(y_kmeans_phone))
for ind, val in enumerate(y_kmeans_tv):
    d = sum(np.sqrt((centroid_positions_tv[val] - X_tv.iloc[ind].values)**2))
    P_dist_tv[ind] = d

P_X_phone = np.zeros((len(y_kmeans_phone), NUMBER_OF_PERSONALITY_QUESTIONS))
P_X_tv = np.zeros((len(y_kmeans_tv), NUMBER_OF_PERSONALITY_QUESTIONS))
#Calculate P value for all the data points
#i is the personality
#j is the centroid's personality
for i in range(NUMBER_OF_PERSONALITY_QUESTIONS):
    for j in range(NUMBER_OF_PERSONALITY_QUESTIONS):
        P_X_phone[get_indices(y_kmeans_phone, i), j] = P_centroids_phone[i,j] - (P_centroids_phone[i,j]/max(P_dist_phone[get_indices(y_kmeans_phone, i)]))*P_dist_phone[get_indices(y_kmeans_phone, i)]
for i in range(NUMBER_OF_PERSONALITY_QUESTIONS):
    for j in range(NUMBER_OF_PERSONALITY_QUESTIONS):
        P_X_tv[get_indices(y_kmeans_tv, i), j] = P_centroids_tv[i,j] - (P_centroids_tv[i,j]/max(P_dist_tv[get_indices(y_kmeans_tv, i)]))*P_dist_tv[get_indices(y_kmeans_tv, i)]

#Creating random forest regressor to perform supervised learning
#This will basically generate mapping between the features and personality
rfr_model_phone = RandomForestRegressor(max_depth=30)
rfr_model_phone.fit(P_X_phone, X_phone)
rfr_model_tv = RandomForestRegressor(max_depth=30)
rfr_model_tv.fit(P_X_tv, X_tv)

def recommendation_algorithm(dataframe, personality_answers_array, tech_answers_array):
	tech_answers_array[2]*=100
	# Make sure to return a list/array object or anything else and changes on the @output route accordingly
	predictedX_features = rfr_model_tv.predict([personality_answers_array])

	#This is the rule for deciding on how to transition to feature question set
	#For now I am choosing closest 20 points
	closest = np.argsort(np.sum((X_tv.values - predictedX_features)**2, axis=1))[:40]

	#After getting the predicted feature vector we can rank the things
	predicted_products = []
	closest_for_fx = np.argsort(np.sum((X_tv.values - tech_answers_array)**2, axis=1))[:3]

	#The datatype is the series, FINAL OUTPUT
	recommended_products = dataframe.iloc[closest_for_fx][PRIMARY_COL_NAME_TV]

	return recommended_products.to_list()

def clear_all_selections():

    cfg.qno=1

    del cfg.answers[:]
    cfg.trait_collection_finished=False

    del cfg.traits[:]



if __name__=="__main__":
	app.run(debug=True, use_reloader=True)
