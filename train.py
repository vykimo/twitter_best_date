#!/usr/bin/env python
# encoding: utf-8
import numpy
import random
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
import argparse
import pandas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import calendar

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f, --file', dest='file', action='store', help='file with twitter accounts')
    return parser.parse_args()


def label_to_mins(label):
	[hour, min] = label.split(":")	
	return int(int(min) + 60*int(hour))
	
def mins_to_label(mins):
	if mins < 139440:
		hour = str(int(mins/60))
		if hour < 10:
			hour = "0" + hour 
		min = str(int(mins%60))
		if min < 10:
			min = "0" + min
	else:
		hour = "00"
		min = "00"
	return ":".join([hour, min])

def build_hashtag_vector(hashtags, lower = 0):
	h = dict()
	i = 0
	for hashtag in hashtags:
		if not hashtag.lower() in h:
			h[hashtag.lower()] = i
			i += 1
	return h

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
def main(run_args):
	if run_args.file:
		# load dataset
		dataframe = pandas.read_csv(run_args.file.strip('\n'), header=None, encoding = "ISO-8859-1")
		dataset = dataframe.values
		hashtags = []
		for h in dataset[:,0]:
			hashtags.extend([h.lower()])
		hashtag_vector = build_hashtag_vector(hashtags,1)
		
		print("Num of Hashtag datas : "+str(len(dataset[:,0])))
		print("Num of unique Hashtags : "+str(len(hashtag_vector)))
		
		# split into input (X) and output (Y) variables
		X = dataset[:,0:3]
		for i in range(0,len(X)):
			# remplace hashtags by vector
			X[i,0] = hashtag_vector[X[i,0].lower()]
			# transform dayhour in minutes
			X[i,2] = label_to_mins(X[i,2])
		Y = dataset[:,3]
		
		# Split data
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
		
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		svr_rbf.fit(X, Y)
		
		
		# constant result
		#seed = 7
		#numpy.random.seed(seed)
		#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
		#kfold = KFold(n_splits=10, random_state=seed)
		#results = cross_val_score(estimator, X, Y, cv=kfold)
		#y_pred = cross_val_predict(estimator, X_test, y_test)
		
		
		#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
		days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":0}
		while 1:
			doc = input("\n============\nWhat hashtag do you want to test? for instance '"+hashtags[random.randint(0, len(hashtags))]+"'\n").lower()
			if doc in hashtag_vector:
				day_label = input("Day of the week?\n")
				day = days[day_label]
				if day >= 0 and day <= 6:
					hour_label = input("Hour of the day? format HH:MM\n").strip('\n')
					hour = label_to_mins(hour_label)
					
					print("\nScore for '"+ str(doc) +"' , Day : "+ str(calendar.day_name[day+1])+", Hour : "+ str(hour_label) +" = "+ str(int(svr_rbf.predict([[hashtag_vector[doc], day, hour]]))))
			else:
				print("Hashtag not found")
		
		
		
		# 1. define the network
		#model = Sequential()
		#model.add(Dense(12, input_dim=3, activation='relu'))
		#model.add(Dense(1, activation='linear'))
		# 2. compile the network
		#model.compile(loss='mean_squared_error', optimizer='sgd')
		# 3. fit the network
		#model.fit(X_train, y_train, epochs=10, batch_size=50)
		# 4. evaluate the network
		#score = model.evaluate(X_test, y_test, verbose=1)
		#print(score)
		#y_pred = model.predict(X_test)
		#for i in range(0,5):
			#print(str(y_pred[i])+" : "+str(y_test[i]))
		

		
if __name__ == "__main__":
    args = parse_arguments()
    main(args)