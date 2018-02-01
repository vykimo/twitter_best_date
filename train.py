#!/usr/bin/env python
# encoding: utf-8
import numpy
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from baselines import baselines
import calendar
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f, --file', dest='file', type=open, help='file with twitter accounts')
    parser.add_argument('-s, --skip', dest='skip', action='store_true', help='skip tweets with no hashtags')
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
	
def main(run_args):
	if run_args.file:
	
		# load dataset
		datas = json.load(run_args.file)
		
		if run_args.skip:
			# Delete empty hashtags
			print("Before cleaning tweets without hashtags : "+str(len(datas)))
			datas = [row for row in datas if len(row['hashtag'])>0]
			print("After cleaning tweets without hashtags : "+str(len(datas)))
			
		# Split data
		#train_size = int(len(datas) * 0.66)
		#train, test = datas[1:train_size], datas[train_size:]
		train, test = train_test_split(datas, test_size=0.33, shuffle=True, random_state=42)
		test_y, train_y = [row['score'] for row in test], [row['score'] for row in train]
		test_X, train_X = [["-".join(row['hashtag']), row['weekday'], row['hour']] for row in test], [["-".join(row['hashtag']), row['weekday'], row['hour']] for row in train]
		
		hashtags = []
		for data in datas:
			composition_h = "-".join(data['hashtag'])
			hashtags.extend([composition_h.lower()])
		hashtag_vector = build_hashtag_vector(hashtags,1)
		
		print("Num of Hashtag datas : "+str(len(datas)))
		print("Num of unique Hashtags : "+str(len(hashtag_vector)))
		
		# baselines
		predictions1 = baselines.overall_mean(train,test)
		test_score1 = mean_squared_error(test_y, predictions1)
		print('=Baseline 1 = Test MSE: %.3f' % test_score1)
		
		predictions2 = baselines.maximum_mean_for_hashtag_in_tweet(train,test)
		test_score2 = mean_squared_error(test_y, predictions2)
		print('=Baseline 2 = Test MSE: %.3f' % test_score2)
		
		predictions3 = baselines.mean_for_user(train,test)
		test_score3 = mean_squared_error(test_y, predictions3)
		print('=Baseline 3 = Test MSE: %.3f' % test_score3)
		
		predictions4 = baselines.maximum_mean_for_hashtag_for_user(train,test)
		test_score4 = mean_squared_error(test_y, predictions4)
		print('=Baseline 4 = Test MSE: %.3f' % test_score4)
		
		predictions5 = []
		for i in range(0,len(predictions1)):
			lis = [predictions1[i], predictions2[i], predictions3[i], predictions4[i]]
			predictions5.append(max(lis))
		test_score5 = mean_squared_error(test_y, predictions5)
		print('=Max Baseline = Test MSE: %.3f' % test_score5)
		
		predictions6 = []
		for i in range(0,len(predictions1)):
			lis = [predictions1[i], predictions2[i], predictions3[i], predictions4[i]]
			predictions6.append(sum(lis)/len(lis))
		test_score6 = mean_squared_error(test_y, predictions6)
		print('=Mean Baseline = Test MSE: %.3f' % test_score6)
		
		
		for i in range(0,len(test_X)):
			# remplace hashtags by vector
			test_X[i][0] = hashtag_vector[test_X[i][0].lower()]
			# transform dayhour in minutes
			test_X[i][2] = label_to_mins(test_X[i][2])
		for i in range(0,len(train_X)):
			# remplace hashtags by vector
			train_X[i][0] = hashtag_vector[train_X[i][0].lower()]
			# transform dayhour in minutes
			train_X[i][2] = label_to_mins(train_X[i][2])
		print("Train model")
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		svr_rbf.fit(train_X, train_y)
		print("Predict model")
		predictions = svr_rbf.predict(test_X)
		test_score = mean_squared_error(test_y, predictions)
		print('=Model Test MSE: %.3f' % test_score)
		
		if test_score < test_score1:
			print("** Baseline 1 OK")
		else:
			print("** Baseline 1 NOT OK")
			
		print('=Model-baselines1 prediction Test MSE: %.3f' % mean_squared_error(predictions1, predictions))
		
		if test_score < test_score2:
			print("** Baseline 2 OK")
		else:
			print("** Baseline 2 NOT OK")	
			
		print('=Model-baselines2 prediction Test MSE: %.3f' % mean_squared_error(predictions2, predictions))
	
		if test_score < test_score3:
			print("** Baseline 3 OK")
		else:
			print("** Baseline 3 NOT OK")
			
		print('=Model-baselines3 prediction Test MSE: %.3f' % mean_squared_error(predictions3, predictions))
	
		if test_score < test_score4:
			print("** Baseline 4 OK")
		else:
			print("** Baseline 4 NOT OK")
			
		print('=Model-baselines4 prediction Test MSE: %.3f' % mean_squared_error(predictions4, predictions))
	
		if test_score < test_score5:
			print("** Baseline 5 OK")
		else:
			print("** Baseline 5 NOT OK")
			
		print('=Model-baselines5 prediction Test MSE: %.3f' % mean_squared_error(predictions5, predictions))
	
		if test_score < test_score6:
			print("** Baseline 6 OK")
		else:
			print("** Baseline 6 NOT OK")
			
		print('=Model-baselines6 prediction Test MSE: %.3f' % mean_squared_error(predictions6, predictions))
	

		days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":0}
		while 1:
			doc = input("\n============\nWhat hashtag do you want to test? for instance '"+hashtags[random.randint(0, len(hashtags))]+"'\n").lower()
			if doc in hashtag_vector:
				day = int(input("Day of the week?\n"))
				if day >= 0 and day <= 6:
					hour_label = input("Hour of the day? format HH:MM\n").strip('\n')
					hour = label_to_mins(hour_label)

					print("\nScore for '"+ str(doc) +"' , Day : "+ str(calendar.day_name[day+1])+", Hour : "+ str(hour_label) +" = "+ str(int(svr_rbf.predict([[hashtag_vector[doc], day, hour]]))))
			else:
				print("Hashtag not found")
				
if __name__ == "__main__":
    args = parse_arguments()
    main(args)