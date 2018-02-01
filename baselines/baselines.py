#!/usr/bin/env python
# encoding: utf-8
import argparse
import json
from sklearn.metrics import mean_squared_error

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test baselines (MSE).')
    parser.add_argument('-f, --file', dest='file', type=open, help='file with tweets gathered')
    parser.add_argument('-s, --skip', dest='skip', action='store_true', help='skip tweets with no hashtags')
    return parser.parse_args()
	
	
	

def overall_mean(train, test):
	output_values = [row['score'] for row in train]
	prediction = sum(output_values) / float(len(output_values))
	predicted = [prediction for i in range(len(test))]
	
	return predicted	
	
def maximum_mean_for_hashtag_in_tweet(train, test):
	predicted = []
	prediction_h = dict()
	
	# We run on each test entry
	for i in range(0,len(test)):
	
		selected_prediction = []
		
		for h in test[i]['hashtag']:
			
			if h in prediction_h:
				selected_prediction.append(prediction_h[h])
			else:			
				scores_h = []
				for j in range(0,len(train)):
					if h in train[j]['hashtag']:
						scores_h.append(train[j]['score'])					
						
				if scores_h:
					prediction_h[h] = sum(scores_h) / len(scores_h)
				else:
					prediction_h[h] = 0
		
			selected_prediction.append(prediction_h[h])
			
		if selected_prediction:
			predicted.append(max(selected_prediction))
		else:
			predicted.append(0)
		
	return predicted

def mean_for_user(train, test):
	users = dict()
	predicted = []
	
	# We run on each user
	for i in range(0,len(test)):
		user = test[i]['user']
		
		# If prediction already calculated for this user
		if user in users:
			predicted.append(users[user])
		# Else calculate prediction
		else:
			scores = []
			for j in range(0,len(train)):
								
				if user == train[j]['user']:
					scores.append(train[j]['score'])
					
			# Mean value of scores
			if scores:
				prediction = sum(scores) / len(scores)
			else:
				prediction = 0
			# Save user prediction
			users[user] = prediction
			predicted.append(users[user])
			
	return predicted
	
def maximum_mean_for_hashtag_for_user(train, test):
	predicted = []
	prediction_h = dict()
	selected_prediction = dict()
	
	# We run on each test entry
	for i in range(0,len(test)):
	
		user = test[i]['user']
		prediction_h[user] = dict()
		selected_prediction[user] = []
		
		for h in test[i]['hashtag']:
			
			if user in prediction_h and h in prediction_h[user]:
				selected_prediction[user].append(prediction_h[user][h])
			else:			
				scores_h = []
				for j in range(0,len(train)):
					if user == train[j]['user'] and h in train[j]['hashtag']:
						scores_h.append(train[j]['score'])					
						
				if scores_h:
					prediction_h[user][h] = sum(scores_h) / len(scores_h)
				else:
					prediction_h[user][h] = 0
					
			selected_prediction[user].append(prediction_h[user][h])
		
		if selected_prediction[user]:
			predicted.append(max(selected_prediction[user]))
		else:
			predicted.append(0)
		
	return predicted

	
def main(run_args):
	print("# Tests")
	if run_args.file:	
		# load dataset
		datas = json.load(run_args.file)

		if run_args.skip:
			# Delete empty hashtags
			print("Before cleaning tweets without hashtags : "+str(len(datas)))
			datas = [row for row in datas if len(row['hashtag'])>0]
			print("After cleaning tweets without hashtags : "+str(len(datas)))
			
		# split datas
		train_size = int(len(datas) * 0.66)
		train, test = datas[1:train_size], datas[train_size:]
	else:
		train = [ {'user': 'user1', 'hashtag':['sport','nhl'], 'score': 10}, {'user': 'user1', 'hashtag':['sport','nba'], 'score': 20}, {'user': 'user2', 'hashtag':['sport','nhl'], 'score': 10}, {'user': 'user1', 'hashtag':['nba'], 'score': 30}]
		test = [ {'user': 'user1', 'hashtag':['sport','nhl'], 'score': 10}, {'user': 'user3', 'hashtag':['sport','nhl'], 'score': 10}, {'user': 'user2', 'hashtag':['sport','nhl'], 'score': 10}, {'user': 'user3', 'hashtag':[], 'score': 10}, {'user': 'user1', 'hashtag':[], 'score': 10} ]

	test_y = [row['score'] for row in test]
	# baselines
	predictions1 = overall_mean(train,test)
	test_score1 = mean_squared_error(test_y, predictions1)
	print('=Baseline 1 = Test MSE: %.3f' % test_score1)
	
	predictions2 = maximum_mean_for_hashtag_in_tweet(train,test)
	test_score2 = mean_squared_error(test_y, predictions2)
	print('=Baseline 2 = Test MSE: %.3f' % test_score2)
	
	predictions3 = mean_for_user(train,test)
	test_score3 = mean_squared_error(test_y, predictions3)
	print('=Baseline 3 = Test MSE: %.3f' % test_score3)
	
	predictions4 = maximum_mean_for_hashtag_for_user(train,test)
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

	
	
if __name__ == "__main__":
    args = parse_arguments()
    main(args)