#!/usr/bin/env python
# encoding: utf-8
import argparse
from sklearn.metrics import mean_squared_error
from baselines import Baselines
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test baselines (MSE).')
    parser.add_argument('-f, --file', dest='file', type=open, help='file with tweets gathered')
    parser.add_argument('-s, --skip', dest='skip', action='store_true', help='skip tweets with no hashtags')
    return parser.parse_args()
	
	
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
		train = [ {'user': 'user1', 'hashtag':['sport','nhl'], 'score': 10, 'score2': 100}, {'user': 'user1', 'hashtag':['sport','nba'], 'score': 20, 'score2': 200}, {'user': 'user2', 'hashtag':['sport','nhl'], 'score': 10, 'score2': 10}, {'user': 'user1', 'hashtag':['nba'], 'score': 30, 'score2': 300}]
		test = [ {'user': 'user1', 'hashtag':['sport','nhl'], 'score': 10, 'score2': 100}, {'user': 'user3', 'hashtag':['sport','nhl'], 'score': 10, 'score2': 30}, {'user': 'user2', 'hashtag':['sport','nhl'], 'score': 10, 'score2': 10}, {'user': 'user3', 'hashtag':[], 'score': 10, 'score2': 30}, {'user': 'user1', 'hashtag':[], 'score': 10, 'score2': 100} ]

	test_y = [[row['score'],row['score2']] for row in test]
	# baselines
	predictions_baseline = []
	for i in range(0,4):		
		baselines = Baselines()
		pred = baselines.int2function(i, train, test)
		test_score = mean_squared_error(test_y, pred)
		print('=Baseline '+str(i+1)+' = Test MSE: %.3f' % test_score)
		predictions_baseline.append({'score':test_score, 'prediction':pred})
	
	predictions1 = []
	predictions2 = []
	for i in range(0,len(predictions_baseline[0]['prediction'])):
		lis = [predictions_baseline[0]['prediction'][i], predictions_baseline[1]['prediction'][i], predictions_baseline[2]['prediction'][i], predictions_baseline[3]['prediction'][i]]
		predictions1.extend(np.matrix(lis).max(0).tolist())
		predictions2.append(np.mean(lis, axis=0))
	test_score1 = mean_squared_error(test_y, predictions1)
	test_score2 = mean_squared_error(test_y, predictions2)
	print('=Max Baseline = Test MSE: %.3f' % test_score1)
	print('=Mean Baseline = Test MSE: %.3f' % test_score2)
	predictions_baseline.append({'score':test_score1, 'prediction':predictions1})
	predictions_baseline.append({'score':test_score2, 'prediction':predictions2})

	
if __name__ == "__main__":
    args = parse_arguments()
    main(args)