import json
from sklearn.metrics import mean_squared_error

class Baselines:
	
	def int2function(self, i, train, test):
		if i==0:
			return self.overall_mean(train,test)
		elif i==1:
			return self.maximum_mean_for_hashtag_in_tweet(train,test)
		elif i==2:
			return self.mean_for_user(train,test)
		else:
			return self.maximum_mean_for_hashtag_for_user(train,test)
	
	def overall_mean(self, train, test):
		output_values = [[row['score'],row['score2']] for row in train]
		prediction0 = sum(output_values[0]) / float(len(output_values[0]))
		prediction1 = sum(output_values[1]) / float(len(output_values[1]))
		predicted = [[prediction0, prediction1] for i in range(len(test))]
		
		return predicted	
		
	def maximum_mean_for_hashtag_in_tweet(self, train, test):
		predicted = []
		prediction_h = dict()
		
		# We run on each test entry
		for i in range(0,len(test)):
		
			selected_prediction = dict()
			selected_prediction['score'] = []
			selected_prediction['score2'] = []
			
			for h in test[i]['hashtag']:
				if not h in prediction_h:
					scores_h = dict()
					scores_h['score'] = []
					scores_h['score2'] = []
					
					for j in range(0,len(train)):
						if h in train[j]['hashtag']:
							scores_h['score'].append(train[j]['score'])
							scores_h['score2'].append(train[j]['score2'])		
							
					prediction_h[h] = dict()
					prediction_h[h]['score'] = []
					prediction_h[h]['score2'] = []
					if scores_h['score'] and scores_h['score2']:
						prediction_h[h]['score'].append(sum(scores_h['score']) / len(scores_h['score']))
						prediction_h[h]['score2'].append(sum(scores_h['score2']) / len(scores_h['score2']))
					else:
						prediction_h[h]['score'].append(0)
						prediction_h[h]['score2'].append(0)
						
				selected_prediction['score'].extend(prediction_h[h]['score'])
				selected_prediction['score2'].extend(prediction_h[h]['score2'])
				
			if selected_prediction['score'] and selected_prediction['score2']:
				predicted.append([max(selected_prediction['score']), max(selected_prediction['score2'])])
			else:
				predicted.append([0,0])
				
		return predicted

	def mean_for_user(self, train, test):
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
				scores = dict()
				scores['score'] = []
				scores['score2'] = []
				for j in range(0,len(train)):
									
					if user == train[j]['user']:
						scores['score'].append(train[j]['score2'])
						scores['score2'].append(train[j]['score'])
				
				prediction = []
				# Mean value of scores
				if scores['score'] and scores['score2']:
					prediction.extend([sum(scores['score']) / len(scores['score']), sum(scores['score2']) / len(scores['score2'])])
				else:
					prediction.extend([0,0])
					
				# Save user prediction
				users[user] = prediction
				predicted.append(users[user])
				
		return predicted
		
	def maximum_mean_for_hashtag_for_user(self, train, test):
		predicted = []
		prediction_h = dict()
		selected_prediction = dict()
		
		# We run on each test entry
		for i in range(0,len(test)):
		
			user = test[i]['user']
			prediction_h[user] = dict()
			selected_prediction[user] = dict()
			selected_prediction[user]['score'] = []
			selected_prediction[user]['score2'] = []
			
			for h in test[i]['hashtag']:
				
				if not user in prediction_h or not h in prediction_h[user]:	
					scores_h = dict()
					scores_h['score'] = []
					scores_h['score2'] = []
					loin = 0
					for j in range(0,len(train)):
						if loin > 100:
							break
						if user == train[j]['user'] and h in train[j]['hashtag']:
							scores_h['score'].append(train[j]['score'])					
							scores_h['score2'].append(train[j]['score2'])
							loin = 0
						else:
							loin += 1
							
					prediction_h[user][h] = dict()
					if scores_h['score'] and scores_h['score2']:
						prediction_h[user][h]['score'] = sum(scores_h['score']) / len(scores_h['score'])
						prediction_h[user][h]['score2'] = sum(scores_h['score2']) / len(scores_h['score2'])
					else:
						prediction_h[user][h]['score'] = 0
						prediction_h[user][h]['score2'] = 0
						
				selected_prediction[user]['score'].append(prediction_h[user][h]['score'])
				selected_prediction[user]['score2'].append(prediction_h[user][h]['score2'])
					
			if selected_prediction[user]['score'] and selected_prediction[user]['score2']:
				predicted.append([max(selected_prediction[user]['score']), max(selected_prediction[user]['score2'])])
			else:
				predicted.append([0, 0])
				
		return predicted
