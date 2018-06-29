import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from baselines.baselines import Baselines
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
import calendar
import json
import os
import re
from gensim.models import Word2Vec
from textblob import TextBlob
import optunity
import optunity.metrics

from sklearn.model_selection import GridSearchCV

class Model:

	def __init__(self):
	
		self.base_path = "data/"
		self.cache_path = self.base_path + "cache/"
		self.publish_path = self.base_path + "publish/"
		self.W2Vmodel = [0,0]
		self.predictions = []
		self.predictions_baseline = []
		
		#self.max_depth = 110
		#self.max_features = 2
		#self.min_samples_split = 10
		#self.n_estimators = 300
		self.min_samples_leaf = 1
		self.min_samples_split = 2
		self.max_depth = None
		self.max_features = "auto"
		self.n_estimators = 15
		
	def predict(self, model_data, data, load=True, full=True):
	
		if load or not self.regr_rf:
			model_config = json.load(model_data)
			self.regr_rf = joblib.load(model_config["model_path"])
			self.W2Vmodel[0] = joblib.load(model_config["w2vmodel_0"])
			self.W2Vmodel[1] = joblib.load(model_config["w2vmodel_1"])
			
		res = [dict(), dict()]
		for i in range(0,2):
			res[i]["max_day"] = 0
			res[i]["max_hour"] = 0
			res[i]["max_score"] = 0
			
		if not "followers_count" in data or not "friends_count" in data or not "listed_count" in data or not "statuses_count" in data or not "text" in data:
			return res
			
		webdataset = dict()
		
		hashtag = re.findall("\B#\w*[a-zA-Z]+\w*", data['text'])
		data['hashtag'] = []
		for h in hashtag:
			data['hashtag'].append(h[1:])
		print(data['hashtag'])
		
		if not "weekday" in data:
			data['weekday'] = 0
		if not "hour" in data:
			data['hour'] = "00:00"
		
		X = [[data['hashtag'], data['weekday'], data['hour'], data['followers_count'], data['friends_count'], data['listed_count'], data['statuses_count'], data['text'], 0, 0, 0, 0, 0, 0, 0, 0, 0]]
		X = self.normalize_dataset(X)
		
		if not full:
			return self.regr_rf.predict(X)[0].tolist()
			
		max_day = [0,0]
		max_score = [0,0]
		max_hour = [0,0]
		score = [0, 0]
		for d in range(0,7):
			print("day "+ str(d))
			
			webdataset[d] = [[],[]]
			
			for m in range(0,1439,60):
				
				X[0][1] = d
				X[0][2] = m
				
				score = self.regr_rf.predict(X)[0]
				
				webdataset[d][0].append({'x' : m, 'y': round(score[0],2)})
				webdataset[d][1].append({'x' : m, 'y': round(score[1],2)})
				
				print("min "+ self.mins_to_label(m) + " - ["+ str(score[0]) +"," + str(score[1])+"]")
				for i in range(0,2):
					if score[i] > max_score[i] :
						max_day[i] = d
						max_hour[i] = m
						max_score[i] = score[i]
		webdataset['score'] = []				
		for i in range(0,2):
			res[i]["max_day"] = max_day[i]
			res[i]["max_hour"] = max_hour[i]
			res[i]["max_score"] = round(max_score[i],2)
			
		webdataset['max'] = res
		webdataset['labels'] = []
		for m in range(0,1439,60):
			webdataset['labels'].append(self.mins_to_label(m))
		
		with open("data/publish/webdataset.json", 'w') as f:
			f.write(json.dumps(webdataset))
		pass
		
		return webdataset
		
	def run(self, file):
		c = 0
		while 1:
			c+=1
			doc = input("\n============\nWhat Text do you want to test?")
			if doc:
				
				data = dict()
				data['followers_count']=43
				data['friends_count']= 34
				data['listed_count']=1
				data['statuses_count'] = 100
				data['text'] = doc
				
				res = self.predict(file, data, c == 1)
				
				print("\nBest Date for '"+ str(doc) +"' = ")
				for i in range(0,2):
					print("score "+ str(i))
					print(res['max'][i]["max_day"])
					print(self.mins_to_label(res['max'][i]["max_hour"]))
		
			else:
				break
	
	
	def train(self, file, tuning, cache, save, svr, ignore):
		
		if svr:
			self.svr = True
			print("== SVR mode ==")
		else:
			self.svr = False
			
		if tuning:
			self.tuning = int(tuning)
		else:
			self.tuning = 0
			
		self.file = file
		
		if cache:
			self.cache = cache
		else:
			self.cache = 0
			
		self.datas = json.load(self.file)
		
		if ignore and ignore > 0:
			class_0 = 0
			class_1 = 0
			class_2 = 0
			self.ignore = ignore
			counter = list()
			for i in range(0, len(self.datas)):
				if self.datas[i]['score2'] > 300:
					counter.append(i)
					continue
				if self.ignore > 1 and self.datas[i]['followers_count'] > 10000:
					counter.append(i)
					continue
				if self.ignore > 2 and self.datas[i]['score'] > 20000:
					counter.append(i)
					continue
				if self.ignore > 3:
					if self.datas[i]['score2'] >= 200:
						if class_2 > 30000:
							counter.append(i)
							continue
						else:
							class_2 += 1
					elif self.datas[i]['score2'] >= 50:
						if class_1 > 30000:
							counter.append(i)
							continue
						else:
							class_1 += 1
					elif self.datas[i]['score2'] >= 0:
						if class_0 > 30000:
							counter.append(i)
							continue
						else:
							class_0 += 1
				
			self.datas = [self.datas[i] for i in range(0, len(self.datas)) if i not in counter]
			print(str(len(counter))+" aberrant values removed")
		else:
			self.ignore = 0
		# Split data
		self.train, self.test = train_test_split(self.datas, test_size=0.33, shuffle=True, random_state=42)
		
		# Format data
		if self.svr:
			self.test_y, self.train_y = [row['score'] for row in self.test], [row['score'] for row in self.train]
		else:
			self.test_y, self.train_y = [[row['score'], row['score2']] for row in self.test], [[row['score'], row['score2']] for row in self.train]
		self.test_X, self.train_X = [[row['hashtag'], row['weekday'], row['hour'], row['followers_count'], row['friends_count'], row['listed_count'], row['statuses_count'], row['text'], 0, 0, 0, 0, 0, 0, 0, 0, 0] for row in self.test], [[row['hashtag'], row['weekday'], row['hour'], row['followers_count'], row['friends_count'], row['listed_count'], row['statuses_count'], row['text'], 0, 0, 0, 0, 0, 0, 0, 0, 0] for row in self.train]
		self.names = ['hashtag', 'weekday', 'hour', 'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'text', 'quote', 'link', '...', '!', '?', '@', 'upper', 'polarity', 'subjectivity' ]
		
		# Prepare features
		self.prepare_columns()
		
		# baselines
		self.cache_baseline()
		
		# Normalize dataset
		print("Prepare dataset...")
		self.cache_dataset()
		
		if self.tuning == 1:
			print("Tuning model")
			if self.svr:
				outer_cv = optunity.cross_validated(x=self.train_X, y=self.train_y, num_folds=3)
				def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
					"""Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

					# define objective function for tuning
					@optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
					def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
						print("tune_cv model C="+str(C)+", gamma="+str(gamma))
						model = SVR(C=C, gamma=gamma).fit(x_train, y_train)
						print("tune_cv model fit")
						predictions = model.predict(x_test)
						return optunity.metrics.mse(y_test, predictions)

					# optimize parameters
					optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 100], gamma=[0, 50])
					print("optimal hyperparameters: " + str(optimal_pars))

					tuned_model = SVR(**optimal_pars).fit(x_train, y_train)
					predictions = tuned_model.predict(x_test)
					return optunity.metrics.mse(y_test, predictions)

				# wrap with outer cross-validation
				compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)
				compute_mse_rbf_tuned()
			else:
				sample_leaf_options = [1,5,10,50,100,200,500]
				for leaf_size in sample_leaf_options :
					print(":: leaf_size = " + str(leaf_size))
					self.min_samples_leaf = leaf_size
					self.cache_model()
					
					print("Predict model")
					self.predictions = self.regr_rf.predict(self.test_X)
					
					print("Feature importance : ")
					print(sorted(zip(map(lambda x: round(x, 4), self.regr_rf.feature_importances_), self.names), reverse=True))
					self.test_score_rf = mean_squared_error(self.test_y, self.predictions)
					print('=Model Test MSE: %.3f' % self.test_score_rf)
					
					self.test_score = self.test_score_rf
					
					self.evaluation()
				
		elif self.tuning == 2:
			print("Tuning model 2")
			param_grid = { \
				'bootstrap': [True, False],\
				'max_depth': [80, 90, 100, 110],\
				'max_features': [2, 3],\
				'min_samples_leaf': [1, 3, 4, 5, 500],\
				'min_samples_split': [8, 10, 12],\
				'n_estimators': [100, 200, 300, 1000]\
			}
			rf = RandomForestRegressor()
			# Instantiate the grid search model
			grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
			grid_search.fit(self.train_X, self.train_y)
			print("grid_search.best_params_=")
			print(grid_search.best_params_)
			
				
		else:
			print("Train model")
			
			self.cache_model()
			
			if save:
				print("Save model")
				self.save_model()
			
			print("Predict model")
			self.predictions = self.regr_rf.predict(self.test_X)
			
			if not self.svr:
				print("Feature importance : ")
				print(sorted(zip(map(lambda x: round(x, 4), self.regr_rf.feature_importances_), self.names), reverse=True))
			self.test_score_rf = mean_squared_error(self.test_y, self.predictions)
			print('=Model Test MSE: %.3f' % self.test_score_rf)
			self.test_score = self.test_score_rf
			print('r2 = ')
			print(r2_score(self.test_y, self.predictions, multioutput='raw_values'))
			
			classif = list()
			classif_pred = list()
			for i in range(0, len(self.test_y)):
				if self.test_y[i][1] >= 200:
					classif.append(2)
				elif self.test_y[i][1] >= 50:
					classif.append(1)		
				elif self.test_y[i][1] >= 0:
					classif.append(0)
					
			for i in range(0, len(self.predictions)):
				if self.predictions[i][1] >= 200:
					classif_pred.append(2)
				elif self.predictions[i][1] >= 50:
					classif_pred.append(1)		
				elif self.predictions[i][1] >= 0:
					classif_pred.append(0)
			
			target_names = ['class 0', 'class 1', 'class 2']
			print(classification_report(classif, classif_pred, target_names=target_names))
			
			x = np.asarray(self.test_y)[:,0]
			y = np.asarray(self.predictions)[:,0]
			max = [np.amax(x), np.amax(y)]
			x1 = [0, np.amax(max)]
			plt.figure()
			plt.plot(x, y, 'r+')
			plt.plot(x1, x1)
			plt.figure()
			x = np.asarray(self.test_y)[:,1]
			y = np.asarray(self.predictions)[:,1]
			max = [np.amax(x), np.amax(y)]
			x1 = [0, np.amax(max)]
			plt.plot(x, y, 'g+')
			plt.plot(x1, x1)
			plt.show()
			
			self.evaluation()
				
	def label_to_mins(self, label):
		[hour, min] = label.split(":")	
		return int(int(min) + 60*int(hour))
		
	def mins_to_label(self, mins):
		if mins < 139440:
			hour = int(mins/60)
			if hour < 10:
				hour = "0" + str(hour) 
			else:
				hour = str(hour) 
			min = int(mins%60)
			if min < 10:
				min = "0" + str(min)
			else:
				min = str(min)
		else:
			hour = "00"
			min = "00"
		return ":".join([hour, min])

	def build_hashtag_vector(self, hashtags, lower = 0):
		h = dict()
		i = 0
		for hashtag in hashtags:
			if not hashtag.lower() in h:
				h[hashtag.lower()] = i
				i += 1
		return h
		
	def build_emo_vector(self, emoticons):
		h = dict()
		i = 0
		for e in emoticons:
			if not e in h:
				h[e] = i
				i += 1
		return h

	def train_baselines(self):	
		for i in range(0,4):		
			baselines = Baselines()
			pred = baselines.int2function(i, self.train, self.test)
			test_score = mean_squared_error(self.test_y, pred)
			print('=Baseline '+str(i+1)+' = Test MSE: %.3f' % test_score)
			self.predictions_baseline.append({'score':test_score, 'prediction':pred})
		
		predictions1 = []
		predictions2 = []
		for i in range(0,len(self.predictions_baseline[0]['prediction'])):
			lis = [self.predictions_baseline[0]['prediction'][i], self.predictions_baseline[1]['prediction'][i], self.predictions_baseline[2]['prediction'][i], self.predictions_baseline[3]['prediction'][i]]
			predictions1.extend(np.matrix(lis).max(0).tolist())
			predictions2.append(np.mean(lis, axis=0).tolist())
		test_score1 = mean_squared_error(self.test_y, predictions1)
		test_score2 = mean_squared_error(self.test_y, predictions2)
		print('=Max Baseline = Test MSE: %.3f' % test_score1)
		print('=Mean Baseline = Test MSE: %.3f' % test_score2)
		self.predictions_baseline.append({'score':test_score1, 'prediction':predictions1})
		self.predictions_baseline.append({'score':test_score2, 'prediction':predictions2})
		
		
	def evaluation(self):
		for i in range(6):
			if self.test_score < self.predictions_baseline[i]['score']:
				print("** Baseline "+ str(i) +" OK")
			else:
				print("** Baseline "+ str(i) +" NOT OK")
			if self.svr:
				temp = []
				for d in self.predictions_baseline[i]['prediction']:
					temp.append(d[0])
				self.predictions_baseline[i]['prediction'] = temp
			print('=Model-baselines '+ str(i) +' prediction Test MSE: %.3f' % mean_squared_error(self.predictions_baseline[i]['prediction'], self.predictions))

	def cache_baseline(self):
		cache = self.cache_path + self.file.name.split('\\')[2].split('.')[0] + "-baselines"
		for i in range(0,self.ignore):
			cache += "-i"
		cache += ".json"
		if not os.path.exists(cache) or self.cache == 2:
			self.train_baselines()
			with open(cache, 'w') as f:
				f.write(json.dumps(self.predictions_baseline))
			pass
		else:
			print("(cache used)")
			self.predictions_baseline = json.load(open(cache))
			for i in range(6):
				print('=Baseline '+str(i)+' = Test MSE: %.3f' % self.predictions_baseline[i]['score'])

	def cache_model(self):
		if self.svr:
			cache = self.cache_path + self.file.name.split('\\')[2].split('.')[0] + "-" + str(self.min_samples_leaf) + "_model_svr.pkl"
		else:
			cache = self.cache_path + self.file.name.split('\\')[2].split('.')[0] + "-" + str(self.min_samples_leaf) + "_model"
			for i in range(0,self.ignore):
				cache += "-i"
			cache += ".pkl"
		
		if not os.path.exists(cache) or self.cache:
			print("(fit new model)")
			if self.svr:
				#self.regr_rf = SVR(kernel='rbf', C=1e3, gamma=5e-10)
				self.regr_rf = SVR(kernel='rbf', C=99.95588, gamma=22.38053)
			else:
				self.regr_rf = RandomForestRegressor(bootstrap=True, max_depth=self.max_depth, max_features=self.max_features, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, n_estimators=self.n_estimators, n_jobs=-1)
			self.regr_rf.fit(self.train_X, self.train_y)		
			joblib.dump(self.regr_rf, cache)
		else:
			print("(cache used)")
			self.regr_rf = joblib.load(cache)
	
	def save_model(self):
		save_date = str( datetime.now().isoformat(timespec='seconds').replace(":","")  )
		
		for i in range(0,self.ignore):
			save_date += "-i"
		# copy model
		joblib.dump(self.regr_rf, self.publish_path + save_date +".model")
		# copy w2v models
		joblib.dump(self.W2Vmodel[0], self.publish_path + save_date +"_text.w2vmodel")
		joblib.dump(self.W2Vmodel[1], self.publish_path + save_date +"_hash.w2vmodel")
		
		model_config = dict()
		model_config["model_path"] = self.publish_path + save_date +".model"
		model_config["w2vmodel_0"] = self.publish_path + save_date +"_text.w2vmodel"
		model_config["w2vmodel_1"] = self.publish_path + save_date +"_hash.w2vmodel"
		
		with open(self.publish_path + save_date +".model.json", 'w') as outfile:
			json.dump(model_config, outfile)
		
		
	def cache_dataset(self):		
		
		cache_test = self.cache_path + self.file.name.split('\\')[2].split('.')[0] + "_test_X"
		for i in range(0,self.ignore):
			cache_test += "-i"
		cache_test += ".pkl"
		cache_train = self.cache_path + self.file.name.split('\\')[2].split('.')[0] + "_train_X"
		for i in range(0,self.ignore):
			cache_train += "-i"
		cache_train += ".pkl"
				
		if not os.path.exists(cache_test) or self.cache:
			self.test_X = self.normalize_dataset(self.test_X)	
			self.train_X = self.normalize_dataset(self.train_X)
			joblib.dump(self.test_X, cache_test)
			joblib.dump(self.train_X, cache_train)
		else:
			print("(cache used)")
			self.test_X = joblib.load(cache_test)
			self.train_X = joblib.load(cache_train)

	def prepare_columns(self):
		hashtags = []
		sentences = []
		for data in self.datas:
			composition_h = "".join(data['hashtag'])
			if composition_h:
				hashtags.extend([list(composition_h)])
			sentences.append(data['text'].split(" "))
			
		# Build Word2Vec model
		self.W2Vmodel[0] = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, hs=1, negative=0)
		self.W2Vmodel[1] = Word2Vec(hashtags, size=100, window=5, min_count=1, workers=4, hs=1, negative=0)

	def normalize_dataset(self, X):
	
		def count_by_lambda(expression, word_array):
			return len(list(filter(expression, word_array)))

		def count_occurences(character, word_array):
			counter = 0
			for j, word in enumerate(word_array):
				for char in word:
					if char == character:
						counter += 1

			return counter

		def count_by_regex(regex, plain_text):
			return len(regex.findall(plain_text))
			
		for i in range(0,len(X)):
		
			hashtag = [list("".join(X[i][0]))]
			X[i][0] = self.W2Vmodel[1].score(hashtag, len(hashtag))
			
			X[i][8] =  list(map(lambda plain_text: int(count_occurences("'", [plain_text.strip("'").strip('"')]) / 2 + count_occurences('"', [plain_text.strip("'").strip('"')]) / 2), X[i][7]))[0]
			X[i][9] = list(map(lambda txt: count_by_regex(re.compile(r"http.?://[^\s]+[\s]?"), txt),X[i][7]))[0]
			X[i][10] = list(map(lambda txt: count_by_regex(re.compile(r"\.\s?\.\s?\."), txt),X[i][7]))[0]
			a = TextBlob(X[i][7])
			X[i][15] = a.sentiment.polarity
			X[i][16] = a.sentiment.subjectivity
			
			sentence = [X[i][7].split()]
			X[i][7] = self.W2Vmodel[0].score(sentence, len(sentence))
			X[i][11] = list(map(lambda txt: count_occurences("!", txt), sentence))[0]
			X[i][12] = list(map(lambda txt: count_occurences("?", txt), sentence))[0]
			X[i][13] = list(map(lambda txt: count_occurences("@", txt), sentence))[0]
			X[i][14] = list(map(lambda txt: count_by_lambda(lambda word: word == word.upper(), txt), sentence))[0]
			
			X[i][2] = self.label_to_mins(X[i][2])
			
		return X
		