#!/usr/bin/env python
# encoding: utf-8
import argparse
import json
from datetime import datetime
from dateutil import parser
import re
import os
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    parser.add_argument('-m, --m', dest='max', action='store', type=int, help='maximum to gather')
    parser.add_argument('-s, --skip', dest='skip', action='store_true', help='skip tweets text content')
    return parser.parse_args()


def main(run_args):
	total_hashtags = 0
	without_hashtags = 0
	tweets = []
	elements = os.listdir('data/datasets')
	random.shuffle(elements)
	for element in elements:
		if element.endswith('_tweets.json'):
		
			user_tweets = json.load(open('data/datasets/' + element))
			account_name = element.replace("_tweets.json","");
			temp_without_hashtags = 0
			temp_total_hashtags = 0
			print("### Get hashtags from : "+account_name+" ###")
			
			for t in user_tweets:
				hour = parser.parse(t['date'])	
				
				# parse hashtags
				hashtag = []
				if t['hashtags']:
					for h in t['hashtags']:
						if re.match("^[a-zA-Z0-9àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßÆæœ_-]*$", h):
							temp_total_hashtags += 1
							hashtag.append(h)					
				else:
					temp_without_hashtags += 1
				#tweets.append({'user': account_name, 'weekday': (int(hour.strftime('%w'))), 'hour':(hour.strftime('%H:%M')), 'hashtag': hashtag, 'score': (t['rt'] * 2 + t['fav']), 'score2': round(t['rt'] * 100 / t['followers_count'],4),'text': t['text'], 'followers_count':t['followers_count'], 'friends_count':t['friends_count'], 'listed_count':t['listed_count'], 'statuses_count':t['statuses_count']})
				tweets.append({'user': account_name, 'weekday': (int(hour.strftime('%w'))), 'hour':(hour.strftime('%H:%M')), 'hashtag': hashtag, 'score': t['rt'], 'score2': round(t['rt'] * 100 / t['followers_count'],4),'text': t['text'], 'followers_count':t['followers_count'], 'friends_count':t['friends_count'], 'listed_count':t['listed_count'], 'statuses_count':t['statuses_count']})
			
			print("Number of Tweets : "+ str(len(user_tweets)))
			print("Number of tweets without hashtags : "+ str(temp_without_hashtags))
			print("Number of hashtags : "+ str(temp_total_hashtags))
			
			# Update final counter
			total_hashtags += temp_total_hashtags
			without_hashtags += temp_without_hashtags
			
			if run_args.max and total_hashtags > run_args.max:
				break

	filename = 'data/gathered/gathering_' +str(len(tweets))+ '_'  +str(total_hashtags)+ '-only_rt.json'
	with open(filename, 'w') as outfile:
		json.dump(tweets, outfile)
		
	print("\n=======================")
	print("Number of tweets : " + str(len(tweets)))
	print("With hashtags : "+ str(len(tweets) - without_hashtags) + " ("+str(round((len(tweets) - without_hashtags)*100/len(tweets),2))+"%)")
	print("Without hashtags : "+ str(without_hashtags) + " ("+str(round(without_hashtags*100/len(tweets),2))+"%)")
	print("Total hashtag : "+ str(total_hashtags)+ " (mean value : "+str(round(total_hashtags*100/len(tweets),2))+"%)")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)