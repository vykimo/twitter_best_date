#!/usr/bin/env python
# encoding: utf-8
import argparse
import json
from datetime import datetime
from dateutil import parser
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f, --file', dest='file', action='store', help='file with twitter accounts')
    parser.add_argument('-m, --m', dest='max', action='store_true', help='maximum to gather')
    return parser.parse_args()


def main(run_args):

	if run_args.file:
		accounts = open(run_args.file)
		
		total_hashtags = 0
		
		tweets = []
		
		for account in accounts:
		
			temp_total_hashtags = 0
			account_name = account.strip('\n');
			print("\n### Get hashtags from : "+account_name+" ###")
	
			data = json.load(open('data\%s_tweets_normalized.json' % account_name))
			
			tweets.extend(data)
			
			temp_total_hashtags = len(data)
			
			print("\n\nNumber of hashtags : "+ str(temp_total_hashtags))
			total_hashtags += temp_total_hashtags
			if run_args.max and total_hashtags > run_args.max:
				break
		
		fname = 'data\gathering_%s_tweets.data' % str(total_hashtags)
		with open(fname, 'w') as f:
			for tweet in tweets:
				f.write(tweet['hashtag']+","+str(tweet['weekday'])+","+tweet['hour']+","+str(tweet['score'])+"\n")
		pass
		print("=======================\n\nTotal hashtag : "+ str(total_hashtags))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)