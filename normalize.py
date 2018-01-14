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
    return parser.parse_args()


def main(run_args):

	if run_args.file:
		accounts = open(run_args.file)
		total_tweets = 0
		total_tweets_used = 0
		total_tweets_skipped = 0
		total_hashtag_skipped = 0
		total_hashtags = 0
		for account in accounts:
			temp_total_tweets = 0
			temp_total_tweets_used = 0
			temp_total_tweets_skipped = 0
			temp_total_hashtag_skipped = 0
			temp_total_hashtags = 0
			account_name = account.strip('\n');
			print("\n### Normalize tweets from : "+account_name+" ###")
	
			data = json.load(open('data\%s_tweets.json' % account_name))
			
			with open('data\%s_tweets_normalized.json' % account_name, 'w') as f:
				f.write("[")
				for tweet in data:
					temp_total_tweets += 1
					
					if len(tweet['hashtags']) == 0:
						temp_total_tweets_skipped += 1
						
					else:
						temp_total_tweets_used += 1
						for hashtag in tweet['hashtags']:
							if re.match("^[a-zA-Z0-9àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßÆæœ_-]*$", hashtag):
								hour = parser.parse(tweet['date'])						
								f.write("{\"hashtag\": \""+hashtag+"\", \"weekday\": "+str(int(hour.strftime('%w')))+", \"hour\": \""+str(hour.strftime('%H:%M'))+"\", \"score\": "+ str(tweet['rt'] * 2 + tweet['fav']) +"}")
								f.write(",")
								temp_total_hashtags += 1
							
							else:
								print(hashtag)
								temp_total_hashtag_skipped += 1
			pass
			with open('data\%s_tweets_normalized.json' % account_name, 'rb+') as f:
				f.seek(0,2)                 # end of file
				size=f.tell()               # the size...
				f.truncate(size-1)          # truncate at that size - how ever many characters
			pass
			with open('data\%s_tweets_normalized.json' % account_name, 'a') as f:
				f.write("]")
			pass
			print("\n\nTotal tweets : "+ str(temp_total_tweets))
			print("Total tweets used : "+ str(temp_total_tweets_used))
			print("Total tweets skipped : "+ str(temp_total_tweets_skipped))
			print("Total hashtag skipped : "+ str(temp_total_hashtag_skipped))
			print("Total rows (Number of hashtags) : "+ str(temp_total_hashtags))
			
			total_tweets += temp_total_tweets
			total_tweets_used += temp_total_tweets_used
			total_tweets_skipped += temp_total_tweets_skipped
			total_hashtags += temp_total_hashtags
			total_hashtag_skipped += temp_total_hashtag_skipped
			
		print("======================\n\nTotal tweets : "+ str(total_tweets))
		print("Total tweets used : "+ str(total_tweets_used))
		print("Total tweets skipped : "+ str(total_tweets_skipped))
		print("Total hashtag skipped : "+ str(total_hashtag_skipped))
		print("Total rows (Number of hashtags) : "+ str(total_hashtags))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)