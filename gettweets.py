#!/usr/bin/env python
# encoding: utf-8
import tweepy
import json
from crawler.twitter import Twitter

def get_all_tweets(screen_name,consumer_key,consumer_secret,access_token,access_secret):
	
	twitter_graph = Twitter()
	outtweets = twitter_graph.get_tweets_from_user(screen_name)
	
	#write
	fname = 'data\%s_tweets.json' % screen_name
	with open(fname, 'w') as f:
		f.write(json.dumps(outtweets))
	pass
	
	return fname,len(outtweets)