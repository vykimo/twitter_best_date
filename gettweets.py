#!/usr/bin/env python
# encoding: utf-8
import tweepy
import json

def get_all_tweets(screen_name,consumer_key,consumer_secret,access_token,access_secret):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
	api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	outtweets = []
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print ("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print ("...%s tweets downloaded so far" % (len(alltweets)))
	
	for tweet in alltweets:
		t = tweet._json
		hashtags = []
		for hashtag in tweet.entities['hashtags']:
			hashtags.extend([hashtag['text']])
		to_insert = [{'id':tweet.id_str, 'date':str(t['created_at']), 'text':tweet.text, 'rt': tweet.retweet_count, 'fav': tweet.favorite_count, 'hashtags':hashtags}]
		outtweets.extend(to_insert)
	
	#write
	fname = 'data\%s_tweets.json' % screen_name
	with open(fname, 'w') as f:
		f.write(json.dumps(outtweets))
	pass
	
	return fname