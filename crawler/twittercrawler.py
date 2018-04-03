from crawler.twitter import Twitter
import os
import json
from datetime import datetime

class TwitterCrawler:

	def __init__(self, count=10, c2=10):
		self.twitter_graph = Twitter()
		self.min_tweets = count
		self.min_accs = c2
		self.retrieved_tweets = 0
		self.retrieved_accounts = 0
		self.already_retrieved = list()

	def crawl_tweets_r(self, screen_name, depth):
	
		# Stop condition
		if self.retrieved_tweets < self.min_tweets and self.retrieved_accounts < self.min_accs:

			# Get Tweets from screen name
			filename = "data/datasets/%s_tweets.json" % screen_name
			# If file exist, don't recreate it
			if os.path.isfile(filename):
				print("skipped (already exist)")
			else:
				# Retrieve tweets
				tweets = self.twitter_graph.get_tweets_from_user(screen_name)
				# If no tweets, do not create file
				if tweets:
					self.retrieved_tweets += len(tweets)
					
					if len(tweets) > 0:
						self.retrieved_accounts += 1
						
					# Write File
					with open(filename, 'w') as f:
						f.write(json.dumps(tweets))
					pass
				
			# User feed is now retrieved
			self.already_retrieved.append(screen_name)
			
			print("node (" + str(self.retrieved_tweets) + "/" + str(self.min_tweets) + " tweets crawled on " + str(self.retrieved_accounts) + "/" + str(self.min_accs) + "), depth = " + str(depth) + "")	
			if depth < 20:
				# Get followers
				followers = self.twitter_graph.get_followers_list(screen_name)
				
				# Delete duplicates
				followers = list(set(followers) - set(self.already_retrieved))
				
				# Loop on each follower
				if followers:
					for follower in followers:
						self.crawl_tweets_r(follower, depth + 1)
				
		
	def crawl_tweets_from_user(self, screen_name):
		print("* Trying to crawl min. " + str(self.min_tweets) + " tweets from min. " + str(self.min_accs) + " accounts Twitter *")		
		self.crawl_tweets_r(screen_name, 0)
		print("* Finish : " + str(self.retrieved_tweets) + " tweets crawled from "+ str(self.retrieved_accounts) +" accounts Twitter *")
		# Write File
		output_name = "data/TwitterCrawl" + str( datetime.now().isoformat(timespec='seconds').replace(":","")  ) + ".json"
		with open(output_name, 'w') as outfile:
			json.dump(self.already_retrieved, outfile)
		
