import tweepy

class Twitter:

	CONSUMER_KEY  = '8InW60jwG61XtwGNyHFU4IbiK'
	CONSUMER_SECRET  = 'AgsKb07ZS9MeWpM8qhOcxTruTgmpUMPELVULZzTt0A2y7sfc99'
	ACCESS_TOKEN  = '936177469853454336-tKbX8leTNrQG7uKZR5qS7hXQ1O3mJUA'
	ACCESS_SECRET  = 'fVEE9Bd8AvjSSHkm4CBUPgLMzq0TkhZI2S8GTeYajjH1y'

	def __init__(self):
	
		#authorize twitter, initialize tweepy
		auth = tweepy.OAuthHandler(Twitter.CONSUMER_KEY, Twitter.CONSUMER_SECRET)
		auth.set_access_token(Twitter.ACCESS_TOKEN, Twitter.ACCESS_SECRET)
		self.api = tweepy.API(auth, wait_on_rate_limit=True)

		
	#Twitter only allows access to a users most recent 3240 tweets with this method
	def get_tweets_from_user(self, screen_name):
	
		# Message
		print("\n### Crawl last tweets from : "+screen_name+" ###")
	
		#initialize a list to hold all the tweepy Tweets
		alltweets = []	
		outtweets = []
		
		try:
			#make initial request for most recent tweets (200 is the maximum allowed count)
			new_tweets = self.api.user_timeline(screen_name = screen_name, count=200, exclude_replies=True)
			
			#save most recent tweets
			alltweets.extend(new_tweets)
			
			#if no tweets
			if len(alltweets) < 1:
				return []
			
			#save the id of the oldest tweet less one
			oldest = alltweets[-1].id - 1
			
			#keep grabbing tweets until there are no tweets left to grab
			while len(new_tweets) > 0:
				
				print ("%s tweets retrieved" % (len(new_tweets)))
			
				#all subsiquent requests use the max_id param to prevent duplicates
				new_tweets = self.api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, exclude_replies=True)
				
				#save most recent tweets
				alltweets.extend(new_tweets)
				
				#if no tweets
				if len(alltweets) < 1:
					return []
					
				#update the id of the oldest tweet less one
				oldest = alltweets[-1].id - 1
			
			# Normalize tweets
			for tweet in alltweets:
			
				t = tweet._json
				hashtags = []
				
				for hashtag in tweet.entities['hashtags']:
					hashtags.extend([hashtag['text']])
					
				to_insert = [{'id':tweet.id_str, 'date':str(t['created_at']), 'text':tweet.text, 'rt': tweet.retweet_count, 'fav': tweet.favorite_count, 'hashtags':hashtags}]
				
				outtweets.extend(to_insert)
				
			print ("= %s tweets downloaded" % (len(outtweets)))
			
		except tweepy.TweepError:
			print("Failed to retrieve tweets, Skipping...")
			
		return outtweets
	
	
	def get_followers_list(self, screen_name):
	
		# Message
		print("### Get Followers from : "+screen_name+" ###")
		
		followers = []
		try:
			for user in self.api.followers(screen_name=screen_name, count=200):
				followers.append(user.screen_name)
		except tweepy.TweepError:
			print("Failed to find followers, Skipping...")
		print("### " + str(len(followers)) + " followers found ###")
		return followers
		
		