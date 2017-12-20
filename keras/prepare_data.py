import json
from datetime import datetime
from dateutil import parser

screen_name = "BarackObama"

fnam1 = 'data\%s_tweets.json' % screen_name
data = json.load(open(fnam1))
fname = 'data\%s_format.data' % screen_name

with open(fname, 'w') as f:
	for tweet in data:
		print tweet['date']
		hour = parser.parse(tweet['date'])
		#hour = datetime.strptime(tweet['date'], '%a %b %d %H:%M:%S %z %Y')
		f.write(str(int(hour.strftime('%w')))+","+str(int(hour.strftime('%H%M')))+","+ str(tweet['rt'] * 2 + tweet['fav']) +"\n")
pass