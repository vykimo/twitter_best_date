import json
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

def get_hashtag(fname):
	data = json.load(open(fname))
	
	# Part 1 : Word Cloud
	count_all_c = Counter()
	count_all_p = Counter()
	for tweet in data:
		terms_hash_count = tweet['hashtags']
		terms_hash_pop = tweet['hashtags'] * (tweet['rt'] * 2 + tweet['fav'])
		count_all_c.update(terms_hash_count)
		count_all_p.update(terms_hash_pop)

	wordcloud_c = WordCloud()
	wordcloud_p = WordCloud()
	wordcloud_c.generate_from_frequencies(frequencies=count_all_c)
	c = dict(count_all_c)
	a = {k: v / c[k] for k, v in dict(count_all_p).items()}
	wordcloud_p.generate_from_frequencies(frequencies=Counter(a))
	print(count_all_p.most_common(5))
	print(Counter(a).most_common(5))
	
	plt.figure(1)
	plt.imshow(wordcloud_c, interpolation="bilinear")
	plt.axis("off")
	plt.title("Hashtag number")
	
	plt.figure(2)
	plt.imshow(wordcloud_p, interpolation="bilinear")
	plt.title("Hashtag popularity")
	plt.axis("off")
	
	
	# Part 2
	plt.figure(3)
	x = []
	y = []	
	name = []
	plt.title("Hashtag Clusters")
	
	# clusters des hashtags
	for tweet in data:
		for h in tweet['hashtags']:
			x.append(tweet['rt'])
			y.append(tweet['fav'])
			name.append(h)
	label_unique = list(set(name))
	
	for i in range(0,len(label_unique)):
		x_temp = []
		y_temp = []
		label = label_unique[i]
		for j in range(0,len(name)):
			if name[j] == label:
				x_temp.append(x[j])
				y_temp.append(y[j])
		plt.plot(x_temp,y_temp,'.',label=label)
	
	plt.xlabel('RT')
	plt.ylabel('Fav')
	
	#plt.legend()
	plt.show()
	