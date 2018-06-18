#!/usr/bin/env python
# encoding: utf-8
import tweepy
import json

filename = "data/gathered/gathering_584753_300767-texts_tweets.json"
data = json.load(open(filename))
acc = []
print(len(data))
for d in data:
	if not d['user'] in acc:
		acc.append(d['user'])
		print(d['user'])

print(len(acc))