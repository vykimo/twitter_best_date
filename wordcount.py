#!/usr/bin/env python
# encoding: utf-8
import argparse
import json
from datetime import datetime
from dateutil import parser
import re
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f, --file', dest='file', action='store', help='file with twitter accounts')
    return parser.parse_args()


def main(run_args):

	if run_args.file:
		accounts = open(run_args.file)
		
		count_all_p = Counter()
		for account in accounts:
		
			account_name = account.strip('\n');
			print("\n### Counts different hashtags from : "+account_name+" ###")
	
			data = json.load(open('data\%s_tweets_normalized.json' % account_name))
			
			for d in data:
				count_all_p.update([d['hashtag']])
			
		print(count_all_p)
		print(len(count_all_p))
		
		while 1:
			doc = input("What hashtag do you want to test?\n")
			if doc in count_all_p:
				print(doc + " = " + str(count_all_p[doc]))
			else:
				print("Hashtag not found")
			

if __name__ == "__main__":
    args = parse_arguments()
    main(args)