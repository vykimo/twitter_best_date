#!/usr/bin/env python
# encoding: utf-8
import argparse
from crawler.twittercrawler import TwitterCrawler

def parse_arguments():
	parser = argparse.ArgumentParser(description='Crawl profile.')
	parser.add_argument('-a, --at', required=True, dest='profile', action='store', help='The profile identifier of an user. Begin with @')
	parser.add_argument('-c1, --count1', required=True, type=int, dest='count1', action='store', help='number of tweets to retrieve')
	parser.add_argument('-c2, --count2', required=True, type=int, dest='count2', action='store', help='number of accounts to retrieve')
	return parser.parse_args()

def main(run_args):
	crawler = TwitterCrawler(run_args.count1, run_args.count2)
	crawler.crawl_tweets_from_user(run_args.profile)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)

