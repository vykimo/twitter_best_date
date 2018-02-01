#!/usr/bin/env python
# encoding: utf-8
import argparse
from crawler.twittercrawler import TwitterCrawler

def parse_arguments():
	parser = argparse.ArgumentParser(description='Generate Word Cloud of popular hashtags of a profile.')
	parser.add_argument('-a, --at', required=True, dest='profile', action='store', help='The profile identifier of an user. Begin with @')
	parser.add_argument('-c, --count', required=True, type=int, dest='count', action='store', help='number of tweets to retrieve')
	return parser.parse_args()

def main(run_args):
	crawler = TwitterCrawler(run_args.count)
	crawler.crawl_tweets_from_user(run_args.profile)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)

