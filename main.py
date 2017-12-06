#!/usr/bin/env python
# encoding: utf-8

import argparse
from gethashtag import *
from gethours import *
from gettweets import *
import config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate Word Cloud of popular hashtags of a profile.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a, --at', dest='profile', action='store', help='The profile identifier of an user. Begin with @')
    parser.add_argument('-s, --skip', dest='skip', action='store_true', help='skip the first step')
    parser.add_argument('-e, --hour', dest='hour', action='store_true', help='to show popularity by hours graph')
    return parser.parse_args()



def main(run_args):
	if run_args.profile:
		if not run_args.skip:
			fname = get_all_tweets(run_args.profile,config.consumer_key,config.consumer_secret,config.access_token,config.access_secret)
		else:
			fname = 'data\%s_tweets.json' % run_args.profile
	
		if run_args.hour:
			get_hoursdays(fname)
		else:
			get_hashtag(fname)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

