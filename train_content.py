#!/usr/bin/env python
# encoding: utf-8
from model.model import Model
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize a json.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d, --data', dest='data', type=open, help='data file with twitter accounts')
    group.add_argument('-f, --file', dest='file', type=open, help='model.json')
    parser.add_argument('-c, --cache', dest='cache', action='count', help='do not use cache')
    parser.add_argument('-t, --tuning', dest='tuning', action='count', help='tuning model')
    parser.add_argument('-s, --save', dest='save', action='count', help='save model for publishing')
    parser.add_argument('-svr', dest='svr', action='count', help='use svr instead of rf')
    return parser.parse_args()

def main(run_args):
	model = Model()
	if run_args.data:
		model.train(run_args.data, run_args.tuning, run_args.cache, run_args.save, run_args.svr)
	elif run_args.file:
		model.run(run_args.file)
				
if __name__ == "__main__":
    args = parse_arguments()
    main(args)