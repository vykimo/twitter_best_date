# Best tweet post date prediction
Individual interdisciplinary project 2017/2018, Supervisors: Tobias Scheffer &amp; Paul Prasse, Submission Date: 02.03.2018 @ University of Potsdam, Germany

## Install

This project is working on Anaconda with :
> Python 3.6.3

Run in console:
```
pip install -r requirements.txt
```
Set your Twitter API keys in `config.py`.
```
#Twitter API credentials
consumer_key  = '#####'
consumer_secret  = '#####'
access_token  = '#####'
access_secret  = '#####'
```

## Tweet crawling
Create a folder _./data/datasets/_ in root.
Run in console:
```
python crawltwitter.py -a [twitter_user] -c1 [max_tweets] -c2 [max_accounts]
```
p(ex) : 
```
python crawltwitter.py -a DataScienceCtrl -c1 10000 -c2 100
```

Data for each accounts retrieved will be stored in _data/datasets_.  
The account names retrieved by this crawling will be stored in _data/TwitterCrawlXXXX-XX-XXXXX.json_

## Gather data
Create a folder _./data/gathered/_ in root.
Run in console:
```
python gather.py -max 10000
```
A json file will be created in _data/gathered_ with datas ready for training.  

## Train models
Create folders _./data/cache/_ and _./data/publish/_ in root.
Run in console:
```
python training.py -d data\gathered\gathering_xxx_xxx.json -save
```
Feature importance will be displayed.
Baselines tests too.
Then, model will be saved in _./data/publish/_.

## Running web server
Run in console:
```
python server.py -f data\publish\xxxxxx.model.json
```
Then, go to http://127.0.0.1:5000 and let play with predictions.
