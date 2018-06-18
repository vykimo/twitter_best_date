import argparse
from flask import Flask, request, Response, current_app, json
from crawler.twitter import Twitter
from model.model import Model
import time
from datetime import date

def parse_arguments():
    parser = argparse.ArgumentParser(description='Starts a server that queries a pretrained model.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f, --file', dest='file', help='model.json')
    return parser.parse_args()


args = parse_arguments()
model_data = args.file
server = Flask(__name__)


@server.route('/', methods=['GET'])
def root():
    return current_app.send_static_file('index.html')

	
@server.route("/score", methods=['GET'])
def score():
    today = date.today()
    data = dict()
    data['text'] = request.args.get('text', default=None);
    data['followers_count'] = request.args.get('followers_count', default=0);
    data['friends_count'] = request.args.get('friends_count', default=0);
    data['listed_count'] = request.args.get('listed_count', default=0);
    data['statuses_count'] = request.args.get('statuses_count', default=0);
    data['weekday'] = int(today.strftime('%w'))
    data['hour'] = today.strftime('%H:%M')
    if data:
        model = Model()
        res = model.predict(open(model_data), data, True, False)
        print(res)
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    else:
        return Response(response="Query parameter 'data' is missing.", status=400, mimetype='text/plain')

		
@server.route("/ask", methods=['GET'])
def ask():
    data = dict()
    data['text'] = request.args.get('text', default=None);
    data['followers_count'] = request.args.get('followers_count', default=0);
    data['friends_count'] = request.args.get('friends_count', default=0);
    data['listed_count'] = request.args.get('listed_count', default=0);
    data['statuses_count'] = request.args.get('statuses_count', default=0);
    if data:
        model = Model()
        model.predict(open(model_data), data, True)
        reaction = json.load(open("data/publish/webdataset.json"))    
        return Response(response=json.dumps(reaction), status=200, mimetype='application/json')
    else:
        return Response(response="Query parameter 'data' is missing.", status=400, mimetype='text/plain')

		
@server.route("/account", methods=['GET'])
def account():
    screen_name = request.args.get('post', default=None)
    if screen_name:        
        twitter_graph = Twitter()
        info = twitter_graph.get_user_infos(screen_name)
        if info:
            info = info._json
            infos = {'id': info["id"], 'profile_image_url_https': info["profile_image_url_https"], 'followers_count': info["followers_count"], 'friends_count': info["friends_count"], 'listed_count': info["listed_count"], 'statuses_count': info["statuses_count"]}
            return Response(response=json.dumps(infos), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({'id':0}), status=200, mimetype='text/plain')
    else:
        return Response(response="Query parameter 'screen_name' is missing.", status=400, mimetype='text/plain')
		
server.run(host='192.168.1.97', threaded=True)
