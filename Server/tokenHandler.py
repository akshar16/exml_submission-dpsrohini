from flask import blueprints
import flask
from tokens import get_entry,get_all,enter_queue,get_token


token_handler = blueprints.Blueprint('token_handler', __name__)




@token_handler.route('/getToken',methods=['POST'])
def getTtoken():
    email = flask.request.get_json()
  
    if email == None:
        return flask.jsonify({"error":"Missing email"}),400
    status = get_token(email.get('email'))
    if status == "Token already sent":
        return flask.jsonify({"error":"Token already sent"}),400
    if status == "invalid email":
        return flask.jsonify({"error":"Invalid email"}),400
    return flask.jsonify({"status":status})

@token_handler.route('/enterQueue',methods=['POST'])
def enterQueue():
    token = flask.request.get_json()
    if token == None:
        return flask.jsonify({"error":"Missing token"}),400
    status = enter_queue(token.get('token'))
    if status == "Token already in queue":
        return flask.jsonify({"error":"Token already in queue"}),400
    if status == "Token not found":
        return flask.jsonify({"error":"Token not found"}),400
    return flask.jsonify({"status":status})

    


@token_handler.route('/queue',methods=['GET'])
def queue():
    return flask.jsonify({"queue":get_all()})

@token_handler.route('/get',methods=['GET'])
def get():
   
    token = flask.request.args.get('token')
    
    if token == None:
        return flask.jsonify({"error":"Missing token"}),400
    status = get_entry(token)
    if status == None:
        return flask.jsonify({"error":"Token not found"}),401
    return flask.jsonify({"status":status})

