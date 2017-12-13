#!/usr/bin/env python
from __future__ import print_function
from bias import compute_bias, extract_bias_features
from flask import Flask, request, jsonify

app = Flask(__name__)


def handle_statement(stmt, features=False):
    """handle the statement returning a dictionary to return to the client"""
    print(stmt)
    result = {}
    result['bias'] = compute_bias(stmt)
    result['statment'] = stmt
    if features:
        exfeatures = extract_bias_features(stmt)
        result['features'] = exfeatures
    return result


def enumerate_pairs(json):
    """a generator of pairs of statement, feature tuples"""
    if isinstance(json, list):
        for elem in json:
            yield (elem['statement'], elem['features'])
    else:
        raise TypeError('Argument must be a list of dictionaries')

@app.route("/bias", methods=['POST', 'GET'])
def bias():
    if request.method == 'GET':
        msg = """You need to post the statement that you want as the body of the http request.
For example:\n
    curl -X POST -d '[{"statement":"Hello cruel world"}]' -H "Content-Type: application/json"  localhost:5000/bias\n

or

    curl -X POST -d '[{"features": 1, "statement":"Hello, the best of all possible, worlds "},\\
                      {"features":0, "statement":"Hello Cruel World"}]'\\
         -H "Content-Type: application/json"  localhost:5000/bias
"""
        return msg, 403
    else:
        reqjson = request.get_json()
        d = [handle_statement(stmt, feat) for stmt, feat in enumerate_pairs(reqjson)]
        return jsonify(d), 200

# def main(instream=stdin, outstream=stdout):
#     """handle a stream of stateents in line delimited text"""
#     while True:
#         stmt = instream.readline().strip()
#         if stmt == '':
#             print('EOF reached closing', file=stderr)
#             break
#         try:
#             handle_datement(stmt)
#         except KeyboardInterrupt:
#             print('Received SIGINT quiting', file=stderr)
#             break
#         except Exception as exp:
#             print(exp)
#             print('Handling Exception, continuing', file=stderr)
#             print('')
#             break
#             continue



# if __name__ == '__main__':
    # main()
