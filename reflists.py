from __future__ import print_function
import json
import os
import sys

wordsets = {}

for filename in sys.argv[1:]:
    print('Working on: ' + filename, file=sys.stderr)
    with open(filename, 'r') as fp:
        try:
            words = fp.read().lower().split('\n')
            key = os.path.basename(filename)
            wordsets[key] = words
        except UnicodeDecodeError as exp:
            print(exp, file=sys.stderr)

print(json.dumps(wordsets, indent=2))
