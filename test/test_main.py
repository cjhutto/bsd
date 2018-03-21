from __future__ import print_function

import os
import pytest

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def test_imports():
    import bsdetector
    import vaderSentiment
    from bsdetector import bias
    return True


def assert_statement(stat):
    from bsdetector import bias
    r = bias.compute_bias(stat)
    print(stat)
    print(r)
    return r


def test_compute():
    from bsdetector import bias
    assert_statement("The cat sucks.")
    fpath = 'input.txt'
    bias.enumerate_sentences(fpath)
    bias.compute_bias('brexit.txt')


def testnormalize():
    from bsdetector import bias
    text = 'the cat sucks'
    features = bias.extract_bias_features(text)
    print(bias.normalized_features(features))


def _get_print_feature_data(filename, output):
    from bsdetector.bias import get_text_from_article_file, print_feature_data
    sentence_list = get_text_from_article_file(filename).split('\n')
    out = StringIO()
    print_feature_data(sentence_list, output_type=output, fileout=out)
    out.seek(0)
    return out


def test_print_json_feature_data():
    test_path = os.path.dirname(os.path.abspath(__file__))
    output = _get_print_feature_data(os.path.join(test_path, 'brexit.txt'), 'json')
    with open(os.path.join(test_path, 'brexit.json'), 'r') as test_json:
        assert test_json.read() == output.read()

