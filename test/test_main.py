from __future__ import print_function
import pytest

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




