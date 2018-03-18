#!/usr/bin/python
# coding: utf-8
"""
Bias Sentence Investigator (BSI): Detecting and Quantifying the Degree of Bias in Text
Created on June 04, 2015
@author: C.J. Hutto
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import json
import multiprocessing
import os
import re
import sys
from collections import OrderedDict
from decorator import contextmanager
from pattern.text.en import Sentence, parse, modality
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as Vader_Sentiment
from bsdetector.caster import caster


class Lexicons(object):
    """Lexicon is a class with static members for managing the existing lists of words.
    Use Lexicon.list(key) in order to access the list with name key.
    """
    pth = os.path.join(os.path.dirname(__file__), 'lexicon.json')
    if os.path.isfile(pth):
        with open(pth, 'r') as filp:
            wordlists = json.loads(filp.read())
    else:
        print(pth, "... file does not exist.")
        wordlists = {}
    # print(list(wordlists.keys()))

    @classmethod
    def list(cls, name):
        """list(name) get the word list associated with key name"""
        return cls.wordlists[name]


def get_text_from_article_file(article_file_path):
    with open(article_file_path, "r") as filep:
        lst = filep.read()
    return lst


def append_to_file(file_name, line):
    """append a line of text to a file"""
    with open(file_name, 'a') as filep:
        filep.write(line)
        filep.write("\n")


def split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    if "e.g." in text:
        text = text.replace("e.g.", "e<prd>g<prd>")
    if "i.e." in text:
        text = text.replace("i.e.", "i<prd>e<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace("\n", " <stop>")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def syllable_count(text):
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    text = "".join(x for x in text if x not in exclude)

    if text is None:
        return 0
    elif len(text) == 0:
        return 0
    else:
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = count - (0.1 * count)
        return count


def lexicon_count(text, removepunct=True):
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    if removepunct:
        text = ''.join(ch for ch in text if ch not in exclude)
    count = len(text.split())
    return count


def sentence_count(text):
    ignore_count = 0
    sentences = split_into_sentences(text)
    for sentence in sentences:
        if lexicon_count(sentence) <= 2:
            ignore_count = ignore_count + 1
    sentence_cnt = len(sentences) - ignore_count
    if sentence_cnt < 1:
        sentence_cnt = 1
    return sentence_cnt


def avg_sentence_length(text):
    lc = lexicon_count(text)
    sc = sentence_count(text)
    a_s_l = float(old_div(lc, sc))
    return round(a_s_l, 1)


def avg_syllables_per_word(text):
    syllable = syllable_count(text)
    words = lexicon_count(text)
    try:
        a_s_p_w = old_div(float(syllable), float(words))
        return round(a_s_p_w, 1)
    except ZeroDivisionError:
        # print "Error(ASyPW): Number of words are zero, cannot divide"
        return 1


def flesch_kincaid_grade(text):
    a_s_l = avg_sentence_length(text)
    a_s_w = avg_syllables_per_word(text)
    f_k_r_a = float(0.39 * a_s_l) + float(11.8 * a_s_w) - 15.59
    return round(f_k_r_a, 1)


def count_feature_freq(feature_list, tokens_list, txt_lwr):
    cnt = 0
    # count unigrams
    for w in tokens_list:
        if w in feature_list:
            cnt += 1
        # count wildcard features
        for feature in feature_list:
            if str(feature).endswith('*') and str(w).startswith(feature[:-1]):
                cnt += 1
    # count n_gram phrase features
    for feature in feature_list:
        if " " in feature and feature in txt_lwr:
            cnt += str(txt_lwr).count(feature)
    return cnt


def check_quotes(text):
    quote_info = dict(has_quotes=False,
                      quoted_list=None,
                      mean_quote_length=0,
                      nonquoted_list=split_into_sentences(text),
                      mean_nonquote_length=avg_sentence_length(text))
    quote = re.compile(r'"([^"]*)"')
    quotes = quote.findall(text)
    if len(quotes) > 0:
        quote_info["has_quotes"] = True
        quote_info["quoted_list"] = quotes
        total_qte_length = 0
        nonquote = text
        for qte in quotes:
            total_qte_length += avg_sentence_length(qte)
            nonquote = nonquote.replace(qte, "")
            nonquote = nonquote.replace('"', '')
            re.sub(r'[\s]+', ' ', nonquote)
        quote_info["mean_quote_length"] = round(old_div(float(total_qte_length), float(len(quotes))), 4)
        nonquotes = split_into_sentences(nonquote)
        if len(nonquotes) > 0:
            quote_info["nonquoted_list"] = nonquotes
            total_nqte_length = 0
            for nqte in nonquotes:
                total_nqte_length += avg_sentence_length(nqte)
            quote_info["mean_nonquote_length"] = round(old_div(float(total_nqte_length), float(len(nonquotes))), 4)
        else:
            quote_info["nonquoted_list"] = None
            quote_info["mean_nonquote_length"] = 0

    return quote_info


def check_neg_persp(input_words, vader_neg, vader_compound, include_nt=True):
    """
    Determine the degree of negative perspective of text
    Returns an float for score (higher is more negative)
    """
    neg_persp_score = 0.0
    neg_words = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
                  "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
                  "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
                  "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
                  "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
                  "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
                  "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
                  "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
    for word in neg_words:
        if word in input_words:
            neg_persp_score += 1
    if include_nt:
        for word in input_words:
            if "n't" in word and word not in neg_words:
                neg_persp_score += 1
    if vader_neg > 0.0:
        neg_persp_score += vader_neg
    if vader_compound < 0.0:
        neg_persp_score += abs(vader_compound)
    return neg_persp_score


def get_caster(text, top_n=10):
    """ Contextual Aspect Summary and Topical-Entity Recognition
        Returns a Python dictionary {KeyWordPhrase : Importance_Score} of the top-N  most important contextual aspects
    """
    cstr_dict = OrderedDict()
    contextual_aspect_summary = caster(text, sort_by="both", term_freq_threshold=2, cos_sim_threshold=0.01, top_n=top_n)
    for keywordphrase, score in contextual_aspect_summary:
        cstr_dict[keywordphrase] = round(score, 3)
    return cstr_dict


ref_lexicons = Lexicons()

##### List of presupposition verbs (comprised of Factive, Implicative, Coherence, Causation, & Assertion markers):
### Factive verbs derived from:
# Paul Kiparsky and Carol Kiparsky. 1970. Fact. In M.Bierwisch and K.E.Heidolph, editors, Progress in
#  Linguistics, pages 143–173.Mouton, The Hague.
### Implicative verbs derived from
# Lauri Karttunen. 1971. Implicative verbs. Language, 47(2):340–358.
##### List of coherence markers derived from:
# Knott, Alistair. 1996. A Data-Driven Methodology for Motivating a Set of
#  Coherence Relations. Ph.D. dissertation, University of Edinburgh, UK.
##### List of assertive derived from:
# Joan B. Hooper. 1975. On assertive predicates. In J. Kimball, editor,
#  Syntax and Semantics, volume 4, pages 91–124. Academic Press, New York.
##### List of Causation words from LIWC
#########################################################################
presup = ref_lexicons.list('presupposition')

##### List of hedge words derived from:
# Ken Hyland. 2005. Metadiscourse: Exploring Interaction in Writing.
# Continuum, London and New York.
##### List of tentative words from LIWC
##### List of NPOV hedge & "weasel" words to watch from
# https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch
#########################################################################
doubt = ref_lexicons.list('doubt_markers')

##### List of biased/partisan words derived from:
# Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan Jurafsky. 2013. Linguistic Models for
#     Analyzing and Detecting Biased Language. Proceedings of ACL 2013.
# and
# Gentzkow, Econometrica 2010: What Drives Media Slant? Evidence from U.S. Daily Newspapers
#########################################################################
partisan = ref_lexicons.list('partisan')

##### List of opinion laden words extracted from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
##### List of strong/weak subjective words extracted from:
# Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
# Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
# Vancouver, Canada.
##### List of degree modifiers derived from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
#########################################################################
value_laden = ref_lexicons.list('value_laden')
vader_sentiment_analysis = Vader_Sentiment()

##### List of figurative expressions derived from:
###English-language idioms
# https://en.wikipedia.org/wiki/English-language_idioms.
# and
### List of English-language metaphors
# https://en.wikipedia.org/wiki/List_of_English-language_metaphors
# and
### List of political metaphors
# https://en.wikipedia.org/wiki/List_of_political_metaphors
### List of NPOV "puffery & peacock" words to watch from
# https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch
#########################################################################
figurative = ref_lexicons.list('figurative')

##### Lists of attribution bias/actor-observer bias/ultimate attribution markers
# LIWC 3rd person pronouns (combines S/he and They)
# LIWC achievement words
# LIWC work words
attribution = ref_lexicons.list('attribution')

#### List of self reference pronouns from LIWC
self_refer = ref_lexicons.list('self_reference')


def extract_bias_features(text, do_get_caster=False):
    features = OrderedDict()
    acsiitext = text
    text_nohyph = acsiitext.replace("-", " ")  # preserve hyphenated words as separate tokens
    txt_lwr = str(text_nohyph).lower()
    words = ''.join(ch for ch in txt_lwr if ch not in '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~').split()
    unigrams = sorted(list(set(words)))
    bigram_tokens = find_ngrams(words, 2)
    bigrams = [" ".join([w1, w2]) for w1, w2 in sorted(set(bigram_tokens))]
    trigram_tokens = find_ngrams(words, 3)
    trigrams = [" ".join([w1, w2, w3]) for w1, w2, w3 in sorted(set(trigram_tokens))]

    ## SENTENCE LEVEL MEASURES
    # word count
    features['word_cnt'] = len(words)

    # unique word count
    features['unique_word_cnt'] = len(unigrams)

    # Flesch-Kincaid Grade Level (reading difficulty) using textstat
    features['fk_gl'] = flesch_kincaid_grade(text)

    # compound sentiment score using VADER sentiment analysis package
    vader_sentiment = vader_sentiment_analysis.polarity_scores(text)
    vader_negative_proportion = vader_sentiment['neg']
    vader_compound_sentiment = vader_sentiment['compound']
    features['vader_sentiment'] = vader_compound_sentiment
    features['vader_senti_abs'] = abs(vader_compound_sentiment)

    # negative-perspective
    features['neg_persp'] = check_neg_persp(words, vader_negative_proportion, vader_compound_sentiment)

    # modality (certainty) score and mood using  http://www.clips.ua.ac.be/pages/pattern-en#modality
    sentence = parse(text, lemmata=True)
    sentence_obj = Sentence(sentence)
    features['certainty'] = round(modality(sentence_obj), 4)

    # quoted material
    quote_dict = check_quotes(text)
    features["has_quotes"] = quote_dict["has_quotes"]
    features["quote_length"] = quote_dict["mean_quote_length"]
    features["nonquote_length"] = quote_dict["mean_nonquote_length"]

    ## LEXICON LEVEL MEASURES
    # presupposition markers
    count = count_feature_freq(presup, words, txt_lwr)
    features['presup_cnt'] = count
    features['presup_rto'] = round(old_div(float(count), float(len(words))), 4)

    # doubt markers
    count = count_feature_freq(doubt, words, txt_lwr)
    features['doubt_cnt'] = count
    features['doubt_rto'] = round(old_div(float(count), float(len(words))), 4)

    # partisan words and phrases
    count = count_feature_freq(partisan, words, txt_lwr)
    features['partisan_cnt'] = count
    features['partisan_rto'] = round(old_div(float(count), float(len(words))), 4)

    # subjective value laden word count
    count = count_feature_freq(value_laden, words, txt_lwr)
    features['value_cnt'] = count
    features['value_rto'] = round(old_div(float(count), float(len(words))), 4)

    # figurative language markers
    count = count_feature_freq(figurative, words, txt_lwr)
    features['figurative_cnt'] = count
    features['figurative_rto'] = round(old_div(float(count), float(len(words))), 4)

    # attribution markers
    count = count_feature_freq(attribution, words, txt_lwr)
    features['attribution_cnt'] = count
    features['attribution_rto'] = round(old_div(float(count), float(len(words))), 4)

    # self reference pronouns
    count = count_feature_freq(self_refer, words, txt_lwr)
    features['self_refer_cnt'] = count
    features['self_refer_rto'] = round(old_div(float(count), float(len(words))), 4)

    # Contextual Aspect Summary and Topical-Entity Recognition (CASTER)
    if do_get_caster:
        """ May incur a performance cost in time to process """
        caster_dict = get_caster(text)
        features['caster_dict'] = caster_dict

    return features


modelbeta = [0.844952,
             -0.015031,
             0.055452,
             0.064741,
             -0.018446,
             -0.008512,
             0.048985,
             0.047783,
             0.028755,
             0.117819,
             0.269963,
             -0.041790,
             0.129693]

modelkeys = ['word_cnt',
             'vader_senti_abs',
             'neg_persp',
             'certainty',
             'quote_length',
             'presup_cnt',
             'doubt_cnt',
             'partisan_cnt',
             'value_cnt',
             'figurative_cnt',
             'attribution_cnt',
             'self_refer_cnt']


def featurevector(features):
    """Extract the features into a vector in the right order, prepends a 1 for constant term."""
    l = [1]
    l.extend(features[k] for k in modelkeys)
    return l


def normalized_features(features):
    """Normalize the features by dividing by the coefficient."""
    beta = modelbeta
    fvec = featurevector(features)
    norm = lambda i: old_div(fvec[i], modelbeta[i])
    return [norm(i) for i in range(len(modelbeta))]


def compute_bias(sentence_text):
    """run the trained regression coefficients against the feature dict"""
    features = extract_bias_features(sentence_text)
    coord = featurevector(features)
    bs_score = sum(modelbeta[i] * coord[i] for i in range(len(modelkeys)))
    return bs_score


@contextmanager
def poolcontext(*args, **kwargs):
    """poolcontext makes it easier to run a function with a process Pool.

    Example:

            with poolcontext(processes=n_jobs) as pool:
                bs_scores = pool.map(compute_bias, sentences)
                avg_bias = sum(bs_scores)
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def roundmean(avg_bias, sentences, k=4):
    """Compute the average and round to k places"""
    avg_bias = round(old_div(float(avg_bias), float(len(sentences))), k)
    return avg_bias


def compute_avg_statement_bias_mp(statements_list_or_str, n_jobs=1):
    """compute_statement_bias_mp a version of compute_statement_bias
    with the multiprocessing pool manager."""
    sentences = list()
    if not isinstance(statements_list_or_str, list):
        if isinstance(statements_list_or_str, str):
            sentences.extend(split_into_sentences(statements_list_or_str))
        else:
            logmessage = "-- Expecting type(list) or type(str); type({}) given".format(type(statements_list_or_str))
            print(logmessage)
    # max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    with poolcontext(processes=n_jobs) as pool:
        bs_scores = pool.map(compute_bias, sentences)
        total_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(total_bias, sentences)
    else:
        avg_bias = 0

    return avg_bias


def compute_avg_statement_bias(statements_list_or_str):
    """compute the bias of a statement from the test.
    returns the average bias over the entire text broken down by sentence.
    """
    sentences = list()
    if not isinstance(statements_list_or_str, list):
        if isinstance(statements_list_or_str, str):
            sentences.extend(split_into_sentences(statements_list_or_str))
        else:
            logmessage = "-- Expecting type(list) or type(str); type({}) given".format(type(statements_list_or_str))
            print(logmessage)

    # max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    bs_scores = []
    for sent in sentences:
        bs_scores.append(compute_bias(sent))

    total_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(total_bias, sentences)
    else:
        avg_bias = 0

    return avg_bias


def make_tsv_output(list_of_sentences):
    """print out a table of output as a tab separated file."""
    # make tab seperated values
    keys_done = False
    logmessage = "-- Example TSV: paste the following into Excel, Data-->Text To Columns-->Delimited-->Tab-->Finish"
    print(logmessage, file=sys.stderr)
    tsv_output = ''
    for sent in list_of_sentences:
        if len(sent) >= 1:
            feature_data = extract_bias_features(sent)
            if not keys_done:
                tsv_output = 'sentence\t' + '\t'.join(list(feature_data.keys())) + '\n'
                keys_done = True
            str_vals = [str(f) for f in list(feature_data.values())]
            tsv_output += sent + '\t' + '\t'.join(str_vals) + '\n'
    return tsv_output


def make_dict_output(list_of_sentences):
    data = []
    for sent in list_of_sentences:
        if len(sent) >= 1:
            feature_data = extract_bias_features(sent)
            feature_data['text'] = sent
            data.insert(0, feature_data)
    return data


def make_json_output(list_of_sentences):
    data = make_dict_output(list_of_sentences)
    return json.dumps(data, indent=2)


def make_html_output(list_of_sentences):
    """create a table of output as an html table."""
    # make HTML table
    sep = '</td><td>'
    hsep = '</th><th>'
    keys_done = False
    logmessage = "-- Example HTML: paste the following in a text editor, save it as 'bias.html', then open with browser"
    print(logmessage)
    html_output = '<html><body><table border="1">'
    for sent in list_of_sentences:
        if len(sent) > 3:
            feature_data = extract_bias_features(sent)
            if not keys_done:
                html_output += '<tr><th>sentence' + hsep + hsep.join(list(feature_data.keys())) + '</th></tr>'
                keys_done = True
            str_vals = [str(f) for f in list(feature_data.values())]
            html_output += '<tr><td>' + sent + sep + sep.join(str_vals) + '</td></tr>'
    html_output += '</table></body></html>'
    return html_output


def print_feature_data(list_of_sentences, output_type='tsv', fileout=sys.stdout):
    """print the data in either html or tsv format"""
    output = ' -- no output available'
    if output_type == 'html':
        output = make_html_output(list_of_sentences)
    elif output_type == 'tsv':
        output = make_tsv_output(list_of_sentences)
    elif output_type == 'json':
        output = make_json_output(list_of_sentences)
    print(output, file=fileout)


def enumerate_sentences(fpath='input_text'):
    """print the bias of each sentence in a document."""
    sentences_list = get_text_from_article_file(fpath).split('\n')
    for statement in sentences_list:
        if len(statement) >= 3:
            biasq = compute_bias(statement)
            yield(biasq, statement)
        else:
            print('-- Statement is too short: {}'.format(statement))
