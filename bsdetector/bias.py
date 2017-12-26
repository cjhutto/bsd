#!/usr/bin/python
# coding: utf-8
"""
Created on June 04, 2015
@author: C.J. Hutto
"""
from __future__ import print_function
import json
import multiprocessing
import os
import sys
import re

from collections import OrderedDict
from decorator import contextmanager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as vader_sentiment
from pattern.text.en import Sentence, parse, modality


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
    text = text.replace("\n", " ")
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
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


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
    ignoreCount = 0
    sentences = split_into_sentences(text)
    for sentence in sentences:
        if lexicon_count(sentence) <= 2:
            ignoreCount = ignoreCount + 1
    sentence_cnt = len(sentences) - ignoreCount
    if sentence_cnt < 1:
        sentence_cnt = 1
    return sentence_cnt


def avg_sentence_length(text):
    lc = lexicon_count(text)
    sc = sentence_count(text)
    ASL = float(lc / sc)
    return round(ASL, 1)


def avg_syllables_per_word(text):
    syllable = syllable_count(text)
    words = lexicon_count(text)
    try:
        ASPW = float(syllable) / float(words)
        return round(ASPW, 1)
    except ZeroDivisionError:
        # print "Error(ASyPW): Number of words are zero, cannot divide"
        return 1


def flesch_kincaid_grade(text):
    ASL = avg_sentence_length(text)
    ASW = avg_syllables_per_word(text)
    FKRA = float(0.39 * ASL) + float(11.8 * ASW) - 15.59
    return round(FKRA, 1)


def count_feature_list_freq(feat_list, words, bigrams, trigrams):
    # Note: probably could be cleaned up a lot... e.g., ... look for whole words
    # import re
    # def find_whole_word(w):
    #    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search
    # lst = sorted(Lexicon.list('ref_coherence_markers'))
    # for w in lst:
    #    excl = [i for i in lst if i != w]
    #    for i in excl:
    #        if find_whole_word(w)(i):
    #            print w, "-->", i
    cnt = 0
    for w in words:
        if w in feat_list:
            cnt += 1
    for b in bigrams:
        if b in feat_list:
            cnt += 1
    for t in trigrams:
        if t in feat_list:
            cnt += 1
    return cnt


def count_phrase_freq(phrase_list, txt_lwr):
    cnt = 0
    for phrase in phrase_list:
        if phrase in txt_lwr:
            cnt += 1
    return cnt


def count_liwc_list_freq(liwc_list, words_list):
    cnt = 0
    for w in words_list:
        if w in liwc_list:
            cnt += 1
        for lw in liwc_list:
            if str(lw).endswith('*') and str(w).startswith(lw):
                cnt += 1
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
        quote_info["mean_quote_length"] = round(float(total_qte_length) / float(len(quotes)), 4)
        nonquotes = split_into_sentences(nonquote)
        if len(nonquotes) > 0:
            quote_info["nonquoted_list"] = nonquotes
            total_nqte_length = 0
            for nqte in nonquotes:
                total_nqte_length += avg_sentence_length(nqte)
            quote_info["mean_nonquote_length"] = round(float(total_nqte_length) / float(len(nonquotes)), 4)
        else:
            quote_info["nonquoted_list"] = None
            quote_info["mean_nonquote_length"] = 0

    return quote_info


ref_lexicons = Lexicons()

##### List of presupposition verbs (comprised of Factive & Implicative verbs):
### Factive verbs derived from:
# Joan B. Hooper. 1975. On assertive predicates. In J. Kimball, editor,
# Syntax and Semantics, volume 4, pages 91–124. Academic Press, New York.
### Implicative verbs derived from
# Lauri Karttunen. 1971. Implicative verbs. Language, 47(2):340–358.
#########################################################################
presup = ref_lexicons.list('ref_presup_verbs')

##### List of coherence markers derived from:
# Knott, Alistair. 1996. A Data-Driven Methodology for Motivating a Set of
# Coherence Relations. Ph.D. dissertation, University of Edinburgh, UK.
#########################################################################
coherence = ref_lexicons.list('ref_coherence_markers')

##### List of assertive derived from:
# Joan B. Hooper. 1975. On assertive predicates. In J. Kimball, editor,
# Syntax and Semantics, volume 4, pages 91–124. Academic Press, New York.
#########################################################################
assertives = ref_lexicons.list('ref_assertive_verbs')

##### List of degree modifiers derived from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
#########################################################################
modifiers = ref_lexicons.list('ref_degree_modifiers')

##### List of hedge words derived from:
# Ken Hyland. 2005. Metadiscourse: Exploring Interaction in Writing.
# Continuum, London and New York.
#########################################################################
hedges = ref_lexicons.list('ref_hedge_words')

##### List of bias words derived from:
# Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan
# Jurafsky. 2013. Linguistic Models for Analyzing and Detecting Biased
# Language. Proceedings of ACL 2013.
#########################################################################
partisan = ref_lexicons.list('ref_partisan_words')

##### List of opinion laden words extracted from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
##### List of strong/weak subjective words extracted from:
# Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
# Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
# Vancouver, Canada.
#########################################################################
value_laden = ref_lexicons.list('ref_value_laden')
vader_sentiment_analysis = vader_sentiment()

##### List of figurative expressions derived from:
# English-language idioms
# https://en.wikipedia.org/wiki/English-language_idioms.
# and
# List of English-language metaphors
# https://en.wikipedia.org/wiki/List_of_English-language_metaphors
# and
# List of political metaphors
# https://en.wikipedia.org/wiki/List_of_political_metaphors
#########################################################################
figurative = ref_lexicons.list('ref_figurative')

##### Lists of LIWC category words
# liwc 3rd person pronoun count (combines S/he and They)
liwc_3pp = ref_lexicons.list('ref_liwc_3pp')
# liwc achievement word count
liwc_achiev = ref_lexicons.list('ref_liwc_achiev')
# liwc causation word count
liwc_causn = ref_lexicons.list('ref_liwc_causn')
# liwc self reference promouns word count
liwc_self = ref_lexicons.list('ref_liwc_self')
# liwc tentative word count
liwc_tent = ref_lexicons.list('ref_liwc_tent')
# liwc work word count
liwc_work = ref_lexicons.list('ref_liwc_work')


def extract_bias_features(text):
    features = OrderedDict()
    text_nohyph = text.replace("-", " ")  # preserve hyphenated words as seperate tokens
    txt_lwr = str(text_nohyph).lower()
    words = ''.join(ch for ch in txt_lwr if ch not in '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~').split()
    unigrams = sorted(list(set(words)))
    bigram_tokens = find_ngrams(words, 2)
    bigrams = [" ".join([w1, w2]) for w1, w2 in sorted(set(bigram_tokens))]
    trigram_tokens = find_ngrams(words, 3)
    trigrams = [" ".join([w1, w2, w3]) for w1, w2, w3 in sorted(set(trigram_tokens))]

    # word count
    features['word_cnt'] = len(words)

    # unique word count
    features['unique_word_cnt'] = len(unigrams)

    # presupposition verb count
    count = count_feature_list_freq(presup, words, bigrams, trigrams)
    features['presup_cnt'] = count
    features['presup_rto'] = round(float(count) / float(len(words)), 4)

    # coherence marker count
    count = count_phrase_freq(coherence, txt_lwr)
    features['cm_cnt'] = count
    features['cm_rto'] = round(float(count) / float(len(words)), 4)

    # assertive verb count
    count = count_feature_list_freq(assertives, words, bigrams, trigrams)
    features['assertive_cnt'] = count
    features['assertive_rto'] = round(float(count) / float(len(words)), 4)

    # degree modifier count
    count = count_feature_list_freq(modifiers, words, bigrams, trigrams)
    features['dm_cnt'] = count
    features['dm_rto'] = round(float(count) / float(len(words)), 4)

    # hedge word count
    count = count_feature_list_freq(hedges, words, bigrams, trigrams)
    features['hedge_cnt'] = count
    features['hedge_rto'] = round(float(count) / float(len(words)), 4)

    # partisan words and phrases count
    count = count_feature_list_freq(partisan, words, bigrams, trigrams)
    features['partisan_cnt'] = count
    features['partisan_rto'] = round(float(count) / float(len(words)), 4)

    # subjective value laden word count
    count = count_feature_list_freq(value_laden, words, bigrams, trigrams)
    features['opinion_cnt'] = count
    features['opinion_rto'] = round(float(count) / float(len(words)), 4)

    # compound sentiment score using VADER sentiment analysis package
    compound_sentiment = vader_sentiment_analysis.polarity_scores(text)['compound']
    features['vader_sentiment'] = compound_sentiment
    features['vader_senti_abs'] = abs(compound_sentiment)

    # modality (certainty) score and mood using  http://www.clips.ua.ac.be/pages/pattern-en#modality
    sentence = parse(text, lemmata=True)
    sentence_obj = Sentence(sentence)
    features['modality'] = round(modality(sentence_obj), 4)

    # Flesch-Kincaid Grade Level (reading difficulty) using textstat
    features['fk_gl'] = flesch_kincaid_grade(text)

    # figurative count
    count = count_phrase_freq(figurative, txt_lwr)
    features['figurative_cnt'] = count
    features['figurative_rto'] = round(float(count) / float(len(words)), 4)

    # liwc 3rd person pronoun count (combines S/he and They)
    count = count_liwc_list_freq(liwc_3pp, words)
    features['liwc_3pp_cnt'] = count
    features['liwc_3pp_rto'] = round(float(count) / float(len(words)), 4)

    # liwc achievement word count
    count = count_liwc_list_freq(liwc_achiev, words)
    features['liwc_achiev_cnt'] = count
    features['liwc_achiev_rto'] = round(float(count) / float(len(words)), 4)

    # liwc causation word count
    count = count_liwc_list_freq(liwc_causn, words)
    features['liwc_causn_cnt'] = count
    features['liwc_causn_rto'] = round(float(count) / float(len(words)), 4)

    # liwc self reference promouns count
    count = count_liwc_list_freq(liwc_self, words)
    features['liwc_self_cnt'] = count
    features['liwc_self_rto'] = round(float(count) / float(len(words)), 4)

    # liwc tentative word count
    count = count_liwc_list_freq(liwc_tent, words)
    features['liwc_tent_cnt'] = count
    features['liwc_tent_rto'] = round(float(count) / float(len(words)), 4)

    # liwc work word count
    count = count_liwc_list_freq(liwc_work, words)
    features['liwc_work_cnt'] = count
    features['liwc_work_rto'] = round(float(count) / float(len(words)), 4)

    # handle quoted material in text
    quote_dict = check_quotes(text)
    features["has_quotes"] = quote_dict["has_quotes"]
    features["mean_quote_length"] = quote_dict["mean_quote_length"]
    features["mean_nonquote_length"] = quote_dict["mean_nonquote_length"]
    return features


def compute_bias(sentence_text):
    """run the trained regression coefficients against the feature dict"""
    features = extract_bias_features(sentence_text)
    bs_score = (-0.5581467 +
                0.3477007 * features['vader_sentiment'] +
                -2.0461103 * features['opinion_rto'] +
                0.5164345 * features['modality'] +
                8.3551389 * features['liwc_3pp_rto'] +
                4.5965115 * features['liwc_tent_rto'] +
                5.737545 * features['liwc_achiev_rto'] +
                -0.953181 * features['partisan_rto'] +
                9.811681 * features['liwc_work_rto'] +
                -16.6359498 * features['presup_rto'] +
                3.059548 * features['hedge_rto'] +
                -3.5770891 * features['assertive_rto'] +
                5.0959142 * features['opinion_rto'])
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
    avg_bias = round(float(avg_bias) / float(len(sentences)), k)
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
        if len(sent) >= 3:
            feature_data = extract_bias_features(sent)
            if not keys_done:
                tsv_output = 'sentence\t' + '\t'.join(feature_data.keys()) + '\n'
                keys_done = True
            str_vals = [str(f) for f in feature_data.values()]
            tsv_output += sent + '\t' + '\t'.join(str_vals) + '\n'
    return tsv_output


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
                html_output += '<tr><th>sentence' + hsep + hsep.join(feature_data.keys()) + '</th></tr>'
                keys_done = True
            str_vals = [str(f) for f in feature_data.values()]
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


if __name__ == '__main__':
    # Demo article file
    # print(compute_avg_statement_bias_mp(get_text_from_article_file("news_articles/brexit_01.txt"), 4))
    '''FPATH = 'input_text'
    for bias, stmt in enumerate_sentences(FPATH):
        msg = 'Bias: {}\t {}'.format(bias, stmt)
        print(msg)

    NEWSPATH = "news_articles/brexit_01.txt"
    print('loading news article: {}'.format(NEWSPATH), file=sys.stderr)
    STATEMENT = get_text_from_article_file(NEWSPATH)
    print(compute_avg_statement_bias(STATEMENT))'''

    demo_output_types = True
    if demo_output_types:
        sentence_list = get_text_from_article_file('input_text').split('\n')
        print_feature_data(sentence_list, output_type='tsv')
