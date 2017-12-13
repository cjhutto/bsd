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

import nltk
# from vaderSentiment.vaderSentiment import sentiment as vader_sentiment
from decorator import contextmanager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as vader_sentiment
from pattern.text.en import parse, Sentence, parse, modality, mood
from pattern.text.en import sentiment as pattern_sentiment
from textstat.textstat import textstat


class Lexicon(object):
    """Lexicon is a class with static members for managing the existing lists of words.

    Use Lexicon.list(key) in order to access the list with name key.
    """
    pth = os.path.join(os.path.dirname(__file__), 'lexicon.json')
    print(pth)
    wordlists = {}
    with open(pth, 'r') as filp:
        wordlists = json.loads(filp.read())
    print(list(wordlists.keys()))

    @classmethod
    def list(cls, name):
        """list(name) get the word list associated with key name"""
        return cls.wordlists[os.path.basename(name)]


def get_text_from_article_file(article_file_path):
    with open(article_file_path, "r") as filep:
        l = filep.read().lower()
    return l


def append_to_file(file_name, line):
    "append a line of text to a file"
    with open(file_name, 'a') as filep:
        filep.write(line)
        filep.write("\n")


def count_feature_list_freq(feat_list, words, bigrams, trigrams):
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


def count_liwc_list_freq(liwc_list, words_list):
    cnt = 0
    for w in words_list:
        if w in liwc_list:
            cnt += 1
        for lw in liwc_list:
            if str(lw).endswith('*') and str(w).startswith(lw):
                cnt += 1
    return cnt


##### List of assertive verbs and factive verbs extracted from:
# Joan B. Hooper. 1975. On assertive predicates. In J. Kimball, editor,
# Syntax and Semantics, volume 4, pages 91–124. Academic Press, New York.
#########################################################################
assertives = Lexicon.list('../ref_lexicons/ref_assertive_verbs')
factives = Lexicon.list('../ref_lexicons/ref_factive_verbs')

##### List of hedges extracted from:
# Ken Hyland. 2005. Metadiscourse: Exploring Interaction in Writing.
# Continuum, London and New York.
#########################################################################
hedges = Lexicon.list('../ref_lexicons/ref_hedge_words')

##### List of implicative verbs extracted from:
# Lauri Karttunen. 1971. Implicative verbs. Language, 47(2):340–358.
#########################################################################
implicatives = Lexicon.list('../ref_lexicons/ref_implicative_verbs')

##### List of strong/weak subjective words extracted from:
# Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
# Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
# Vancouver, Canada.
#########################################################################
subj_strong = Lexicon.list('../ref_lexicons/ref_subj_strong')
subj_weak = Lexicon.list('../ref_lexicons/ref_subj_weak')

##### List of bias words extracted from:
# Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan
# Jurafsky. 2013. Linguistic Models for Analyzing and Detecting Biased
# Language. Proceedings of ACL 2013.
#########################################################################
biased = Lexicon.list('../ref_lexicons/ref_bias_words')

##### List of coherence markers extracted from:
# Knott, Alistair. 1996. A Data-Driven Methodology for Motivating a Set of
# Coherence Relations. Ph.D. dissertation, University of Edinburgh, UK.
# Note: probably could be cleaned up a lot... e.g., ...
# import re
# def find_whole_word(w):
#    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search
# lst = sorted(Lexicon.list('ref_coherence_markers'))
# for w in lst:
#    excl = [i for i in lst if i != w]
#    for i in excl:
#        if find_whole_word(w)(i):
#            print w, "-->", i
#########################################################################
coherence = Lexicon.list('../ref_lexicons/ref_coherence_markers')

##### List of degree modifiers and opinion words extracted from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
#########################################################################
modifiers = Lexicon.list('../ref_lexicons/ref_degree_modifiers')
opinionLaden = Lexicon.list('../ref_lexicons/ref_vader_words')
vader_sentiment_analysis = vader_sentiment()

##### Lists of LIWC category words
# liwc 3rd person pronoun count (combines S/he and They)
liwc_3pp = ["he", "hed", "he'd", "her", "hers", "herself", "hes", "he's", "him", "himself", "his", "oneself",
            "she", "she'd", "she'll", "shes", "she's", "their*", "them", "themselves", "they", "theyd",
            "they'd", "theyll", "they'll", "theyve", "they've"]
# liwc auxiliary verb count
liwc_aux = ["aint", "ain't", "am", "are", "arent", "aren't", "be", "became", "become", "becomes",
            "becoming", "been", "being", "can", "cannot", "cant", "can't", "could", "couldnt",
            "couldn't", "couldve", "could've", "did", "didnt", "didn't", "do", "does", "doesnt",
            "doesn't", "doing", "done", "dont", "don't", "had", "hadnt", "hadn't", "has", "hasnt",
            "hasn't", "have", "havent", "haven't", "having", "hed", "he'd", "heres", "here's",
            "hes", "he's", "id", "i'd", "i'll", "im", "i'm", "is", "isnt", "isn't", "itd", "it'd",
            "itll", "it'll", "it's", "ive", "i've", "let", "may", "might", "mightve", "might've",
            "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "ought", "oughta",
            "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "shall", "shant", "shan't",
            "she'd", "she'll", "shes", "she's", "should", "shouldnt", "should'nt", "shouldn't",
            "shouldve", "should've", "thatd", "that'd", "thatll", "that'll", "thats", "that's",
            "theres", "there's", "theyd", "they'd", "theyll", "they'll", "theyre", "they're",
            "theyve", "they've", "was", "wasnt", "wasn't", "we'd", "we'll", "were", "weren't",
            "weve", "we've", "whats", "what's", "wheres", "where's", "whod", "who'd", "wholl",
            "who'll", "will", "wont", "won't", "would", "wouldnt", "wouldn't", "wouldve", "would've",
            "youd", "you'd", "youll", "you'll", "youre", "you're", "youve", "you've"]
# liwc adverb count
liwc_adv = ["about", "absolutely", "actually", "again", "also", "anyway*", "anywhere", "apparently",
            "around", "back", "basically", "beyond", "clearly", "completely", "constantly", "definitely",
            "especially", "even", "eventually", "ever", "frequently", "generally", "here", "heres", "here's",
            "hopefully", "how", "however", "immediately", "instead", "just", "lately", "maybe", "mostly",
            "nearly", "now", "often", "only", "perhaps", "primarily", "probably", "push*", "quick*", "rarely",
            "rather", "really", "seriously", "simply", "so", "somehow", "soon", "sooo*", "still", "such",
            "there", "theres", "there's", "tho", "though", "too", "totally", "truly", "usually", "very", "well",
            "when", "whenever", "where", "yet"]
# liwc preposition count
liwc_prep = ["about", "above", "across", "after", "against", "ahead", "along", "among*", "around", "as", "at",
             "atop", "away", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond",
             "by", "despite", "down", "during", "except", "for", "from", "in", "inside", "insides", "into", "near",
             "of", "off", "on", "onto", "out", "outside", "over", "plus", "since", "than", "through*", "thru", "til",
             "till", "to", "toward*", "under", "underneath", "unless", "until", "unto", "up", "upon", "wanna", "with",
             "within", "without"]
# liwc conjunction count
liwc_conj = ["also", "although", "and", "as", "altho", "because", "but", "cuz", "how", "however", "if", "nor",
             "or", "otherwise", "plus", "so", "then", "tho", "though", "til", "till", "unless", "until", "when",
             "whenever", "whereas", "whether", "while"]
# liwc discrepency word count
liwc_discr = ["besides", "could", "couldnt", "couldn't", "couldve", "could've", "desir*", "expect*", "hope", "hoped",
              "hopeful", "hopefully",
              "hopefulness", "hopes", "hoping", "ideal*", "if", "impossib*", "inadequa*", "lack*", "liabilit*",
              "mistak*", "must", "mustnt",
              "must'nt", "mustn't", "mustve", "must've", "need", "needed", "needing", "neednt", "need'nt", "needn't",
              "needs", "normal", "ought",
              "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "outstanding", "prefer*", "problem*",
              "rather", "regardless",
              "regret*", "should", "shouldnt", "should'nt", "shouldn't", "shoulds", "shouldve", "should've",
              "undesire*", "undo", "unneccess*",
              "unneed*", "unwant*", "wanna", "want", "wanted", "wanting", "wants", "wish", "wished", "wishes",
              "wishing", "would", "wouldnt",
              "wouldn't", "wouldve", "would've", "yearn*"]
# liwc tentative word count
liwc_tent = ["allot", "almost", "alot", "ambigu*", "any", "anybod*", "anyhow", "anyone*", "anything", "anytime",
             "anywhere",
             "apparently", "appear", "appeared", "appearing", "appears", "approximat*", "arbitrar*", "assum*", "barely",
             "bet",
             "bets", "betting", "blur*", "borderline*", "chance", "confus*", "contingen*", "depend", "depended",
             "depending",
             "depends", "disorient*", "doubt*", "dubious*", "dunno", "fairly", "fuzz*", "generally", "guess", "guessed",
             "guesses",
             "guessing", "halfass*", "hardly", "hazie*", "hazy", "hesita*", "hope", "hoped", "hopeful", "hopefully",
             "hopefulness",
             "hopes", "hoping", "hypothes*", "hypothetic*", "if", "incomplet*", "indecis*", "indefinit*", "indetermin*",
             "indirect*",
             "kind(of)", "kinda", "kindof", "likel*", "lot", "lotof", "lots", "lotsa", "lotta", "luck", "lucked",
             "lucki*", "luckless*",
             "lucks", "lucky", "mainly", "marginal*", "may", "maybe", "might", "mightve", "might've", "most", "mostly",
             "myster*", "nearly",
             "obscur*", "occasional*", "often", "opinion", "option", "or", "overall", "partly", "perhaps", "possib*",
             "practically", "pretty",
             "probable", "probablistic*", "probably", "puzzl*", "question*", "quite", "random*", "seem", "seemed",
             "seeming*", "seems", "shaki*",
             "shaky", "some", "somebod*", "somehow", "someone*", "something*", "sometime", "sometimes", "somewhat",
             "sort", "sorta", "sortof",
             "sorts", "sortsa", "spose", "suppose", "supposed", "supposes", "supposing", "supposition*", "tempora*",
             "tentativ*", "theor*",
             "typically", "uncertain*", "unclear*", "undecided*", "undetermin*", "unknow*", "unlikel*", "unluck*",
             "unresolv*", "unsettl*",
             "unsure*", "usually", "vague*", "variab*", "varies", "vary", "wonder", "wondered", "wondering", "wonders"]
# liwc certainty word count
liwc_cert = ["absolute", "absolutely", "accura*", "all", "altogether", "always", "apparent", "assur*", "blatant*",
             "certain*", "clear", "clearly",
             "commit", "commitment*", "commits", "committ*", "complete", "completed", "completely", "completes",
             "confidence", "confident",
             "confidently", "correct*", "defined", "definite", "definitely", "definitive*", "directly", "distinct*",
             "entire*", "essential",
             "ever", "every", "everybod*", "everything*", "evident*", "exact*", "explicit*", "extremely", "fact",
             "facts", "factual*", "forever",
             "frankly", "fundamental", "fundamentalis*", "fundamentally", "fundamentals", "guarant*", "implicit*",
             "indeed", "inevitab*",
             "infallib*", "invariab*", "irrefu*", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've",
             "necessar*", "never", "obvious*",
             "perfect*", "positiv*", "precis*", "proof", "prove*", "pure*", "sure*", "total", "totally", "true",
             "truest", "truly", "truth*",
             "unambigu*", "undeniab*", "undoubt*", "unquestion*", "wholly"]
# liwc causation word count
liwc_causn = ["activat*", "affect", "affected", "affecting", "affects", "aggravat*", "allow*", "attribut*", "based",
              "bases", "basis",
              "because", "boss*", "caus*", "change", "changed", "changes", "changing", "compel*", "compliance",
              "complie*", "comply*",
              "conclud*", "consequen*", "control*", "cos", "coz", "create*", "creati*", "cuz", "deduc*", "depend",
              "depended", "depending",
              "depends", "effect*", "elicit*", "experiment", "force*", "foundation*", "founded", "founder*",
              "generate*", "generating",
              "generator*", "hence", "how", "hows", "how's", "ignit*", "implica*", "implie*", "imply*", "inact*",
              "independ*", "induc*",
              "infer", "inferr*", "infers", "influenc*", "intend*", "intent*", "justif*", "launch*", "lead*", "led",
              "made", "make", "maker*",
              "makes", "making", "manipul*", "misle*", "motiv*", "obedien*", "obey*", "origin", "originat*", "origins",
              "outcome*", "permit*",
              "pick ", "produc*", "provoc*", "provok*", "purpose*", "rational*", "react*", "reason*", "response",
              "result*", "root*", "since",
              "solution*", "solve", "solved", "solves", "solving", "source*", "stimul*", "therefor*", "thus",
              "trigger*", "use", "used", "uses",
              "using", "why"]
# liwc work word count
liwc_work = ["absent*", "academ*", "accomplish*", "achiev*", "administrat*", "advertising", "advis*", "agent", "agents",
             "ambiti*", "applicant*",
             "applicat*", "apprentic*", "assign*", "assistan*", "associat*", "auditorium*", "award*", "beaten",
             "benefits", "biolog*", "biz",
             "blackboard*", "bldg*", "book*", "boss*", "broker*", "bureau*", "burnout*", "business*", "busy",
             "cafeteria*", "calculus", "campus*",
             "career*", "ceo*", "certif*", "chairm*", "chalk", "challeng*", "champ*", "class", "classes", "classmate*",
             "classroom*", "collab*",
             "colleague*", "colleg*", "com", "commerc*", "commute*", "commuting", "companies", "company", "comput*",
             "conferenc*", "conglom*",
             "consult*", "consumer*", "contracts", "corp", "corporat*", "corps", "counc*", "couns*", "course*",
             "coworker*", "credential*",
             "credit*", "cubicle*", "curricul*", "customer*", "cv*", "deadline*", "dean*", "delegat*", "demote*",
             "department*", "dept", "desk*",
             "diplom*", "director*", "dissertat*", "dividend*", "doctor*", "dorm*", "dotcom", "downsiz*", "dropout*",
             "duti*", "duty", "earn*",
             "econ*", "edit*", "educat*", "elementary", "employ*", "esl", "exam", "exams", "excel*", "executive*",
             "expel*", "expulsion*",
             "factories", "factory", "facult*", "fail*", "fax*", "feedback", "finaliz*", "finals", "financ*", "fired",
             "firing", "franchis*",
             "frat", "fratern*", "freshm*", "gmat", "goal*", "gov", "govern*", "gpa", "grad", "grade*", "grading",
             "graduat*", "gre", "hardwork*",
             "headhunter*", "highschool*", "hire*", "hiring", "homework*", "inc", "income*", "incorp*", "industr*",
             "instruct*", "interview*",
             "inventory", "jd", "job*", "junior*", "keyboard*", "kinderg*", "labor*", "labour*", "laidoff", "laptop*",
             "lawyer*", "layoff*",
             "lead*", "learn*", "lectur*", "legal*", "librar*", "lsat", "ltd", "mailroom*", "majoring", "majors",
             "manag*", "manufact*", "market*",
             "masters", "math*", "mcat", "mda", "meeting*", "memo", "memos", "menial", "mentor*", "merger*", "mfg",
             "mfr", "mgmt", "mgr", "midterm*",
             "motiv*", "negotiat*", "ngo", "nonprofit*", "occupa*", "office*", "org", "organiz*", "outlin*",
             "outsourc*", "overpaid", "overtime",
             "overworked", "paper*", "pay*", "pc*", "pen", "pencil*", "pens", "pension*", "phd*", "photocop*", "pledg*",
             "police", "policy",
             "political", "politics", "practice", "prereq*", "presentation*", "presiden*", "procrastin*", "produc*",
             "prof", "profession*",
             "professor*", "profit*", "profs", "program*", "project", "projector*", "projects", "prom", "promot*",
             "psych", "psychol*", "publish",
             "qualifi*", "quiz*", "read", "recruit*", "register*", "registra*", "report*", "requir*", "research*",
             "resource", "resources",
             "resourcing", "responsib*", "resume", "retire*", "retiring", "review*", "rhetor*", "salar*", "scholar",
             "scholaring", "scholarly",
             "scholars", "scholarship*", "scholastic*", "school*", "scien*", "secretar*", "sector*", "semester*",
             "senior*", "servic*",
             "session*", "sickday*", "sickleave*", "sophom*", "sororit*", "staff*", "stapl*", "stipend*", "stock",
             "stocked", "stocker",
             "stocks", "student*", "studied", "studies", "studious", "study*", "succeed*", "success*", "supervis*",
             "syllabus*", "taught", "tax",
             "taxa*", "taxed", "taxes", "taxing", "teach*", "team*", "tenure*", "test", "tested", "testing", "tests",
             "textbook*", "theses",
             "thesis", "toefl", "trade*", "trading", "transcript*", "transfer*", "tutor*", "type*", "typing",
             "undergrad*", "underpaid",
             "unemploy*", "universit*", "unproduc*", "upperclass*", "varsit*", "vita", "vitas", "vocation*", "vp*",
             "wage", "wages", "warehous*",
             "welfare", "work ", "workabl*", "worked", "worker*", "working*", "works", "xerox*"]
# liwc achievement word count
liwc_achiev = ["abilit*", "able*", "accomplish*", "ace", "achiev*", "acquir*", "acquisition*", "adequa*", "advanc*",
               "advantag*", "ahead",
               "ambiti*", "approv*", "attain*", "attempt*", "authorit*", "award*", "beat", "beaten", "best", "better",
               "bonus*", "burnout*",
               "capab*", "celebrat*", "challeng*", "champ*", "climb*", "closure", "compet*", "conclud*", "conclus*",
               "confidence", "confident",
               "confidently", "conquer*", "conscientious*", "control*", "create*", "creati*", "crown*", "defeat*",
               "determina*", "determined",
               "diligen*", "domina*", "domote*", "driven", "dropout*", "earn*", "effect*", "efficien*", "effort*",
               "elit*", "enabl*", "endeav*",
               "excel*", "fail*", "finaliz*", "first", "firsts", "founded", "founder*", "founding", "fulfill*", "gain*",
               "goal*", "hero*", "honor*",
               "honour*", "ideal*", "importan*", "improve*", "improving", "inadequa*", "incapab*", "incentive*",
               "incompeten*", "ineffect*",
               "initiat*", "irresponsible*", "king*", "lazie*", "lazy", "lead*", "lesson*", "limit*", "lose", "loser*",
               "loses", "losing", "loss*",
               "lost", "master", "mastered", "masterful*", "mastering", "mastermind*", "masters", "mastery", "medal*",
               "mediocr*", "motiv*",
               "obtain*", "opportun*", "organiz*", "originat*", "outcome*", "overcome", "overconfiden*", "overtak*",
               "perfect*", "perform*",
               "persever*", "persist*", "plan", "planned", "planner*", "planning", "plans", "potential*", "power*",
               "practice", "prais*",
               "presiden*", "pride", "prize*", "produc*", "proficien*", "progress", "promot*", "proud*", "purpose*",
               "queen", "queenly", "quit",
               "quitt*", "rank", "ranked", "ranking", "ranks", "recover*", "requir*", "resolv*", "resourceful*",
               "responsib*", "reward*", "skill",
               "skilled", "skills", "solution*", "solve", "solved", "solves", "solving", "strateg*", "strength*",
               "striv*", "strong*", "succeed*",
               "success*", "super", "superb*", "surviv*", "team*", "top", "tried", "tries", "triumph*", "try", "trying",
               "unable", "unbeat*",
               "unproduc*", "unsuccessful*", "victor*", "win", "winn*", "wins", "won", "work ", "workabl*", "worked",
               "worker*", "working*", "works"]


def extract_bias_features(text):
    features = {}
    text = unicode(text, errors='ignore') if not isinstance(text, unicode) else text
    txt_lwr = str(text).lower()
    words = nltk.word_tokenize(txt_lwr)
    words = [w for w in words if len(w) > 0 and w not in '.?!,;:\'s"$']
    unigrams = sorted(list(set(words)))
    bigram_tokens = nltk.bigrams(words)
    bigrams = [" ".join([w1, w2]) for w1, w2 in sorted(set(bigram_tokens))]
    trigram_tokens = nltk.trigrams(words)
    trigrams = [" ".join([w1, w2, w3]) for w1, w2, w3 in sorted(set(trigram_tokens))]

    # word count
    features['word_cnt'] = len(words)

    # unique word count
    features['unique_word_cnt'] = len(unigrams)

    # coherence marker count
    count = count_feature_list_freq(coherence, words, bigrams, trigrams)
    features['cm_cnt'] = count
    features['cm_rto'] = round(float(count) / float(len(words)), 4)

    # degree modifier count
    count = count_feature_list_freq(modifiers, words, bigrams, trigrams)
    features['dm_cnt'] = count
    features['dm_rto'] = round(float(count) / float(len(words)), 4)

    # hedge word count
    count = count_feature_list_freq(hedges, words, bigrams, trigrams)
    features['hedge_cnt'] = count
    features['hedge_rto'] = round(float(count) / float(len(words)), 4)

    # factive verb count
    count = count_feature_list_freq(factives, words, bigrams, trigrams)
    features['factive_cnt'] = count
    features['factive_rto'] = round(float(count) / float(len(words)), 4)

    # assertive verb count
    count = count_feature_list_freq(assertives, words, bigrams, trigrams)
    features['assertive_cnt'] = count
    features['assertive_rto'] = round(float(count) / float(len(words)), 4)

    # implicative verb count
    count = count_feature_list_freq(implicatives, words, bigrams, trigrams)
    features['implicative_cnt'] = count
    features['implicative_rto'] = round(float(count) / float(len(words)), 4)

    # bias words and phrases count
    count = count_feature_list_freq(biased, words, bigrams, trigrams)
    features['bias_cnt'] = count
    features['bias_rto'] = round(float(count) / float(len(words)), 4)

    # opinion word count
    count = count_feature_list_freq(opinionLaden, words, bigrams, trigrams)
    features['opinion_cnt'] = count
    features['opinion_rto'] = round(float(count) / float(len(words)), 4)

    # weak subjective word count
    count = count_feature_list_freq(subj_weak, words, bigrams, trigrams)
    features['subj_weak_cnt'] = count
    features['subj_weak_rto'] = round(float(count) / float(len(words)), 4)

    # strong subjective word count
    count = count_feature_list_freq(subj_strong, words, bigrams, trigrams)
    features['subj_strong_cnt'] = count
    features['subj_strong_rto'] = round(float(count) / float(len(words)), 4)

    # composite sentiment score using VADER sentiment analysis package
    compound_sentiment = vader_sentiment_analysis.polarity_scores(text)['compound']
    features['vader_sentiment'] = compound_sentiment

    # subjectivity score using Pattern.en
    pattern_subjectivity = pattern_sentiment(text)[1]
    features['subjectivity'] = round(pattern_subjectivity, 4)

    # modality (certainty) score and mood using  http://www.clips.ua.ac.be/pages/pattern-en#modality
    sentence = parse(text, lemmata=True)
    sentence_obj = Sentence(sentence)
    features['modality'] = round(modality(sentence_obj), 4)
    features['mood'] = mood(sentence_obj)

    # Flesch-Kincaid Grade Level (reading difficulty) using textstat
    features['fk_gl'] = textstat.flesch_kincaid_grade(text)

    # liwc 3rd person pronoun count (combines S/he and They)
    count = count_liwc_list_freq(liwc_3pp, words)
    features['liwc_3pp_cnt'] = count
    features['liwc_3pp_rto'] = round(float(count) / float(len(words)), 4)

    # liwc auxiliary verb count
    count = count_liwc_list_freq(liwc_aux, words)
    features['liwc_aux_cnt'] = count
    features['liwc_aux_rto'] = round(float(count) / float(len(words)), 4)

    # liwc adverb count
    count = count_liwc_list_freq(liwc_adv, words)
    features['liwc_adv_cnt'] = count
    features['liwc_adv_rto'] = round(float(count) / float(len(words)), 4)

    # liwc preposition count
    count = count_liwc_list_freq(liwc_prep, words)
    features['liwc_prep_cnt'] = count
    features['liwc_prep_rto'] = round(float(count) / float(len(words)), 4)

    # liwc conjunction count
    count = count_liwc_list_freq(liwc_conj, words)
    features['liwc_conj_cnt'] = count
    features['liwc_conj_rto'] = round(float(count) / float(len(words)), 4)

    # liwc discrepency word count
    count = count_liwc_list_freq(liwc_discr, words)
    features['liwc_discr_cnt'] = count
    features['liwc_discr_rto'] = round(float(count) / float(len(words)), 4)

    # liwc tentative word count
    count = count_liwc_list_freq(liwc_tent, words)
    features['liwc_tent_cnt'] = count
    features['liwc_tent_rto'] = round(float(count) / float(len(words)), 4)

    # liwc certainty word count
    count = count_liwc_list_freq(liwc_cert, words)
    features['liwc_cert_cnt'] = count
    features['liwc_cert_rto'] = round(float(count) / float(len(words)), 4)

    # liwc causation word count
    count = count_liwc_list_freq(liwc_causn, words)
    features['liwc_causn_cnt'] = count
    features['liwc_causn_rto'] = round(float(count) / float(len(words)), 4)

    # liwc work word count
    count = count_liwc_list_freq(liwc_work, words)
    features['liwc_work_cnt'] = count
    features['liwc_work_rto'] = round(float(count) / float(len(words)), 4)

    # liwc achievement word count
    count = count_liwc_list_freq(liwc_achiev, words)
    features['liwc_achiev_cnt'] = count
    features['liwc_achiev_rto'] = round(float(count) / float(len(words)), 4)

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
                5.6573254 * features['liwc_discr_rto'] +
                -0.953181 * features['bias_rto'] +
                9.811681 * features['liwc_work_rto'] +
                -16.6359498 * features['factive_rto'] +
                3.059548 * features['hedge_rto'] +
                -3.5770891 * features['assertive_rto'] +
                5.0959142 * features['subj_strong_rto'] +
                4.872367 * features['subj_weak_rto'])
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


def compute_statement_bias_mp(statement_text, n_jobs=1):
    """compute_statement_bias_mp a version of compute_statement_bias
    with the multiprocessing pool manager."""
    sentences = nltk.sent_tokenize(statement_text.decode("ascii", "ignore"))
    max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    avg_bias = 0

    with poolcontext(processes=n_jobs) as pool:
        bs_scores = pool.map(compute_bias, sentences)
        avg_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(avg_bias, sentences)

    return avg_bias


def compute_statement_bias(statement_text):
    """compute the bias of a statement from the test.

    Warning: assumes that the statement is in ascii.

    returns the average bias over the entire text broken down by sentence.
    """
    sentences = nltk.sent_tokenize(statement_text.decode("ascii", "ignore"))
    max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    avg_bias = 0
    bs_scores = []
    for sent in sentences:
        bs_scores.append(compute_bias(sent))

    avg_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(avg_bias, sentences)

    return avg_bias


def make_tsv_output(list_of_sentences):
    """print out a table of output as a tab separated file."""
    # make tab seperated values
    keys_done = False
    logmessage = "-- Example TSV: paste the following into Excel then do Data-->Text To Columns-->Delimited-->Tab-->Finish"
    print(logmessage, file=sys.stderr)
    tsv_output = ''
    for sent in list_of_sentences:
        if len(sent) > 3:
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
    logmessage = "-- Example HTML: paste the following in a text editor and save it as 'bias.html', then open with a browser"
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


def print_feature_data(list_of_sentences, output_type='tsv', file=sys.stdout):
    """print the data in either html or tsv format"""
    output = ' -- no output available'
    if output_type == 'html':
        output = make_html_output(list_of_sentences)
    elif output_type == 'tsv':
        output = make_tsv_output(list_of_sentences)
    print(output, file=file)


def enumerate_sentences(fpath='input_text'):
    """print the bias of each sentence in a document."""
    sentences_list = get_text_from_article_file(fpath).split('\n')
    for statement in sentences_list:
        if len(statement) > 3:
            biasq = compute_bias(statement)
            yield(biasq, statement)
        else:
            print('statement is too short: {}'.format(statement))

if __name__ == '__main__':
    # Demo article file
    #print(compute_statement_bias_mp(get_text_from_article_file("news_articles/brexit_01.txt"), 4))
    FPATH = 'input_text'
    for bias, stmt in enumerate_sentences(FPATH):
        msg = 'Bias: {}\t {}'.format(bias, stmt)
        print(msg)

    NEWSPATH = "news_articles/brexit_01.txt"
    print('loading news article: {}'.format(NEWSPATH), file=sys.stderr)
    STATEMENT = get_text_from_article_file(NEWSPATH)
    print(compute_statement_bias(STATEMENT))

    #demo_output_types = True
    #if demo_output_types:
    #    sentence_list = Lexicon.list('input_text')
    #    print_feature_data(sentence_list, output_type='html')
