#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Aspects Summary and Topic-Entity Recognition (CASTER): extract important keywords, topics, entities from text
Created on January 08, 2018
@author: C.J. Hutto
"""
from __future__ import print_function
import re
import operator
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def tostring(barray):
    """Convert a byte array to string in utf-8 noop if already a string."""
    if isinstance(barray, str):
        return barray
    if isinstance(barray, bytes):
        # return str(barray, 'utf-8', errors='replace')  # Python 3
        return barray.decode(encoding="utf=8", errors="replace")  # Python 2


def encode_ignore(text):
    """
    encode a bytes into a str with utf-8, 'ignore
    see also tostring.
    """
    if not isinstance(text, str):
        text = str(text.encode('utf-8', 'ignore'))
    else:
        pass
    return text


def get_list_from_file(file_name):
    """read the lines from a a file into a list"""
    with open(file_name, "r") as f1:
        lst = f1.read().split('\n')
    return lst


def append_to_file(file_name, line):
    """append a line of text to a file"""
    with open(file_name, 'a') as f1:
        f1.write(line)
        f1.write("\n")


def squeeze_whitespace(text):
    """removes extra white space"""
    return re.sub(r'[\s]+', ' ', text)


def remove_usernames(text):
    """removes @usernames"""
    return re.sub(r'@[^\s]+', '', text)


def remove_hashtags(text):
    """removes # hashtags"""
    return re.sub(r'#[^\s]+', '', text)


def remove_digits(text):
    """removes # digits"""
    return re.sub(r'[0123456789]', '', text)


def remove_nonalphanumerics(text):
    """removes # punctuation and non-alpha-numeric symbol characters"""
    return re.sub(r'[`~!@#$%^&*()-=+[\]{}\\|;:\'",<.>/?_]', ' ', text)


def remove_whitespace(text):
    """removes # redundant/additional white space (incl. those from punctuation replacements)"""
    return re.sub(r'[\s]+', ' ', text)  # includes the set [ \t\n\r\f\v]


def replace_flooded_chars(text):
    """replace 3 or more repetitions of any character patterns w/ 2 occurrences of the shortest pattern"""
    return re.sub(r'(.+?)\1\1+', r'\1\1', text)


STOPWORDS = get_list_from_file('stopWordsSMART.txt')


def remove_stopwords(text, word_char_limit=3, stopwords_list=STOPWORDS):
    """ prune out any undesired words / STOPWORDS """
    return " ".join([word for word in text.split() if len(word) >= word_char_limit and word not in stopwords_list])


def get_pos_tags(text):
    """Used when tokenizing words"""
    text = tostring(text)
    regex_patterns = r"""(?x)      # set flag to allow verbose regexps
          (?:[A-Z]\.)+  # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*            # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    """
    # POS tagging
    # postoks = nltk.pos_tag(text.split())
    toks = nltk.regexp_tokenize(text, regex_patterns)
    assert isinstance(toks, list), "toks is not a list of str, cannot tokenize."
    postoks = nltk.tag.pos_tag(toks)
    # fix a weird pos-tagging error in NLTK
    prior_pos = ''
    for i in range(0, len(postoks)):
        if prior_pos == 'TO' and 'VB' not in postoks[i][1]:
            old = postoks.pop(i)
            postoks.insert(i, (old[0], 'VB'))
        prior_pos = postoks[i][1]
    # print('getPOStags_returns:', postoks)
    return postoks


def extract_entity_names(tree):
    entity_names = []
    if hasattr(tree, 'label') and tree.label:
        if tree.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                entity_names.extend(extract_entity_names(child))
    # entity_names = [str(entity.encode("utf-8", "ignore")).lower() for entity in set(entity_names)]
    entity_names = list(set(entity_names))
    return entity_names


def nltk_extract_entities(text):
    if not isinstance(text, str):
        text = str(text.encode('utf-8', 'replace'))
    text = tostring(text)

    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    entity_names = list(set(entity_names))
    return entity_names


def get_keywords_phrases(text):
    try:
        text = encode_ignore(text)

        lemmatizer = nltk.WordNetLemmatizer()
        # stemmer = nltk.stem.porter.PorterStemmer()

        # Based on... Extract key phrases with NLTK ... https://gist.github.com/alexbowe/879414
        # This gist is part of a blog post (http://alexbowe.com/au-naturale/)
        # in which the paper is cited:
        # S. N. Kim, T. Baldwin, and M.-Y. Kan. Evaluating n-gram based evaluation metrics for automatic
        # keyphrase extraction. Technical report, University of Melbourne, Melbourne 2010.
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        # POS tagging
        postoks = get_pos_tags(text)

        this_tree = chunker.parse(postoks)

        from nltk.corpus import stopwords
        stopwords = stopwords.words('english')

        def leaves(tree):
            """Finds NP (nounphrase) leaf nodes of a chunk tree."""
            # for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                yield subtree.leaves()

        def normalise(word):
            """Normalises words to lowercase and stems and lemmatizes it."""
            word = word.lower()
            # word = stemmer.stem_word(word)
            word = lemmatizer.lemmatize(word)
            return word

        def acceptable_word(word):
            """Checks conditions for acceptable word: length, stopword."""
            accepted = bool(2 <= len(word) <= 40 and word.lower() not in stopwords)
            return accepted

        def get_terms(tree):
            """a generator for the normalized, acceptable, leaf terms"""
            for leaf in leaves(tree):
                this_term = [normalise(w) for w, t in leaf if acceptable_word(w)]
                yield this_term

        terms = get_terms(this_tree)
        phrases = []
        terms_freq_dict = {}
        for termList in terms:
            phrase = " ".join([str(term) for term in termList])
            phrases.append(phrase)
            if phrase not in terms_freq_dict:
                terms_freq_dict[phrase] = 1
            else:
                terms_freq_dict[phrase] += 1
        sorted_tfd = sorted(list(terms_freq_dict.items()), key=operator.itemgetter(1), reverse=True)
        return sorted_tfd
    except Exception as e:
        error_msg = "processing error: ", str(e)
        return {'keyWordsPhrases': error_msg}


def join_with_space(sequence):
    return " ".join(str(s) for s in sequence)


def is_str_set_subset_of_list_str_sets(phrase, phraselist):
    phrase_set = set(encode_ignore(phrase).split())
    for p in phraselist:
        p_set = set(encode_ignore(p).split())
        if phrase_set.issubset(p_set):
            return True
    return False


def extract_summary_terms(nerkwp_count_tpl_list, comparison_text=None, sort_by="both",
                          tf_threshold=2, sim_threshold=0.01, top_n=10):
    term_freq_dict = {}
    for nerkwp_count_tpl in nerkwp_count_tpl_list:
        term, freq = nerkwp_count_tpl
        term_freq_dict[term] = freq
    consolidated_tf_dict = {}
    terms = list(term_freq_dict.keys())
    for key_term in terms:
        if key_term not in consolidated_tf_dict:
            consolidated_tf_dict[key_term] = term_freq_dict[key_term]
        for key_term2 in terms:
            if key_term != key_term2 and len(key_term2) > len(key_term) and key_term in key_term2:
                # print("combining terms: {}, {}".format(key_term, key_term2))
                consolidated_tf_dict[key_term2] = (term_freq_dict[key_term] + term_freq_dict[key_term2])
                if key_term in consolidated_tf_dict:
                    del consolidated_tf_dict[key_term]
    for key_term in list(consolidated_tf_dict.keys()):
        if key_term in consolidated_tf_dict and consolidated_tf_dict[key_term] < tf_threshold:
            del consolidated_tf_dict[key_term]

    if comparison_text:
        text_list = [comparison_text]
        text_list.extend(list(consolidated_tf_dict.keys()))
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(text_list)
        similarity_matrix = (tfidf * tfidf.T).A

        consolidated_tfidf_dict = {}
        for consolidated_term in text_list[1:]:
            sim_score = similarity_matrix[0][text_list.index(consolidated_term)]
            if sim_score >= sim_threshold:
                consolidated_tfidf_dict[consolidated_term] = sim_score
        if sort_by == "tfidf":
            query_term_tuple_sortby_tfidf = sorted(list(consolidated_tfidf_dict.items()),
                                                   key=operator.itemgetter(1), reverse=True)
            return query_term_tuple_sortby_tfidf[:top_n]

        if sort_by == "both":
            product_tf_tfidf_dict = {}
            for term in consolidated_tfidf_dict:
                product_tf_tfidf = consolidated_tf_dict[term] * consolidated_tfidf_dict[term]
                if product_tf_tfidf >= tf_threshold * sim_threshold:
                    product_tf_tfidf_dict[term] = product_tf_tfidf
            query_term_tuple_sortby_both = sorted(list(product_tf_tfidf_dict.items()),
                                                  key=operator.itemgetter(1), reverse=True)
            # print("query_term_tuple_sortby_both: {}".format(query_term_tuple_sortby_both))
            return query_term_tuple_sortby_both[:top_n]

    # final return option is sortedBy == "tf":
    query_term_tuple_sortby_tf = sorted(list(consolidated_tf_dict.items()),
                                        key=operator.itemgetter(1), reverse=True)
    return query_term_tuple_sortby_tf[:top_n]


def combine_ners_kwp(ner_list, kwp_count_tpl_list, original_text):
    original_text = str(encode_ignore(original_text)).lower()
    kwp_list = [x[0] for x in kwp_count_tpl_list]
    terms_freq_dict = {}
    for entity in ner_list:
        entity = encode_ignore(entity)
        entity = str(entity).lower().strip()
        if len(entity) >= 3:
            bonus = 0
            for kwp in kwp_list:
                if entity in kwp:
                    # give a bonus to NEs appearing in the keywords and phrases list
                    bonus += bonus + 2
            if entity not in terms_freq_dict:
                if entity in original_text:
                    # give a bonus to NEs appearing in the original text
                    terms_freq_dict[entity] = bonus + 2
                else:
                    terms_freq_dict[entity] = bonus + 1
            else:
                if entity in original_text:
                    # give a bonus to NEs appearing in the original text
                    terms_freq_dict[entity] += bonus + 2
                else:
                    terms_freq_dict[entity] += bonus + 1
    for kwp_count_tpl in kwp_count_tpl_list:
        phrase = str(encode_ignore(kwp_count_tpl[0])).lower().strip()
        count = kwp_count_tpl[1]
        if len(phrase) >= 3:
            bonus = 0
            for entity in ner_list:
                for kwp in kwp_list:
                    if entity in kwp:
                        # give a bonus to NEs appearing in the keywords and phrases list
                        bonus += 2
            if phrase not in terms_freq_dict:
                if phrase in ner_list:
                    # give a bonus to phrases appearing in the NER list
                    terms_freq_dict[phrase] = bonus + count + 2
                if phrase in original_text:
                    terms_freq_dict[phrase] = bonus + count + 1
                else:
                    terms_freq_dict[phrase] = bonus + count
            else:
                if phrase in ner_list:
                    # give a bonus to phrases appearing in the NER list
                    terms_freq_dict[phrase] += bonus + count + 2
                if phrase in original_text:
                    terms_freq_dict[phrase] += bonus + count + 1
                else:
                    terms_freq_dict[phrase] += bonus + count
    sorted_tf_list = sorted(list(terms_freq_dict.items()), key=operator.itemgetter(1), reverse=True)
    return sorted_tf_list


def caster(text, sort_by="both", term_freq_threshold=2, cos_sim_threshold=0.01, top_n=10):
    text = encode_ignore(text)
    squeezed_text = squeeze_whitespace(text)
    nltk_entities_ext = nltk_extract_entities(squeezed_text)
    kwp_ext = get_keywords_phrases(squeezed_text)
    sorted_term_measure_dict = combine_ners_kwp(nltk_entities_ext, kwp_ext, original_text=text)
    sorted_term_measure_dict = extract_summary_terms(sorted_term_measure_dict, sim_threshold=cos_sim_threshold,
                                                     tf_threshold=term_freq_threshold,
                                                     comparison_text=squeezed_text, sort_by=sort_by)
    return sorted_term_measure_dict[:top_n]


def castr_demo(text):
    """build all the models we will use for comparison to CASTR_original-informed topic models"""
    text = encode_ignore(text)
    print("-- Original text 'document':", encode_ignore(text))
    '''pos_text = getPOStags(basicTextProcess(text, remove_nonalphanumeric=False))
    input("Press Enter to continue...")

    print("\n-- Part Of Speech (POS) tagging:\n", pos_text)
    drs = getDefsRelWrdsSyns(text)
    input("Press Enter to continue...")

    print("\n-- Contextualized semantics; words commonly used in POS-based WSD, e.g., definitions, related concepts:")
    for def_relwrds_syns_tpl in drs:
        word = def_relwrds_syns_tpl[0]
        d_tpl_lst = def_relwrds_syns_tpl[1]  # --> list of (def_cnt_nmbr, pos, definition)
        defTermFrq_tpl_lst = def_relwrds_syns_tpl[2]  # --> list of [ (term, freq_of_term_in_defs) ]
        freq_words_in_defs = [tpl[0] for tpl in defTermFrq_tpl_lst]
        synSetTerm_tpl_lst = def_relwrds_syns_tpl[
            3]  # --> [(term, cosSim_of_termSynSet_to_wordSynSet)] not POS limited
        similar_synSets_to_wordSynSet = [tpl[0] for tpl in synSetTerm_tpl_lst]
        print(word)
        for definition in d_tpl_lst:
            print("\t", definition[2])
        print("\t", freq_words_in_defs)
        print("\t", similar_synSets_to_wordSynSet)'''
    squeezed_text = squeeze_whitespace(text)
    raw_input("Press Enter to continue...")

    print("\n-- Squeezed text (i.e., tabs, newlines, unnecessary white space removed):", encode_ignore(squeezed_text))
    nltk_entities_lst = nltk_extract_entities(squeezed_text)
    raw_input("Press Enter to continue...")

    print("\n-- Named Entity Recognition (NER): names, places, orgs, events:\n",
          nltk_entities_lst)
    kwp_ext = get_keywords_phrases(squeezed_text)
    combined_nerkwp_cnt_tuplst = combine_ners_kwp(nltk_entities_lst, kwp_ext, original_text=text)
    raw_input("Press Enter to continue...")

    print("\n-- Named Entities and other important keywords/phrases (based on POS & WSD), ordered by frequency):\n",
          combined_nerkwp_cnt_tuplst)
    query_terms_tf_freq_tuplst = extract_summary_terms(combined_nerkwp_cnt_tuplst,
                                                       comparison_text=squeezed_text, sort_by="tf")
    query_terms_tfidf_cossim_tuplst = extract_summary_terms(combined_nerkwp_cnt_tuplst,
                                                            comparison_text=squeezed_text, sort_by="tfidf")
    query_terms_both_tuplst = extract_summary_terms(combined_nerkwp_cnt_tuplst,
                                                    comparison_text=squeezed_text, sort_by="both")
    raw_input("Press Enter to continue...")

    print("\n-- Preparing SUMMARY TERMS based on word-order sensitive n-gram combinations of \
    Named Entities & Important Keywords/Phrases...")
    print("   Search Terms are ordered by textual similarity of {search term} to {document text}.")
    print("   (Only the terms which exceed our user-definable Freq or Cos-Sim thresholds)")
    print("   -- Sorted by Term Frequency -- ")
    for query, freq_score in query_terms_tf_freq_tuplst:
        print(query, round(freq_score, 2))
    print("   -- Sorted by TF*IDF -- ")
    for query, cos_sim_score in query_terms_tfidf_cossim_tuplst:
        print(query, round(cos_sim_score, 2))
    print("   -- Sorted by Term Frequency X tfidf -- ")
    for query, comb_score in query_terms_both_tuplst:
        print(query, round(comb_score, 2))


if __name__ == '__main__':
    from datetime import datetime

    # --- START TIMING -------
    start = datetime.now()
    print(" Started: ", start.strftime('%Y-%m-%d %I:%M:%S %p %Z'), "\n")

    # --- CASTR_original DEMO -------
    tester = """ The discovery of what looks like the aftermath of a brutal clash between two groups of prehistoric 
    hunter-gatherers on the shore of an African lake is certain to stir up a debate about human nature that goes 
    all the way back to Adam and Eve. \n\n  The biblical creation story posits that our forebears were inherently pure 
    and peaceful and only fell into nasty struggles for dominance with the knowledge of the forbidden fruit. \n A 
    corollary advanced by one school of archaeologists and anthropologists holds that our Stone Age ancestors were not 
    inherently violent, and, apart from the odd murder, did not wage organized war until they started to coalesce into 
    societies. \n Not so, proclaim proponents of a rival theory that war has deep biological roots, and we've been 
    waging it forever. \r\n That's what we are, argued the philosopher Thomas Hobbes; not so, declared Jean-Jacques 
    Rousseau. \n\tEven President Obama jumped into the debate when, in his Nobel acceptance speech in 2009, he asserted 
    that "War, in one form or another, appeared with the first man." What scientists found at a place called Nataruk on 
    what was once the shore of a lagoon on Lake Turkana \t\t in Kenya were skeletons showing unmistakable evidence of
    violent deaths - crushed skulls, imbedded arrow or spear                                  points and the like. \n 
    \n It was obviously a terribly violent encounter. 
    But was it war?
    The skeletons, alas, do not provide a conclusive answer, the scientists acknowledged.
    War, broadly defined as large-scale violent clashes, was fairly common between settled societies, and it is not 
    clear whether the dwellers on the fertile land around Lake Turkana at the time of the Nataruk clash were already 
    forming such societies, which would make a violent encounter less surprising, or whether the foraging groups banded 
    together to fight. """
    test_strings = [tester]
    for test_text in test_strings:
        # castr_demo(test_text)
        print("Contextual Aspect Summary and Topical-Entity Recognition:")
        print("(only those above the definable TF and/or cos-sim thresholds, and limited to the top-N results")
        summaryterms = caster(test_text, sort_by="both", term_freq_threshold=2, cos_sim_threshold=0.01, top_n=10)
        for keywordphrase, score in summaryterms:
            print(keywordphrase, round(score, 2))

    # --- END TIMING -------
    elapsed = datetime.now() - start
    print("\n Done! ", datetime.now().strftime('%Y-%m-%d %I:%M:%S %p %Z'))
    print("Elapsed", elapsed)
