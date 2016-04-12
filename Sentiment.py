import nltk

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)
(100, 100)
subj_docs[0]


(['smart', 'and', 'alert', ',', 'thirteen', 'conversations', 'about', 'one',
'thing', 'is', 'a', 'small', 'gem', '.'], 'subj')


train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=1)
len(unigram_feats)

sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
    "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
    "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
    "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
    "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
    "The book was good.",         # positive sentence
    "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
    "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
    "A really bad, horrible book.",       # negative sentence with booster words
    "At least it isn't a horrible book.", # negated negative sentence with contraction
    ":) and :D",     # emoticons handled
    "",              # an empty string is correctly handled
    "Today sux",     #  negative slang handled
    "Today sux!",    #  negative slang with punctuation emphasis handled
    "Today SUX!",    #  negative slang with capitalization emphasis
    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
 ]
paragraph = "It was one of the worst movies I've seen, despite good reviews. \
... Unbelievably bad acting!! Poor direction. VERY poor production. \
... The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"
from nltk import tokenize
lines_list = tokenize.sent_tokenize(paragraph)
sentences.extend(lines_list)
tricky_sentences = [
   "Most automated sentiment analysis tools are shit.",
   "VADER sentiment analysis is the shit.",
   "Sentiment analysis has never been good.",
   "Sentiment analysis with VADER has never been this good.",
   "Warren Beatty has never been so entertaining.",
   "I won't say that the movie is astounding and I wouldn't claim that \
   the movie is too banal either.",
  "I like to hate Michael Bay films, but I couldn't fault this one",
  "It's one thing to watch an Uwe Boll film, but another thing entirely \
   to pay for it",
  "The movie was too good",
  "This movie was actually neither that funny, nor super witty.",
  "This movie doesn't care about cleverness, wit or any other kind of \
   intelligent humor.",
  "Those who find ugly meanings in beautiful things are corrupt without \
   being charming.",
  "There are slow and repetitive parts, BUT it has just enough spice to \
   keep it interesting.",
  "The script is not fantastic, but the acting is decent and the cinematography \
   is EXCELLENT!",
  "Roger Dodger is one of the most compelling variations on this theme.",
  "Roger Dodger is one of the least compelling variations on this theme.",
  "Roger Dodger is at least compelling as a variation on the theme.",
  "they fall in love with the product",
  "but then it breaks",
  "usually around the time the 90 day warranty expires",
  "the twin towers collapsed today",
  "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
   under orders and in the ''least offensive way possible.''",
  "These are my sentences.",
  "Aslan is the best dog in the whole wide world.",
  "Charles made a very foolish decision to not play ultimate frisbee. :D",
  "Oh woe is me.",
  "He sins nothing, unless because he sins nothing.",
  "The one who is happy - I don't understand what he may want so that he may be happier.",
  "With friends all things are in common.",
  "many fear their reputation, few fear conscience.",
  "no book is so bad that it is not useful in some part.",
  "Kindness is brought about by kindness.",
  "it must be warned lest we open our ears to flatterers.",
  "Nothing is more unpredictable than the mob,\
 nothing more obscure than the public opinion,\
 nothing more deceptive than the whole of the political system.",
  "Hunger is the best appetizer of food and drink.",
  "How long will you go on abusing our patience, Charles.",
  "Less alone than when he was alone",
  "For it is better to accept injustice than to make injustice",
  "you who owes all to a name",
  "worldliness must be applied, nto hateful and not excessively sought",
  "It is difficult to retain what you have learned unless you should practice",
  "How was New York City?",
  "Charles and Micheal should play Ultimate Frisbee this Friday.",
  "Charles should study more.",
  "The date of the start of the history of the United States is a subject of debate among historians. Older textbooks start with the arrival of Christopher Columbus in 1492 and emphasize the European background, or they start around 1600 and emphasize the American frontier. In recent decades American schools and universities typically have shifted back in time to include more on the colonial period and much more on the prehistory of the Native peoples.[1][2]",
"Indigenous people lived in what is now the United States for thousands of years before European colonists began to arrive, mostly from England, after 1600. The Spanish had small settlements in Florida and the Southwest, and the French along the Mississippi River and the Gulf Coast. By the 1770s, thirteen British colonies contained two and a half million people along the Atlantic coast east of the Appalachian Mountains. In the 1760s, the British government imposed a series of new taxes while rejecting the American argument that any new taxes had to be approved by the people (see Stamp Act 1765). Tax resistance, especially the Boston Tea Party (1774), led to punitive laws (the Intolerable Acts) by Parliament designed to end self-government in Massachusetts. American Patriots (as they called themselves) adhered to a political ideology called republicanism that emphasized civic duty, virtue, and opposition to corruption, fancy luxuries and aristocracy."

 ]
sentences.extend(tricky_sentences)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()