import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import heapq

stop_words = set(stopwords.words("english"))

def extract_keywords(text, num_keywords=10):

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])

    words = vectorizer.get_feature_names_out()
    counts = X.toarray()[0]

    word_freq = dict(zip(words, counts))

    keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:num_keywords]

    return keywords


def summarize_text(text, num_sentences=3):

    sentences = nltk.sent_tokenize(text)

    word_frequencies = {}

    for word in nltk.word_tokenize(text.lower()):
        if word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    sentence_scores = {}

    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return " ".join(summary_sentences)
