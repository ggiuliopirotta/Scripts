from    gensim.corpora import Dictionary
from    gensim.corpora.mmcorpus import MmCorpus
from    gensim.models import Phrases
from    gensim.parsing.preprocessing import remove_stopwords
import  logging
import  matplotlib.pyplot as plt
from    nltk.stem.wordnet import WordNetLemmatizer
from    nltk.tokenize import RegexpTokenizer
import  numpy as np
import  pandas as pd
import  re
import  smart_open
import  tarfile



def create_logger(logger_name):
    '''
    Create a logger with DEBUG level to append logs to file logs.log
    '''

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler for the logger
    file_handler = logging.FileHandler("results/logs.log")
    file_handler.setLevel(logging.DEBUG)

    # create formatter
    class CustomFormatter(logging.Formatter):
        def format(self, record):

            record.model_name   = getattr(record, "model_name", "N/A")
            record.params       = getattr(record, "params", "N/A")
            record.train_time   = getattr(record, "train_time", "N/A")
            record.perplexity   = getattr(record, "perplexity", "N/A")
            return super().format(record)

    formatter = CustomFormatter("Model: %(model_name)s - Params: %(params)s - Train Time: %(train_time)s - Perplexity: %(perplexity)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def download_corpus(url):
    '''
    Extract documents from tarball located at a url
    '''

    with smart_open.open(url, 'rb') as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r"nipstxt/nips\d+/\d+\.txt", member.name):

                    # extract and decode the file
                    member_bytes = tar.extractfile(member).read()
                    member_text = member_bytes.decode(encoding="utf-8", errors="replace")
                    yield member_text


def tokenize_doc(tokenizer, doc):
    '''
    Tokenize a document into words with a tokenizer
    '''

    # convert to lowercase and split into single words
    tokens = doc.lower()
    tokens = tokenizer.tokenize(tokens)

    # remove numbers and one-character words
    doc_tokens = [token for token in tokens if not token.isnumeric() and len(token) > 1]
    return doc_tokens


def add_multigrams(docs, min_count=20):
    '''
    Compose bigrams and trigrams expressions in a document
    '''

    # initialize multigrams detector
    multigrams = Phrases(docs, min_count=min_count)

    for i in range(len(docs)):
        for token in multigrams[docs[i]]:
            if '_' in token:

                # add the detected multigram to the document if it appears n times
                docs[i].append(token)
    return docs


def process_corpus(file_path):
    '''
    Download and process a corpus of documents
    '''

    # define tokenizer and lemmatizer
    tokenizer   = RegexpTokenizer(r'\w+')
    lemmatizer  = WordNetLemmatizer()

    # download and process
    corpus      = list(download_corpus(url=file_path))
    corpus      = [remove_stopwords(doc) for doc in corpus]
    docs_tokens = [tokenize_doc(tokenizer, doc) for doc in corpus]
    docs_lemmas = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs_tokens]

    # create a dictionary and filter out common terms
    docs        = docs_tokens
    dictionary  = Dictionary(docs)
    dictionary.filter_extremes(no_below=10)

    # convert to bag-of-words
    corpus      = [dictionary.doc2bow(doc) for doc in docs]

    # save corpus and dictionary
    MmCorpus.serialize("data/corpus.mm", corpus)
    dictionary.save("data/dictionary.dict")
    print("INFO: Successfully processed corpus")


def shuffle_corpus(corpus):
    '''
    Shuffle the order of documents in a corpus
    '''

    # manually shuffle the corpus
    shuffled_corpus = []
    indexed_corpus  = {idx: doc for idx, doc in enumerate(corpus)}
    idxs            = np.random.permutation(len(corpus))

    for idx in idxs:
        shuffled_corpus.append(indexed_corpus[idx])

    return corpus


def load_corpus(corpus_file, dictionary_file):
    '''
    Load corpus and dictionary from files
    '''

    corpus      = MmCorpus(corpus_file)
    dictionary  = Dictionary.load(dictionary_file)

    print("INFO: Successfully loaded corpus")
    return corpus, dictionary


def create_folds(corpus, folds_size):
    '''
    Split a corpus into n_folds parts
    '''

    indexed_corpus = {idx: doc for idx, doc in enumerate(corpus)}
    folds = []

    for i in range(0, len(corpus), folds_size):
        folds.append([indexed_corpus[idx] for idx in range(i, i+folds_size)])

    return folds

def log_model(logger, model_name, params, train_time, perplexity):
    '''
    Log model training information and metrics
    '''

    info = {
        "model_name"    : model_name,
        "params"        : params,
        "train_time"    : train_time,
        "perplexity"    : perplexity
    }
    logger.log(logging.INFO, "Trained model", extra=info)


def compute_perplexity(model, corpus, n_docs):
    '''
    Compute the average perplexity of a model on a corpus
    '''

    # select docs to be evaluated
    idxs = np.random.choice(a=np.arange(len(corpus)), size=n_docs, replace=False)

    tot_log_likelihood  = 0
    tot_words           = 0

    for idx in idxs:
        doc = corpus[idx]

        # get the topic distribution for the document
        doc_topics_mixture  = model[doc]

        for word_id, word_count in doc:

            # the likelihood of a word is computed as the sum of its probability to appear in any topic of the doc
            # this is discounted by the mixture value of the topic in the doc
            word_prob           = sum([prob*model.get_topics()[int(topic_id)][word_id] for topic_id, prob in doc_topics_mixture])
            # the total likelihood of the document is the sum of the log likelihood of all words
            tot_log_likelihood += np.log(word_prob+1e-10)*word_count
            tot_words          += word_count

    # the perplexity is the exponential of the negative average log likelihood
    perplexity      = np.exp(-tot_log_likelihood/tot_words)
    return perplexity


def save_results(params, file_path="results/perplexities.csv"):

    results = pd.read_csv(file_path)
    results = results._append(params, ignore_index=True)
    results.to_csv(file_path, index=False)


def sample_topics(model, corpus, n_samples=10):
    '''
    Show the number of topics for HDP over n posterior sample
    '''

    n_sampled_topics = []
    for _ in range(n_samples):

        topics = []
        for doc in corpus:
            topic_dist = model.get_topics()

            for word_id, word_count in doc:

                # the probability for a word to be assigned to a topic is proportional to the topic-word and document-topic distribution
                topic_dist_word         = model.lda_alpha * topic_dist[:, word_id]
                topic_dist_normalized   = topic_dist_word / sum(topic_dist)
                topic_id                = np.random.choice(np.arange(len(topic_dist_normalized)), 1, p=topic_dist_normalized)

                topics.append(topic_id[0])
        n_sampled_topics.append(len(set(topics)))

    # save samples to file
    samples = pd.read_csv("results/samples.csv")
    samples = samples._append({"sampled_topics": n_sampled_topics}, ignore_index=True)
    samples.to_csv("results/samples.csv", index=False)

    return n_sampled_topics


def show_results(file_path="results/perplexities.csv"):
    '''
    Show the results of the cross-validation
    '''

    results     = pd.read_csv(file_path, index_col=False)
    results_lda = results.loc[results["model_type"] == "LDA"]
    results_hdp = np.ones(len(results_lda)) * results.loc[results["model_type"] == "HDP"]["perplexity"].values[0]

    plt.plot(results_lda["n_topics"], results_lda["perplexity"], label="LDA")
    plt.plot(results_lda["n_topics"], results_hdp, label="HDP")

    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.show()

def show_samples(file_path="results/samples.csv"):
    '''
    Show the number of topics generated by HDP over n posterior samples
    '''

    samples = pd.read_csv(file_path)
    samples = samples["sampled_topics"].values

    plt.hist(samples, bins=range(min(samples), max(samples)+1), align="left", rwidth=0.8)
    plt.xticks(np.arange(min(samples), max(samples), 1))
    plt.xlabel("Number of Topics")
    plt.ylabel("Frequency")
    plt.show()

show_samples()
