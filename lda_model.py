from    gensim.corpora import Dictionary
from    gensim.models import LdaModel
from    gensim.models.callbacks import PerplexityMetric
from    nltk.stem.wordnet import WordNetLemmatizer
from    nltk.tokenize import RegexpTokenizer
import  utils
import  time


def process_corpus(file_path):
    '''
    Download and process a corpus of documents
    '''

    tokenizer   = RegexpTokenizer(r'\w+')
    lemmatizer  = WordNetLemmatizer()

    # download and extract the corpus
    docs        = list(utils.download_corpus(url=file_path))

    # tokenize and lemmatize
    docs_tokens = [utils.tokenize_doc(tokenizer, doc) for doc in docs]
    docs_lemmas = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs_tokens]

    # add bigrams and trigrams
    docs        = utils.add_multigrams(docs_lemmas)

    # create a dictionary and filter out common terms
    dictionary  = Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # convert to bag-of-words
    corpus      = [dictionary.doc2bow(doc) for doc in docs]
    print("INFO: Successfully processed corpus")

    return corpus, dictionary


# set LDA model parameters
n_topics    = 20
chunk_size  = 2000              # documents processed at a time
passes      = 20                # how many times the model goes through the corpus to LEARN the topics
alpha       = "symmetric"       # prior for the per-document topic distribution
n_iters     = 400               # how many times the model goes through the corpus to INFER the topics
eval_every  = n_iters // 10     # perplexity estimation


def train_lda(corpus, dictionary):

    _start  = time.time()
    model   = LdaModel(
        corpus      = corpus,
        num_topics  = n_topics,
        id2word     = dictionary,
        chunksize   = chunk_size,
        passes      = passes,
        alpha       = alpha,
        eta         = 'auto',
        eval_every  = eval_every,
        iterations  = n_iters
    )
    _end    = time.time()

    utils.log_model(
        logger      = lda_logger,
        model_name  = "LDA",
        params      = {
            "n_topics"      : n_topics,
            "chunk_size"    : chunk_size,
            "passes"        : passes,
            "alpha"         : alpha,
            "n_iters"       : n_iters
        },
        train_time  = _end - _start,
        perplexity  = model.log_perplexity(corpus)
    )

    print("INFO: Trained LDA model")
    return model


# define the file path and load the corpus
file_path           = "https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz"
corpus, dictionary  = process_corpus(file_path)

# set up logging
lda_logger          = utils.create_logger("lda_logger")

# load the dictionary
_                   = dictionary[0]
lda_model           = train_lda(corpus=corpus, dictionary=dictionary.id2token)
