from    gensim.models import LdaModel
import  utils
import  time


# set LDA model parameters
n_topics    = 100
chunk_size  = 256              # documents processed at a time
passes      = 1                 # how many times the model goes through the corpus to LEARN the topics
alpha       = "symmetric"       # prior for the per-document topic distribution
n_iters     = 50                # how many times the model goes through the corpus to INFER the topics
eval_every  = None              # perplexity estimation


def train_lda(corpus, dictionary, logger):

    _start  = time.time()
    model   = LdaModel(
        corpus      = corpus,
        num_topics  = n_topics,
        id2word     = dictionary,
        chunksize   = chunk_size,
        passes      = passes,
        alpha       = alpha,
        eta         = "symmetric",
        eval_every  = eval_every,
        iterations  = n_iters
    )

    _end    = time.time()
    print("INFO: Trained LDA model, calculating perplexity ...")

    utils.log_model(
        logger      = logger,
        model_name  = "LDA",
        params      = {
            "n_topics"      : n_topics,
            "chunk_size"    : chunk_size,
            "passes"        : passes,
            "alpha"         : alpha,
            "n_iters"       : n_iters
        },
        train_time  = _end - _start,
        perplexity  = utils.compute_perplexity(model=model, corpus=corpus, n_docs=10)
    )

    print("INFO: Evaluated model")
    return model


# define the file path and load the corpus
# file_path           = "https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz"
# corpus, dictionary  = utils.process_corpus(file_path)
corpus, dictionary  = utils.load_corpus(corpus_file="corpus.mm", dictionary_file="dictionary.dict")
# load the dictionary
_                   = dictionary[0]

# set up logging
lda_logger          = utils.create_logger("lda_logger")

# train the LDA model
lda_model           = train_lda(
    corpus      = corpus,
    dictionary  = dictionary.id2token,
    logger      = lda_logger
)
