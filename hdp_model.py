from    gensim.models import HdpModel
import  utils
import  time


# set HDP model parameters
# max_chunks  = 100
max_time    = 60
chunk_size  = 256
gamma       = 1
alpha       = 1


def train_hdp(corpus, dictionary, logger):

    _start  = time.time()
    model   = HdpModel(
        corpus      = corpus,
        id2word     = dictionary,
        # max_chunks  = max_chunks,
        chunksize   = chunk_size,
        gamma       = gamma,
        alpha       = alpha
    )

    _end    = time.time()
    print("INFO: Trained HDP model, calculating perplexity ...")

    utils.log_model(
        logger      = logger,
        model_name  = "HDP",
        params      = {
            # "max_chunks" : max_chunks,
            "max_time"   : max_time,
            "chunk_size" : chunk_size
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
hdp_logger          = utils.create_logger("hdp_logger")

# train the HDP model
hdp_model           = train_hdp(
    corpus      = corpus,
    dictionary  = dictionary.id2token,
    logger      = hdp_logger
)

hdp_model.print_topics()
