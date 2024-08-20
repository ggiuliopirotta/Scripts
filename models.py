import  numpy as np
from    gensim.models import LdaModel, HdpModel
import  utils
import  time


# set model parameters
lda_topics  = 100           # no. of topics
chunk_size  = 256           # documents processed at a time
alpha       = "symmetric"   # prior for the document-topic distribution
eta         = "symmetric"   # prior for the topic-word distribution


def train_lda(
        corpus,
        dictionary,
        logger,
        n_topics    = lda_topics,
        chunk_size  = chunk_size,
        alpha       = alpha,
        eta         = eta
    ):

    _start  = time.time()
    model   = LdaModel(
        corpus      = corpus,
        num_topics  = n_topics,
        id2word     = dictionary,
        chunksize   = chunk_size,
        alpha       = alpha,
        eta         = eta
    )
    _end    = time.time()

    print("INFO: Trained LDA model, calculating perplexity ...")
    perplexity = utils.compute_perplexity(model=model, corpus=corpus, n_docs=10)

    utils.log_model(
        logger      = logger,
        model_name  = "LDA",
        params      = {
            "n_topics"      : n_topics,
            "chunk_size"    : chunk_size,
            "alpha"         : alpha,
            "eta"           : eta
        },
        train_time  = _end - _start,
        perplexity  = perplexity
    )

    print("INFO: Evaluated model")
    return model, perplexity


def train_hdp(corpus, dictionary, logger, chunk_size=chunk_size):

    _start  = time.time()
    model   = HdpModel(
        corpus      = corpus,
        id2word     = dictionary,
        chunksize   = chunk_size,
        T           = 150
    )
    _end    = time.time()

    print("INFO: Trained HDP model, calculating perplexity ...")
    perplexity = utils.compute_perplexity(model=model, corpus=corpus, n_docs=20)

    utils.log_model(
        logger      = logger,
        model_name  = "HDP",
        params      = {
            "chunk_size" : chunk_size
        },
        train_time  = _end - _start,
        perplexity  = perplexity
    )

    print("INFO: Evaluated model")
    return model, perplexity


def cross_validate(corpus, dictionary, model_type, model_params={}, n_folds=5):

    # shuffle and split the corpus into n_folds parts
    folds_size  = len(corpus) // n_folds
    folds       = utils.create_folds(corpus, folds_size)

    # train and evaluate the model on each fold
    perplexities = []
    for i, fold in enumerate(folds):
        print(f"INFO: Training and evaluating {model_type} model on fold {i+1}/{n_folds} ...")

        # train the model on the remaining folds
        train_corpus = []
        for j, docs in enumerate(folds):
            if j != i:
                for doc in docs:
                    train_corpus.append(doc)

        logger          = utils.create_logger(f"{model_type}-n_topics:{model_params.get('n_topics', lda_topics)}-fold:{i}")

        if model_type == "LDA":
            model, perplexity = train_lda(
                corpus      = train_corpus,
                dictionary  = dictionary,
                logger      = logger,
                n_topics    = model_params.get("n_topics", lda_topics),
                chunk_size  = model_params.get("chunk_size", chunk_size),
            )
        else:
            model, perplexity = train_hdp(
                corpus      = train_corpus,
                dictionary  = dictionary,
                logger      = logger,
                chunk_size  = model_params.get("chunk_size", chunk_size)
            )

        perplexities.append(perplexity)

    # compute the average perplexity
    avg_perplexity = np.mean(perplexities)

    # TODO: return best model
    return (model, lda_topics, avg_perplexity) if model_type == "LDA" else (model, None, avg_perplexity)
