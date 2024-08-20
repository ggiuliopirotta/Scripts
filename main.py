import  models
import  utils

# load the corpus and dictionary
corpus, dictionary  = utils.load_corpus(corpus_file="data/corpus.mm", dictionary_file="data/dictionary.dict")
_                   = dictionary[0]

# cross validate LDA
for n in range(20, 161, 10):

    model_type = "LDA"
    model, n_topics, perplexity = models.cross_validate(
        corpus          = corpus,
        dictionary      = dictionary,
        model_type      = model_type,
        model_params    = {"n_topics": n},
        n_folds         = 5
    )
    params = {
        "model_type"    : model_type,
        "n_topics"      : n,
        "perplexity"    : perplexity
    }

    # save results to csv file
    utils.save_results(params=params)


# train HDP
model_type = "HDP"
model, n_topics, perplexity = models.cross_validate(
    corpus          = corpus,
    dictionary      = dictionary,
    model_type      = model_type,
    model_params    = {},
    n_folds         = 2
)

params = {
    "model_type"    : model_type,
    "n_topics"      : None,
    "perplexity"    : perplexity
}

# save results to csv file
utils.save_results(params=params)

n_sampled_topics = utils.sample_topics(model, corpus, n_samples=5)
print(n_sampled_topics)
