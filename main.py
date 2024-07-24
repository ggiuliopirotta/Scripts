import  models
import  utils

# define the file path and load the corpus
# file_path           = "https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz"
# corpus, dictionary  = utils.process_corpus(file_path)
corpus, dictionary  = utils.load_corpus(corpus_file="corpus.mm", dictionary_file="dictionary.dict")
# load the dictionary
_                   = dictionary[0]

# cross validate
# for n in range(10, 201, 10):
#
#     model_type = "LDA"
#     model, n_topics, perplexity = models.cross_validate(
#         corpus          = corpus,
#         dictionary      = dictionary,
#         model_type      = model_type,
#         model_params    = {"n_topics": n},
#         n_folds         = 5
#     )
#     params = {
#         "model_type"    : model_type,
#         "n_topics"      : n,
#         "perplexity"    : perplexity
#     }
#
#     # save results to csv file
#     utils.save_results(params=params)

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
#
# # save results to csv file
# utils.save_results(params=params)

n_sampled_topics = utils.sample_topics(model, corpus, n_samples=100)
print(n_sampled_topics)