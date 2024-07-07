from    gensim.models import Phrases
import  logging
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
    file_handler = logging.FileHandler("logs.log")
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
