import requests
import torch
from functools import partial
from torchtext.data import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

MIN_WORD_FREQUENCY = 5
CBOW_N_WORDS = 4
SKIPGRAM_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 2048
BATCH_SIZE = 2
ARTICLES = ["Grammatica de su sardu", "Sardigna", "Mare Mediterràneu", "Terra"
            "Sardu campidanesu", "Gadoni", "Massimo Pittau", "Santandria",
            "Mesaustu", "Augustu (imperadore romanu)",
            "Cristianèsimu", "Provìntzia de Nùgoro", "Casteddu",
            "Limbas romanzas", "Limba italiana", "Edade mèdia",
            "Impèriu bizantinu", "Impèriu otomanu", "Europa",
            "Mare Ruju", "Penìsula àraba", "Àrabos", "Islam"]

def fetch_wikipedia_article(page_title, lang='sc'):
    """
    Fetch the plain text content of a Wikipedia article.
    """
    S = requests.Session()
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        'action': "query",
        'prop': "extracts",
        'exlimit': "max",
        'titles': page_title,
        'explaintext': True,
        'format': "json"
    }

    response = S.get(url=URL, params=PARAMS)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page.get('extract', "Page not found.")

def download_articles(articles):
    text = []
    for article in articles:
        article_text = fetch_wikipedia_article(article)
        text.append(article_text)
    return to_map_style_dataset(text)

def build_vocab(data_iter):
    """
    Builds vocabulary from iterator where data_iter is an iterable of strings (articles).
    """
    tokenizer = get_tokenizer("basic_english")  # You might want to create or specify a tokenizer suitable for Sardinian.
    vocab = build_vocab_from_iterator(
        (tokenizer(article) for article in data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle/center word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(
    model_name, articles, batch_size, shuffle, vocab=None
):

    data_iter = download_articles(articles)
    tokenizer = get_tokenizer("basic_english") 

    if not vocab:
        vocab = build_vocab(data_iter)
        
    text_pipeline = lambda x: vocab(tokenizer(x))
    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab

