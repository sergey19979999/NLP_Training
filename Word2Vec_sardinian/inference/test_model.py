import numpy as np
import torch
import sys

def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        dist = dists[sim_word_id]
        topN_dict[sim_word] = np.round(dist, 3)
    return topN_dict


folder = "Word2Vec_sardinian/model_folder"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
embed_norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
embed_norms = np.reshape(embed_norms, (len(embed_norms), 1))
embeddings = embeddings / embed_norms
embeddings.shape

# tokens from vocabulary
tokens = vocab.get_itos()
len(tokens)
breakpoint()
test = get_top_similar("aristanis")