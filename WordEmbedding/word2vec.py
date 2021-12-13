from collections import Counter

import torch
import torch.optim as optim
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.training import GradientDescentTrainer
from torch.nn import CosineSimilarity
from torch.nn import functional
from skip_gram_reader import SkipGramReader

EMBEDDING_DIM = 256
BATCH_SIZE = 256

class MyDatasetReader:
    def __init__(self):
        self.reader = SkipGramReader()
        
    def data_reader(self):
        # plain text8 corpus
        text8 = self.reader.read('https://realworldnlpbook.s3.amazonaws.com/data/text8/text8')
        # to reduce computation, using part of text8
        # initialize SimpleDataLoader() to call index_with to get word ID
        data_loader = SimpleDataLoader(text8, batch_size = BATCH_SIZE)
        # build word list, min_count means each word lower bound occurence is 5
        # for example : ("the", "dog", "barked", "at", "mailman") => [0, 1, 0, 0, 0] dog is central word
        vocab = Vocabulary.from_instances(text8, min_count = {'token_in': 5, 'token_out': 5})
        # if the size of input is 10000 ,then the size of output is 10000
        embedding_in = Embedding(num_embeddings = vocab.get_vocab_size('token_in'), embedding_dim = EMBEDDING_DIM)
        # tokens  = > ID
        data_loader.index_with(vocab)
        return vocab, data_loader, embedding_in


class SkipGramModel(Model):
    def __init__(self, vocab, embedding_in):
        super().__init__(vocab)
        self.embedding_in = embedding_in
        # fake task
        self.linear = torch.nn.Linear(in_features = EMBEDDING_DIM, out_features = vocab.get_vocab_size('token_out'), bias = False)

    def forward(self, token_in, token_out):
        # converts input tensors(word IDs) to word embeddings
        embedded_in = self.embedding_in(token_in)
        logits = self.linear(embedded_in)
        loss = functional.cross_entropy(logits, token_out)

        return {'loss': loss}

def get_related(token: str, embedding: Model, vocab: Vocabulary, num_synonyms: int = 10):
    """Given a token, return a list of top N most similar words to the token."""
    token_id = vocab.get_token_index(token, 'token_in') # example vocab.get_token_index("dog", 'token_in') => 1187 ID 
    # get skip_gram embedding 
    token_vec = embedding.weight[token_id] # torch.Size([256])
    cosine = CosineSimilarity(dim = 0)
    sims = Counter()

    for index, token in vocab.get_index_to_token_vocabulary('token_in').items():
        sim = cosine(token_vec, embedding.weight[index]).item()
        sims[token] = sim

    return sims.most_common(num_synonyms)

def main():
    # dataset reader
    dataset_reader = MyDatasetReader()
    vocab, data_loader, embedding_in = dataset_reader.data_reader()
    # initialize SkipGramModel
    model = SkipGramModel(vocab = vocab, embedding_in = embedding_in)
    optimizer = optim.Adam(model.parameters())
    trainer = GradientDescentTrainer(model = model, optimizer = optimizer, data_loader = data_loader, num_epochs = 5, cuda_device = -1)
    # training is word pairs like (input word, output word) 
    # final output is probabilty distribution
    trainer.train()
    #[('one', 1.0), ('five', 0.9210125803947449), ('six', 0.9103827476501465), ('eight', 0.9083806872367859), ('three', 0.9069738984107971), ('seven', 0.9066463708877563), ('four', 0.903078556060791), ('nine', 0.9021583199501038), ('two', 0.8900124430656433), ('century', 0.8656787872314453)]
    # [('december', 1.0), ('april', 0.9368302822113037), ('july', 0.9221360087394714), ('september', 0.9131699204444885), ('march', 0.8920833468437195), ('february', 0.8917140364646912), ('miles', 0.8873108625411987), ('mid', 0.8868160843849182), ('bc', 0.8862193822860718), ('budget', 0.8848403692245483)]
    print(get_related('one', embedding_in, vocab))
    print(get_related('december', embedding_in, vocab))

if __name__  == "__main__":
    main()