from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training import GradientDescentTrainer
from overrides import overrides
from lstm_calssifier import LstmClassifier

EMBEDDING_DIM = 16
HIDDEN_DIM = 16

class TatoebaSentenceReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]=None):
        super().__init__()
        self.tokenizer = CharacterTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens, label=None):
        fields = {}
        fields['tokens'] = TextField(tokens, self.token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                lang_id, sent = line.rstrip().split('\t')
                tokens = self.tokenizer.tokenize(sent)
                yield self.text_to_instance(tokens, lang_id)

class MyDatasetReader:
    def __init__(self):
        self.reader = TatoebaSentenceReader()
        self.train_path = 'https://s3.amazonaws.com/realworldnlpbook/data/tatoeba/sentences.top10langs.train.tsv'
        self.dev_path = 'https://s3.amazonaws.com/realworldnlpbook/data/tatoeba/sentences.top10langs.dev.tsv' 
        self.sampler = BucketBatchSampler(batch_size=32, sorting_keys=["tokens"])

    def data_reader(self):
        train_data_loader = MultiProcessDataLoader(self.reader, self.train_path, batch_sampler=self.sampler)
        dev_data_loader = MultiProcessDataLoader(self.reader, self.dev_path, batch_sampler=self.sampler)
        vocab = Vocabulary.from_instances(train_data_loader.iter_instances(),min_count={'tokens': 3})
        train_data_loader.index_with(vocab)
        dev_data_loader.index_with(vocab)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        return word_embeddings, vocab, train_data_loader, dev_data_loader

def classify(text: str, model: LstmClassifier):
    # predict result using trained model
    tokenizer = CharacterTokenizer()
    token_indexers = {'tokens': SingleIdTokenIndexer()}
    tokens = tokenizer.tokenize(text)
    instance = Instance({'tokens': TextField(tokens, token_indexers)})
    logits = model.forward_on_instance(instance)['logits']
    label_id = np.argmax(logits)
    label = model.vocab.get_token_from_index(label_id, 'labels')
    # text: Kurşun kalemin yok, değil mi?, label: tur
    print('text: {}, label: {}'.format(text, label))

def main():
    # read language detection dataset
    dataset_reader = MyDatasetReader()
    word_embeddings, vocab, train_data_loader, dev_data_loader = dataset_reader.data_reader()
    # initialize LSTM Model
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab, positive_label='eng')
    # define optimizer
    optimizer = optim.Adam(model.parameters())
    # initialize trainer
    trainer = GradientDescentTrainer(model=model,optimizer=optimizer,data_loader=train_data_loader,validation_data_loader=dev_data_loader,patience=10,num_epochs=10,cuda_device=-1)
    # train model
    trainer.train()
    # predict result 
    print(classify('Take your raincoat in case it rains.', model))

if __name__ == "__main__":
    main()