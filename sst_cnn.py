#!/usr/bin/python
# -*- coding:utf8 -*-
from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, CnnEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training import GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.common import JsonDict
from overrides import overrides
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

EMBEDDING_DIM = 128 
HIDDEN_DIM = 128

# 数据集读取
class MyDatasetReader:
    def __init__(self):
        # SentimentTreeBankDataset
        # SingleIdTokenIndexer` produces an array of shape (num_tokens,)
        self.token_indexer = SingleIdTokenIndexer(token_min_padding_length=5)
        self.reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': self.token_indexer})
        self.sampler = sampler = BucketBatchSampler(batch_size = 128, sorqting_keys=["tokens"])
        self.train_data_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/train.txt'
        self.dev_data_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/dev.txt'

    def data_reader(self):
        train_data_loader = MultiProcessDataLoader(self.reader, self.train_data_path, batch_sampler = self.sampler)
        dev_data_loader = MultiProcessDataLoader(self.reader, self.dev_data_path, batch_sampler = self.sampler)
        vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(), dev_data_loader.iter_instances()),min_count={'tokens': 3})
        token_embedding = Embedding(num_embeddings = vocab.get_vocab_size('tokens'), embedding_dim = EMBEDDING_DIM)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        return word_embeddings, vocab, train_data_loader, dev_data_loader

# 构建分类器，编码器选用cnn
class CnnClassifier(Model):
    def __init__(self, embedder, vocab, positive_label: str = '4'):
        super().__init__(vocab)
        self.embbedder = embedder
        # self.encoder = CnnEncoder(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
        self.encoder = CnnEncoder(embedding_dim=EMBEDDING_DIM, num_filters=8, ngram_filter_sizes=(2, 3, 4, 5))
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        positive_index = vocab.get_token_index(positive_label, namespace='labels')

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        # return mask padding matrix (tokens:text_field_tensors)
        mask = get_text_field_mask(tokens)
        # embedding 
        embedding = self.embbedder(tokens)
        # encoder means lstm
        encoder_outputs = self.encoder(embedding, mask)
        logits = self.linear(encoder_outputs)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset),**self.f1_measure.get_metric(reset)}

#构建预测器
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = dataset_reader._tokenizer or SpacyTokenizer()

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance([str(t) for t in tokens])

#训练过程
def main():
    # read dataset
    dataset_reader = MyDatasetReader()
    # get data for training
    word_embeddings, vocab, train_data_loader, dev_data_loader = dataset_reader.data_reader()
    # build model
    model = CnnClassifier(word_embeddings, vocab)
    # choose optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # train model
    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)
    trainer = GradientDescentTrainer(model=model,optimizer=optimizer,data_loader=train_data_loader,validation_data_loader=dev_data_loader,patience=10,num_epochs=10,cuda_device=-1)
    trainer.train() 
    # predict result
    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict('We are very happy to show you the 🤗 Transformers library.')['logits']
    label_id = np.argmax(logits)
    print(label_id)
    

if __name__ == "__main__":
    main()