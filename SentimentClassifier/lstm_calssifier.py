from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
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

class MyDatasetReader:
    # Sentence -> Word ID -> Word Embedding
    def __init__(self):
        # SentimentTreeBankDataset 树结构解析 
        self.reader = StanfordSentimentTreeBankDatasetReader()
        # BucketBatchSampler that groups instances into buckets of similar lengths
        self.sampler = BucketBatchSampler(batch_size = 128, sorting_keys=["tokens"])
        self.train_data_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/train.txt'
        self.dev_data_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/dev.txt'

    def data_reader(self):
        train_data_loader = MultiProcessDataLoader(self.reader, self.train_data_path, batch_sampler = self.sampler)
        dev_data_loader = MultiProcessDataLoader(self.reader, self.dev_data_path, batch_sampler = self.sampler)
        # Vocabulary class that manages mapping from units to IDs.
        # vocab.get_token_from_index(10,'tokens') = 'is' 词表
        vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(), dev_data_loader.iter_instances()),min_count={'tokens': 3})
        # In this step: Word ID -> Word Embedding
        # num_embeddings : `int` Means size of the dictionary of embeddings (vocabulary size).
        # embedding_dim : `int` Means the size of each embedding vector.
        token_embedding = Embedding(num_embeddings = vocab.get_vocab_size('tokens'), embedding_dim = EMBEDDING_DIM)
        # word_embeddingsBasicTextFieldEmbedder((token_embedder_tokens): Embedding()) => Dict
        # A dictionary mapping token embedder names to implementations.
        # These names should match the corresponding indexer used to generate
        # the tensor passed to the TokenEmbedder.
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}) 
        return word_embeddings, vocab, train_data_loader, dev_data_loader, self.reader

class LstmClassifier(Model):
    def __init__(self, embedder, vocab, positive_label: str = '4'):
        super().__init__(vocab)
        self.embbedder = embedder # actually word_embeddings
        # create LSTM input = EMBEDDING_DIM, output = HIDDEN_DIM
        # PytorchSeq2VecWrapper(Seq2VecEncoder) => Seq2VecEncoder transform tensor matrix into vector
        self.encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
        # transform embedding to vector
        self.linear = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        # vocab.get_token_index('is','tokens') = 10 
        # In this step: Sentence -> Word ID
        positive_index = vocab.get_token_index(positive_label, namespace='labels')

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        # forward function is where the computation happens
        # return mask padding matrix (tokens:text_field_tensors)

        """
        tokens {'tokens': tensor([[6081,    4,   16,    5, 2037,    4, 4578, 3156, 1005,    1,    2,    0],
        [  27,    9,    7, 1252,   88, 4918,   19,    4,  234,  154,    2,    0],
        [   1,  113,    4,  324, 2583,   50,    1, 3355,  185,   51,    0,    0],
        [   1,  133,   38,   21,  899,  351,    9,  127,   20, 5742,    2,    0],
        [  14,  148,  212,  113,   73,  111,    4,   20,   56,   26,    2,    0],
        [  14, 3884, 1836,  370,  404, 6343,   55,    7,    1, 4867,    2,    0],
        [   1,   76,    9,    4,  340,   29,    1,    3, 4419,  185,    0,    0],
        [  45,  279,   26,  218,   63, 3996,   12,    4, 5487,    2,    0,    0],
        [   1,   55,    4,  893,    3,    1,   30,    1, 3453,  359,    2,    0],
        [4445,    9, 1196,    5,  827, 3672,   19,    1,    5, 1372,    2,    0],
        [ 265,    7,    1,   10,    1,  506,    7, 1411,  450,    1,    2,    0],
        [  14,  118,    1,   39,  884,    8, 1896,    5,  239,    2,    0,    0],
        [  14,  103,  283,   35,   64, 2538,   65,   99, 3152, 5491,    2,    0],
        [  50,   22,   51, 1213, 1635,    6,    4,    1,  217,    2,    0,    0],
        [  14,  439,    3, 4497,   88,    3,   10,    4,  316,    2,    0,    0],
        [1767, 5400,    9,    1,   34, 1133, 5401,    6,   77,    1,    2,    0],
        [  50,  382,  256,   51, 1157,   10,    7,  187, 2206,    2,    0,    0],
        [ 174,    7,   16,  109,    3, 2314, 2038,    8, 3919,    2,    0,    0],
        [ 735,  518,   24, 2589,   67,   28,  152,  522,    1,    2,    0,    0],
        [ 354,   87,    9,    4,    1, 6005,    3,    1,   13,   46,  297,    2],
        [  27,    9,    7,  306,    1, 2021,    5,   45,  492,   42, 4821,    2],
        [1544,   11,   99,  238,   35, 1064,   10,  554,   12,   21,  518,    2],
        [  22,  396,    3, 5217,    5, 2266,  403,  235,   68,    2,    0,    0],
        [  50, 1671,    9,   51,   58, 1922,   16,  299,   14, 4292,    2,    0],
        [  43,    7, 1227,  240,    9,  177, 2643, 3821,  912, 2008,    0,    0],
        [ 265,  312,    3,   53,   91,   26,  296,    4, 1629,   12,    1,    2],
        [  85, 1593,  905,   17,  669,  948,  371,  576,   16,   29,    0,    0],
        [   1,    3,    1, 2382,    5,    3, 1171,    3,    1,  653,    2,    0],
        [  14, 1367,  210, 5028,    4,  665,  716,    6,    1,  544,    2,    0],
        [1334,    1,  928,    5, 4470,    8, 1104,   34,  920,   88,    2,    0],
        [  85,    1,   20,   18,   54,   63, 2299,    5,   54,   80,  335,    2],
        [  22,   80, 1325,   33,    5,   13,   10,   26,   11,   79,    2,    0]])}

        mask tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False]])
        """
        
        mask = get_text_field_mask(tokens)# torch.Size([32, 12]) =>(batch_size, num_tokens)
        
        # get word_embeddings
        embedding = self.embbedder(tokens)# torch.Size([32, 12, 128])=>(batch_size, num_tokens, embedding_dim)
        encoder_outputs = self.encoder(embedding, mask)# torch.Size([32, 128]) =>(batch_size, embedding_dim)
        logits = self.linear(encoder_outputs)# torch.Size([32, 5])=>(batch_size, num_labels)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset),**self.f1_measure.get_metric(reset)}

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

def main():
    # read dataset
    dataset_reader = MyDatasetReader()
    # get data for training
    word_embeddings, vocab, train_data_loader, dev_data_loader, reader = dataset_reader.data_reader()
    # build model
    model = LstmClassifier(word_embeddings, vocab)
    # choose optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # train model
    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)
    trainer = GradientDescentTrainer(model=model,optimizer=optimizer,data_loader=train_data_loader,validation_data_loader=dev_data_loader,patience=10,num_epochs=20,cuda_device=-1)
    # common training loop implement
    # MAX_EPOCHS = 10
    # model = LstmClassifier(word_embeddings, vocab)
    # for epoch in range(MAX_EPOCHS):
    #   for instance, label in train_set:
    #         prediction = model.forward(instance)
    #         loss = loss_function(prediction, labels)
    #         new_model = optimizer(model, loss)
    #         model = new_model
    trainer.train() 
    # predict result

    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict('We are very happy to show you the 🤗 Transformers library.')['logits']
    label_id = np.argmax(logits)

    # sst labels
    # 0  very negative
    # 1  negative
    # 2  neutral
    # 3  positive
    # 4 very positive
    print(label_id)# 0-4
    print(logits)
    

if __name__ == "__main__":
    main()