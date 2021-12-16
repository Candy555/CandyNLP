import csv
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.training import GradientDescentTrainer
from overrides import overrides

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128

class NERDatasetReader(DatasetReader):
    def __init__(self, file_path: str, token_indexers: Dict[str, TokenIndexer]=None):
        super().__init__()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.instances = []
        file_path = cached_path(file_path)
        sentence = []
        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            next(csv_file)
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] and sentence:
                    tokens, labels = self._convert_sentence(sentence)
                    self.instances.append(self.text_to_instance(tokens, labels))
                    sentence = [row]
                else:
                    sentence.append(row)

            if sentence:
                tokens, labels = self._convert_sentence(sentence)
                self.instances.append(self.text_to_instance(tokens, labels))

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: List[str]=None):
        fields = {}

        text_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = text_field
        if labels:
            fields['labels'] = SequenceLabelField(labels, text_field)

        return Instance(fields)

    def _convert_sentence(self, rows: List[Tuple[str]]) -> Tuple[List[Token], List[str]]:
        """Given a list of rows, returns tokens and labels."""
        _, tokens, _, labels = zip(*rows)
        tokens = [Token(t) for t in tokens]

        # NOTE: the original dataset seems to confuse gpe with geo, and the distinction
        # seems arbitrary. Here we replace both with 'gpe'
        labels = [label.replace('geo', 'gpe') for label in labels]
        return tokens, labels

    @overrides
    def _read(self, split: str):
        for i, inst in enumerate(self.instances):
            if split == 'train' and i % 10 != 0:
                yield inst
            elif split == 'dev' and i % 10 == 0:
                yield inst

class MyDatasetReader:
     def __init__(self):
        # dataset example:
        # Word	POS	Tag
        # Thousands	NNS	O
        # of	IN	O
        # demonstrators	NNS	O
        # have	VBP	O
        # marched	VBN	O
        # through	IN	O
        # London	NNP	B-geo
        # to	TO	O
        # protest	VB	O
        # the	DT	O
        # war	NN	O
        # in	IN	O
        # Iraq	NNP	B-geo
        # and	CC	O
        # demand	VB	O
        # the	DT	O
        # withdrawal	NN	O
        # of	IN	O
        # British	JJ	B-gpe
        # troops	NNS	O
        # from	IN	O
        # that	DT	O
        # country	NN	O
        self.reader = NERDatasetReader('https://s3.amazonaws.com/realworldnlpbook/data/entity-annotated-corpus/ner_dataset.csv')
        self.sampler = BucketBatchSampler(batch_size=16, sorting_keys=["tokens"])

    def data_reader(self):
        train_data_loader = MultiProcessDataLoader(self.reader, 'train', batch_sampler=self.sampler)
        dev_data_loader = MultiProcessDataLoader(self.reader, 'dev', batch_sampler=self.sampler)
        vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(),dev_data_loader.iter_instances()))
        train_data_loader.index_with(vocab)
        dev_data_loader.index_with(vocab)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_SIZE)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        return word_embeddings, vocab, train_data_loader, dev_data_loader

class LstmTagger(Model):
    # not lstm+crf which is classification problem, compute each word logits
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.hidden2labels = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                             out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.f1 = SpanBasedF1Measure(vocab, tag_namespace='labels')

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2labels(encoder_out)
        output = {'logits': logits}
        if labels is not None:
            self.accuracy(logits, labels, mask)
            self.f1(logits, labels, mask)
            output['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_metrics = self.f1.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'prec': f1_metrics['precision-overall'],
                'rec': f1_metrics['recall-overall'],
                'f1': f1_metrics['f1-measure-overall']}

def predict(tokens: List[str], model: LstmTagger) -> List[str]:
    token_indexers = {'tokens': SingleIdTokenIndexer()}
    tokens = [Token(t) for t in tokens]
    inst = Instance({'tokens': TextField(tokens, token_indexers)})
    logits = model.forward_on_instance(inst)['logits']
    label_ids = np.argmax(logits, axis=1)
    labels = [model.vocab.get_token_from_index(label_id, 'labels')
              for label_id in label_ids]
    return labels

def main():
    dataset_reader = MyDatasetReader()
    word_embeddings, vocab, train_data_loader, dev_data_loader = dataset_reader.data_reader()
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, bidirectional=True, batch_first=True))
    model = LstmTagger(word_embeddings, encoder, vocab)
    optimizer = optim.Adam(model.parameters())
    trainer = GradientDescentTrainer(model=model,optimizer=optimizer,data_loader=train_data_loader,validation_data_loader=dev_data_loader,patience=10,num_epochs=20,cuda_device=-1)
    trainer.train()
    tokens = ['Apple', 'is', 'looking', 'to', 'buy', 'U.K.', 'startup', 'for', '$1', 'billion', '.']
    labels = predict(tokens, model)
    # Apple/B-org is/O looking/O to/O buy/O U.K./O startup/O for/O $1/O billion/O ./O
    print(' '.join('{}/{}'.format(token, label) for token, label in zip(tokens, labels)))

if __name__ == "__main__":
    main()