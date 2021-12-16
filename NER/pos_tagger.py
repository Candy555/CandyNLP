from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training import GradientDescentTrainer
from allennlp_models.structured_prediction.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.predictors import Predictor
from typing import List
from allennlp.common import JsonDict

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128

class MyDatasetReader:
    def __init__(self):
      # UniversalDependenciesDatasetReader(): Reads a file in the conllu Universal Dependencies format
      self.reader = UniversalDependenciesDatasetReader()
      self.train_path = 'https://s3.amazonaws.com/realworldnlpbook/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu'
      self.dev_path = 'https://s3.amazonaws.com/realworldnlpbook/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu'
      self.sampler = BucketBatchSampler(batch_size=32, sorting_keys=["words"])
    
    def data_reader(self):
      train_data_loader = MultiProcessDataLoader(self.reader, self.train_path, batch_sampler=self.sampler)
      dev_data_loader = MultiProcessDataLoader(self.reader, self.dev_path, batch_sampler=self.sampler)
      vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(), dev_data_loader.iter_instances()))
      train_data_loader.index_with(vocab)
      dev_data_loader.index_with(vocab)
      token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_SIZE)
      word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
      return word_embeddings, vocab, train_data_loader, dev_data_loader

class LstmTagger(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),out_features=vocab.get_vocab_size('pos'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                words: Dict[str, torch.Tensor],
                pos_tags: torch.Tensor = None,
                **args) -> Dict[str, torch.Tensor]:
        # mask shape: torch.Size([32, 14])
        mask = get_text_field_mask(words)
        # embeddings shape: torch.Size([32, 14, 128])
        embeddings = self.embedder(words)
        # encoder_out shape: torch.Size([32, 14, 128])
        encoder_out = self.encoder(embeddings, mask)
        # tag_logits shape: torch.Size([32, 14, 19]) => (batch_size, sent_len, num_labels)
        tag_logits = self.linear(encoder_out)

        output = {"tag_logits": tag_logits}
        if pos_tags is not None:
            self.accuracy(tag_logits, pos_tags, mask)
            # tagger loss
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, pos_tags, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

class UniversalPOSPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, words: List[str]) -> JsonDict:
        return self.predict_json({"words" : words})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        words = json_dict["words"]
        # This is a hack - the second argument to text_to_instance is a list of POS tags
        # that has the same length as words. We don't need it for prediction so
        # just pass words.
        return self._dataset_reader.text_to_instance(words, words)

def main():
  dataset_reader = MyDatasetReader()
  word_embeddings, vocab, train_data_loader, dev_data_loader = dataset_reader.data_reader()
  encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))
  model = LstmTagger(word_embeddings, encoder, vocab)
  optimizer = optim.Adam(model.parameters())
  trainer = GradientDescentTrainer(model=model,optimizer=optimizer,data_loader=train_data_loader,validation_data_loader=dev_data_loader,patience=10,num_epochs=10,cuda_device=-1)
  trainer.train()
  predictor = UniversalPOSPredictor(model, reader)
  tokens = ['The', 'dog', 'ate', 'the', 'apple', '.']
  logits = predictor.predict(tokens)['tag_logits']
  tag_ids = np.argmax(logits, axis=-1)
  print([vocab.get_token_from_index(tag_id, 'pos') for tag_id in tag_ids])
    
if __name__ == "__main__":
    main()