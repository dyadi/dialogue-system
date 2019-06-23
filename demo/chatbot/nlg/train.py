import numpy as np
import torch
import torch.nn as nn
from load import build_vocab, load_data, init_model, init_dataset
import nltk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#nltk.download('punkt')
class NLG(object):
    def __init__(self, filename, epoch=200):
        super(NLG, self).__init__()
        self.intent_dict, self.text, self.intent_nhot = load_data(filename)
        self.word2index, self.index2word, self.text = build_vocab(self.text)
        self.dataset, self.dataloader = init_dataset(self.intent_nhot, self.text)
        self.training_epoch = epoch
        self.slot_size = len(self.intent_dict)
        self.vocab_size = len(self.word2index)
        self.encoder, self.decoder, self.optimizer = init_model(self.slot_size, self.vocab_size)
        self.train(self.training_epoch)

    def train(self, EPOCH):
        for epoch in range(EPOCH):
            total_loss = 0
            total_bleu_score = 0
            for i, (n_hot_slot, message) in enumerate(self.dataloader):
                encoder_outputs, encoder_hidden = self.encoder(n_hot_slot)
                decoder_outputs, decoder_hidden, loss, bleu_score, predict_sentence, target_sentence = self.decoder(encoder_hidden, message, self.index2word)
                total_loss += loss
                total_bleu_score += bleu_score
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 50 == 0 and i != 0:
                    print("Epoch = " + str(epoch) + ", " + "Loss = " + str(total_loss / 50) + ", " + "BLEU_Score = " + str(total_bleu_score / 50))
                    print("Predict_sentence:")
                    print(predict_sentence[1])
                    print("Target_sentence:")
                    print(target_sentence[1])
                    total_loss = 0
                    total_bleu_score = 0

            if epoch % 10 == 0:
                filename = str(epoch) + ".pkl"
                torch.save({
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, filename)


if __name__ == '__main__':
    nlg = NLG('../data/movie.json')
