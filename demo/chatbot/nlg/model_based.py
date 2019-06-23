import numpy as np
import torch
import torch.nn as nn
from .load import init_model, load_dict
from .model import Encoder, Decoder
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelBasedNLG:
    def __init__(self, model_dir):
        self.intent_dict, self.word2index, self.index2word = load_dict(
                os.path.join(model_dir, 'intent_dict.pkl'),
                os.path.join(model_dir, 'dict.pkl'))

        self.slot_size = len(self.intent_dict)
        self.vocab_size = len(self.word2index)
        self.encoder, self.decoder, _ = init_model(self.slot_size, self.vocab_size, train=False)
        state_dict = torch.load(os.path.join(model_dir, 'model.pkl'), map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])

    def predict(self, dialog_act):
        n_hot_input = torch.zeros(1, self.slot_size).to(device)
        for i in range(len(dialog_act)):
            if 'slot' in dialog_act[i]:
                intent_num = self.intent_dict[dialog_act[i]['intent'] + "_" + dialog_act[i]['slot']]
            else:
                intent_num = self.intent_dict[dialog_act[i]['intent']]
            n_hot_input[0][intent_num] = 1

        message = ""
        encoder_outputs, encoder_hidden = self.encoder(n_hot_input)
        response = self.decoder(encoder_hidden, message, self.index2word)
        punctuation = [',', '.', '?', '!', ':']
        min_num_punct = 1000
        
        for i in range(len(dialog_act)):
            if 'filler' in dialog_act[i]:
                response = response.replace("==" + dialog_act[i]['slot'] + '==', dialog_act[i]['filler'])
        # print(response)
        return response


# if __name__ == '__main__':
#     dialog_act = [
#           {
#             "intent": "inform",
#             "slot": "moviename",
#             "filler": "The Revenant"
#           },
#           {
#             "intent": "inform",
#             "slot": "date",
#             "filler": "tomorrow"
#           },
#           {
#             "intent": "inform",
#             "slot": "starttime",
#             "filler": "8pm"
#           },
#           {
#             "intent": "inform",
#             "slot": "theater",
#             "filler": "Regal MacArthur Marketplace Stadium 16"
#           },
#           {
#             "intent": "confirm_question"
#           }
#         ]
#     response = predict(dialog_act)


