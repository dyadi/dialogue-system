import torch.nn as nn
import torch
from torch.autograd import Variable
import random
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, slot_size, hidden_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(Encoder, self).__init__()

        self.slot_size = slot_size
        self.linear = nn.Linear(slot_size, hidden_size)
        self.gru = nn.GRU(hidden_size, output_size, batch_first=True)

    def forward(self, input_seqs, hidden=None):
        # input_seqs (batch_size * hidden_size)
        input_seqs = self.linear(input_seqs.unsqueeze(1))
        outputs, hidden = self.gru(input_seqs, hidden)
        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size,teacher_forcing_ratio, 
                 sos_id, train):
        """Define layers for a vanilla rnn decoder"""
        super(Decoder, self).__init__()

        # init the parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()
        self.loss_function = nn.NLLLoss(ignore_index=1)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
        self.train = train

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        batch_size = inputs.size(1)
        embedded = self.embedding(inputs)
        embedded.view(1, batch_size, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        rnn_output = output.squeeze(0)  # squeeze the time dimension
        output = self.log_softmax(self.out(rnn_output))
        return output, hidden

    def forward(self, context_vector, targets, index2word):

        # context_vector: context_vector from encoder, targets: correct answer
        # Prepare variable for decoder on time_step_0
        # decoder_input: sos.id , hidden: context_vector
        # train
        if self.train:
            loss = 0
            batch_size = context_vector.size(1)
            decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
            decoder_hidden = context_vector
            max_target_length = targets.size(1)
            output_example = [[] for i in range(batch_size)]
            eos_start = [0] * batch_size
            targets_eos = [0] * batch_size
            targets_belu = []
            # print(max_target_length)
            # print(decoder_hidden)
            decoder_outputs = Variable(torch.zeros(
                batch_size,
                max_target_length
            ))  # (time_steps, batch_size, vocab_size)

            decoder_input = decoder_input.to(device)
            decoder_outputs = decoder_outputs.to(device)

            # max_length
            tmp = 0
            max_length = max_target_length
            for t in range(max_target_length):
                for i in range(batch_size):
                    if targets[i, t] != 1:
                        break
                    if i == batch_size - 1:
                        tmp = 1
                        max_length = t + 1
                if tmp == 1:
                    break

            # Unfold the decoder RNN on the time dimension
            for t in range(max_length - 1):
                # print(decoder_input.size())
                use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False
                decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                loss += self.loss_function(decoder_outputs_on_t, targets[:, t + 1])
                # decoder_outputs[:,t] = decoder_outputs_on_t
                if use_teacher_forcing and t != max_target_length - 1:
                    #print('yes')
                    decoder_input = targets[:, t + 1].unsqueeze(0)
                    # print(decoder_input)
                else:
                    #print('no')
                    decoder_input = self._decode_to_index(decoder_outputs_on_t)

                # print(index2word[int(decoder_input[0][0])])
                # print(decoder_input)
                for j in range(batch_size):
                    if decoder_input[0][j] != 1 and eos_start[j] == 0:
                        output_example[j].append(index2word[int(decoder_input[0][j])])
                    elif eos_start[j] == 0:
                        eos_start[j] = t
            # Calculate the bleu score
            total_bleu_score = 0            
            for j in range(batch_size):
                targets_eos[j] = (targets[j] == 1).nonzero()[0]
                tmp_target = []
                for k in range(1, targets_eos[j]):
                    tmp_target.append(index2word[int(targets[j][k])])
                total_bleu_score += sentence_bleu([tmp_target], output_example[j])
                targets_belu.append(tmp_target)
                # else:
                  #   break
            total_bleu_score /= batch_size
            #print('finish')
            #print(output_example)
            

            loss /= (max_length - 1)
            return decoder_outputs, decoder_hidden, loss, total_bleu_score, output_example, targets_belu
        
        # # evaluate
        else:
            batch_size = 1
            max_response_length = 100
            decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size])).to(device)
            decoder_hidden = context_vector
            decoder_outputs = Variable(torch.zeros(
                 batch_size,
                 max_response_length
            )).to(device)
            response = ""
            punctuation = [',', '.', '?', '!', ':', '(', ')']
            for t in range(max_response_length):
                decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

                decoder_input = self._decode_to_index(decoder_outputs_on_t)
                if index2word[int(decoder_input[0][0])] != "==EOS==":
                    if index2word[int(decoder_input[0][0])] in punctuation:
                        response = response[:-1]
                    if len(response) > 2:
                        if response[-2] in punctuation and response[-2] != ',':
                            response += (index2word[int(decoder_input[0][0])] + ' ').capitalize()
                        else:
                            response += (index2word[int(decoder_input[0][0])] + ' ')
                    else:
                        response += (index2word[int(decoder_input[0][0])] + ' ').capitalize()
                else:
                    #print(t)
                    break
            #print(index2word)
            return response


    def _decode_to_index(self, decoder_output):
        """
        evaluate on the logits, get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        # S = 1 x B, 1 is the index of top1 class
        index = index.transpose(0, 1).to(device)
        return index
