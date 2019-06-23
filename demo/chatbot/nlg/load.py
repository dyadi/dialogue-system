import numpy as np
import torch
import json
import nltk
import copy
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .model import Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def takeSecond(elem):
    return elem[1]

def init_dataset(intent_nhot, text, batch_size=32):
    intent_list = intent_nhot
    intent = torch.zeros((len(intent_list), len(intent_list[0]))).to(device)
    text = text.to(device)
    for i in range(len(intent_list)):
        intent[i] = intent_list[i]
    dataset = TensorDataset(intent, text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader

def init_model(slot_size, vocab_size, embedding_size=512, output_size=512,
               hidden_size=512, teaching_ratio=0.6, learning_rate=5e-4, sos_id=0, train=True):
    encoder = Encoder(slot_size, embedding_size, output_size).to(device)
    decoder = Decoder(hidden_size, vocab_size, teacher_forcing_ratio=teaching_ratio,
                      sos_id=sos_id, train=train).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=learning_rate)
    return encoder, decoder, optimizer

def load_dict(intent_dict_file, word_dict_file):
    intent_dict = torch.load(intent_dict_file)
    word2index = torch.load(word_dict_file)['word2index']
    index2word = torch.load(word_dict_file)['index2word']
    return intent_dict, word2index, index2word

def load_data(filename):
    print('Loading data ...')
    with open(filename, 'r', encoding = 'latin-1') as f:
        data = json.load(f)

    # total data size = 2889, train_data size = 2000 
    total_text = []
    intent_dict = {}
    intent = []
    intent_nhot = []
    slot_filler = []
    intent_set = set()
    intent_list = []
    intent_count = {}
    for i in data:
        for j in i['message']:
            intent.append([])
            slot_filler.append({})
            text = j['text']
            filler_list = []
            for k, element in enumerate(j['dialog_act']):
                if 'filler' in element:
                    filler_list.append([k, len(element['filler'])])
                else:
                    filler_list.append([k, 0])

            filler_list.sort(key= takeSecond, reverse = True)
            tmp_Dialog_act = copy.deepcopy(j['dialog_act'])
            for k, index in enumerate(filler_list):
                j['dialog_act'][k] = tmp_Dialog_act[index[0]]

            for k in j['dialog_act']:
                if 'slot' in k:
                    intent_string = '{}_{}'.format(k['intent'], k['slot'])
                else:
                    intent_string = k['intent']
                if intent_string not in intent_set:
                    intent[-1].append(len(intent_set))
                    intent_count[intent_string] = 1
                    intent_dict[intent_string] = len(intent_set)
                    intent_list.append(intent_string)
                    intent_set.add(intent_string)
                else:
                    intent[-1].append(intent_dict[intent_string])
                    intent_count[intent_string] += 1
                if 'slot' in k and 'filler' in k:
                    slot_filler[-1][k['slot']] = k['filler']
                    
                    if isinstance(k['filler'], str):
                        text = text.replace(" " + k['filler'] + " ", ' =={}== '.format(k['slot']))
                        text = text.replace(" " + k['filler'] + ".", ' =={}== .'.format(k['slot']))
                        text = text.replace(" " + k['filler'] + ",", ' =={}== ,'.format(k['slot']))
                        text = text.replace(" " + k['filler'] + "?", ' =={}== ?'.format(k['slot']))
                        text = text.replace(" " + k['filler'] + ":", ' =={}== :'.format(k['slot']))
                        if text.find(" ") == -1:
                            text = text.replace(k['filler'], '=={}=='.format(k['slot']))
                        elif text.find(k['filler']) == 0:
                            text = text.replace(k['filler'] + " ", '=={}== '.format(k['slot']))
                            text = text.replace(k['filler'] + ".", '=={}== .'.format(k['slot']))
                            text = text.replace(k['filler'] + ",", '=={}== ,'.format(k['slot']))
                            text = text.replace(k['filler'] + "?", '=={}== ?'.format(k['slot']))
                            text = text.replace(k['filler'] + ":", '=={}== :'.format(k['slot']))
                            text = text.replace(k['filler'], '=={}=='.format(k['slot']))
                        elif text.find(k['filler']) == len(text) - len(k['filler']):
                            text = text.replace(" " + k['filler'], ' =={}=='.format(k['slot']))

                    else:
                        for l in k['filler']:
                            if 'options' in l:
                                text = text.replace(" " + l['options'] + " " , ' =={}== '.format(k['slot']))
                                text = text.replace(" " + l['options'] + "." , ' =={}== .'.format(k['slot']))
                                text = text.replace(" " + l['options'] + "," , ' =={}== ,'.format(k['slot']))
                                text = text.replace(" " + l['options'] + "?" , ' =={}== ?'.format(k['slot']))
                                text = text.replace(" " + l['options'] + ":" , ' =={}== :'.format(k['slot']))
                                if text.find(" ") == -1:
                                    text = text.replace(l['options'], '=={}=='.format(k['slot']))
                                elif text.find(l['options']) == 0:
                                    text = text.replace(l['options'] + " " , '=={}== '.format(k['slot']))
                                    text = text.replace(l['options'] + "." , '=={}== .'.format(k['slot']))
                                    text = text.replace(l['options']+ "," , '=={}== ,'.format(k['slot']))
                                    text = text.replace(l['options'] + "?" , '=={}== ?'.format(k['slot']))
                                    text = text.replace(l['options'] + ":" , '=={}== :'.format(k['slot']))
                                    text = text.replace(l['options'], '=={}=='.format(k['slot']))
                                elif text.find(l['options']) == len(text) - len(l['options']):
                                    text = text.replace(" " + l['options'], ' =={}=='.format(k['slot']))
                            else:
                                for s in l:
                                    if 'mc_slot' in s and 'mc_filler' in s:
                                        if isinstance(s['mc_filler'], str):
                                            text = text.replace(" " + s['mc_filler'] + " ", ' =={}== '.format(s['mc_slot']))
                                            text = text.replace(" " + s['mc_filler'] + ".", ' =={}== .'.format(s['mc_slot']))
                                            text = text.replace(" " + s['mc_filler'] + ",", ' =={}== ,'.format(s['mc_slot']))
                                            text = text.replace(" " + s['mc_filler'] + "?", ' =={}== ?'.format(s['mc_slot']))
                                            text = text.replace(" " + s['mc_filler']+ ":", ' =={}== :'.format(s['mc_slot']))
                                            if text.find(" ") == -1:
                                                text = text.replace(s['mc_filler'], '=={}=='.format(s['mc_slot']))
                                            elif text.find(s['mc_filler']) == 0:
                                                text = text.replace(s['mc_filler'] + " ", '=={}== '.format(s['mc_slot']))
                                                text = text.replace(s['mc_filler']+ ".", '=={}== .'.format(s['mc_slot']))
                                                text = text.replace(s['mc_filler'] + ",", '=={}== ,'.format(s['mc_slot']))
                                                text = text.replace(s['mc_filler'] + "?", '=={}== ?'.format(s['mc_slot']))
                                                text = text.replace(s['mc_filler'] + ":", '=={}== :'.format(s['mc_slot']))
                                                text = text.replace(s['mc_filler'], '=={}=='.format(s['mc_slot']))
                                            elif text.find(s['mc_filler']) == len(text) - len(s['mc_filler']):
                                                text = text.replace(" " + s['mc_filler'], ' =={}=='.format(s['mc_slot']))

                                        else:
                                            for o in s['mc_filler']:
                                                if 'mc_options' in o:
                                                    text = text.replace(" " + o['mc_options'] + " " , ' =={}== '.format(s['mc_slot']))
                                                    text = text.replace(" " + o['mc_options'] + "." , ' =={}== .'.format(s['mc_slot']))
                                                    text = text.replace(" " + o['mc_options'] + "," , ' =={}== ,'.format(s['mc_slot']))
                                                    text = text.replace(" " + o['mc_options'] + "?" , ' =={}== ?'.format(s['mc_slot']))
                                                    text = text.replace(" " + o['mc_options'] + ":" , ' =={}== :'.format(s['mc_slot']))
                                                    if text.find(" ") == -1:
                                                        text = text.replace(o['mc_options'], '=={}=='.format(s['mc_slot']))
                                                    elif text.find(o['mc_options']) == 0:
                                                        text = text.replace(o['mc_options'] + " " , '=={}== '.format(s['mc_slot']))
                                                        text = text.replace(o['mc_options'] + "." , '=={}== .'.format(s['mc_slot']))
                                                        text = text.replace(o['mc_options'] + "," , '=={}== ,'.format(s['mc_slot']))
                                                        text = text.replace(o['mc_options'] + "?" , '=={}== ?'.format(s['mc_slot']))
                                                        text = text.replace(o['mc_options']+ ":" , '=={}== :'.format(s['mc_slot']))
                                                        text = text.replace(o['mc_options'], '=={}=='.format(s['mc_slot']))
                                                    elif text.find(o['mc_options']) == len(text) - len(o['mc_options']):
                                                        text = text.replace(" " + o['mc_options'], ' =={}=='.format(s['mc_slot']))
                    
            text = nltk.tokenize.word_tokenize(text.lower().strip())
            total_text.append(['==SOS=='] + text + ['==EOS=='])
    data_count = len(total_text)
    intent_update = {}
    
    for intent_id, intent_string in enumerate(intent_list):
        intent_update[intent_dict[intent_string]] = intent_id
    intent_dict = {}
    for intent_id, intent_string in enumerate(intent_list):
        intent_dict[intent_string] = intent_id
    
    for tmp_intent in intent:
        for i, value in enumerate(tmp_intent):
            tmp_intent[i] = intent_update[value]
    for tmp_intent in intent:
        if not tmp_intent:
            intent_nhot.append(torch.zeros(len(intent_set)))
        else:
            intent_nhot.append(torch.zeros(len(intent_set)).scatter_(0, torch.LongTensor(tmp_intent), torch.ones(len(intent_set))))
    avg_intent = sum([torch.sum(x) for x in intent_nhot]) / data_count
    pos_weight = (len(intent_set) - avg_intent) / avg_intent
    print('Intent size:', len(intent_set))
    #print(len(total_text))
    #print(len(intent_nhot))
    torch.save(intent_dict, 'intent_dict.pkl')
    return intent_dict, total_text, intent_nhot

def build_vocab(text):
    word2index = {'==SOS==': 0, '==EOS==': 1, '==PAD==': 2, '==OOV==' : 3}
    index2word = {0: '==SOS==', 1: '==EOS==', 2: '==PAD==', 3: '==OOV=='}
    vocab_size = 4
    for i, sentence in enumerate(text):
        for j, word in enumerate(sentence):
            if word not in word2index:
                word2index[word] = vocab_size
                index2word[vocab_size] = word
                vocab_size += 1
            text[i][j] = word2index[word]
    print('Vocab size:', vocab_size)
    #print(index2word)
    text = covert_to_tensor(text, word2index)
    torch.save({
        "word2index": word2index,
        "index2word": index2word
        }, 'dict.pkl')
    return word2index, index2word, text

def covert_to_tensor(text, word2index):
    max_length = -1
    count = 0
    for i in text:
        if len(i) > max_length:
            max_length = len(i)
    for i in text:
        for j in range(max_length - len(i)):
            i.append(word2index["==EOS=="]) 

    return torch.LongTensor(text).to(device) 
'''
_, total_text = load_data('../data/movie.json')
print(total_text[2538])
a, b = build_vocab(total_text)
#print(a)
'''

