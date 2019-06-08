import json
import numpy as np
import copy

UNK = '<UNK>'

class StateTracker(object):   
    def __init__(self, intent_set, transitive_intent_set, slot_set, max_turn):
        self.intent_set = intent_set
        self.transitive_intent_set = transitive_intent_set
        self.slot_set = slot_set
        self.max_turn = max_turn
        self.initial_epsidoe()

    def initial_epsidoe(self):
        self.turn = 0
        self.current_slots = {}
        self.current_slots['user'] = {
                intent: {} for intent in self.transitive_intent_set}
        self.current_slots['agent'] = {
                intent: {} for intent in self.transitive_intent_set}
        self.last_action = {'user': {}, 'agent': {}}

    def update(self, action):
        if action['message_from'] == 'user':
            from_slots = self.current_slots['user']
            to_slots = self.current_slots['agent']
            self.last_action['user'] = copy.deepcopy(action['dialog_act'])
        elif action['message_from'] == 'agent':
            from_slots = self.current_slots['agent']
            to_slots = self.current_slots['user']
            self.last_action['agent'] = copy.deepcopy(action['dialog_act'])
        else:
            raise KeyError(action['message_from'])
        
        for dialog_act in action['dialog_act']:
            if 'slot' in dialog_act:
                if 'filler' in dialog_act:
                    slot_val = dialog_act['filler']
                else:
                    slot_val = UNK
                from_slots[dialog_act['intent']][dialog_act['slot']] = slot_val
                # TODO remove requested to_slots
                # to_slots[dialog_act['intent']][dialog_act['slot']] = slot_val

        self.turn += 1

    def get_dialog_state(self):
        dialog_state = {}
        dialog_state['turn'] = self.turn
        dialog_state['current_slots'] = copy.deepcopy(self.current_slots)
        dialog_state['last_action'] = copy.deepcopy(self.last_action)
        return dialog_state

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += '\n'
        repr_str += 'turn: {}\n'.format(self.turn)
        repr_str += 'slots: {}\n'.format(
                json.dumps(self.current_slots, indent=2))
        repr_str += 'last_action: {}\n'.format(
                json.dumps(self.last_action, indent=2))
        return repr_str


class NeuralStateTracker(StateTracker):
    def __init__(self, *args, **kwargs):
        super(NeuralStateTracker, self).__init__(*args, **kwargs)
        self.reverse_intent_dict = {v:i for i, v in enumerate(self.intent_set)}
        self.reverse_slot_dict = {v:i for i, v in enumerate(self.slot_set)}
        self.reverse_transitive_intent_dict = {
                v:i for i, v in enumerate(self.transitive_intent_set)}

    def encode_dialog_state(self, dialog_state):
        x = []
        x += [dialog_state['turn'] / self.max_turn]
        for intent in self.transitive_intent_set:
            x += [1 if slot in dialog_state['current_slots']['user'][intent] 
                    else 0 for slot in self.slot_set]
            x += [1 if slot in dialog_state['current_slots']['agent'][intent] 
                    else 0 for slot in self.slot_set]
        x += [1 if intent in dialog_state['last_action']['user'] 
                else 0 for intent in self.intent_set]
        x += [1 if intent in dialog_state['last_action']['agent'] 
                else 0 for intent in self.intent_set]
        # TODO: encode filler
        # TODO: encode slot of last_action
        return np.array(x).astype(np.float32)

    def decode_dialog_state(self, x):
        pass

    def encode_dialog_action(self, dialog_action):
        pass
    
    def decode_dialog_action(self, y):
        pass
