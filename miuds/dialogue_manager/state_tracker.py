import json
import numpy as np
import copy

UNK = 'UNK'


class StateTracker(object):
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 ontology, max_turn=10):
        self.intent_set = intent_set
        self.transitive_intent_set = transitive_intent_set
        self.slot_set = slot_set
        self.ontology = ontology
        self.max_turn = max_turn
        self.initial_episode()

    def initial_episode(self):
        self.turn = 0
        self.current_slots = {}
        self.current_slots['user'] = {'inform': {}, 'request': {}}
        self.current_slots['agent'] = {'inform': {}, 'request': {}}
        self.last_action = {'user': {}, 'agent': {}}
        self.avail_filler = {}

    def _update_last_action(self, action, message_from):
        self.last_action[message_from] = {}
        from_action = self.last_action[message_from]
        for dialog_act in action:
            intent = dialog_act['intent']
            if intent not in from_action:
                from_action[intent] = {}
            if 'slot' in dialog_act:
                slot = dialog_act['slot']
                if 'filler' in dialog_act:
                    from_action[intent][slot] = dialog_act['filler']
                else:
                    from_action[intent][slot] = UNK

    def update(self, action, message_from):
        if message_from == 'user':
            from_slots = self.current_slots['user']
            to_slots = self.current_slots['agent']
        elif message_from == 'agent':
            from_slots = self.current_slots['agent']
            to_slots = self.current_slots['user']
        else:
            raise KeyError(message_from)

        self._update_last_action(action, message_from)

        for dialog_act in action:
            intent = dialog_act['intent']
            if 'slot' in dialog_act:
                if 'filler' in dialog_act:
                    slot_val = dialog_act['filler']
                else:
                    slot_val = UNK
                if intent not in from_slots:
                    from_slots[intent] = {}
                from_slots[intent][dialog_act['slot']] = slot_val
                # TODO remove requested to_slots

        self.avail_filler = self.retrieve_avail_filler()

        self.turn += 1

    def retrieve_avail_filler(self):
        constraint = {}
        # User slots have higher priority
        for message_from in ['agent', 'user']:
            for intent, slots in self.current_slots[message_from].items():
                for slot, slot_val in slots.items():
                    if slot_val != UNK:
                        constraint[slot] = slot_val
        return self.ontology.retrieve(constraint)

    def get_dialog_state(self):
        dialog_state = {}
        dialog_state['turn'] = self.turn
        dialog_state['current_slots'] = copy.deepcopy(self.current_slots)
        dialog_state['last_action'] = copy.deepcopy(self.last_action)
        dialog_state['avail_filler'] = copy.deepcopy(self.avail_filler)
        return dialog_state

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += '\n'
        repr_str += 'turn: {}\n'.format(self.turn)
        repr_str += 'avail_filler: {}\n'.format(len(self.avail_filler))
        repr_str += 'slots:\n'
        repr_str += json.dumps(self.current_slots, indent=2)
        repr_str += '\nlast_action:\n'
        repr_str += json.dumps(self.last_action, indent=2)
        return repr_str
