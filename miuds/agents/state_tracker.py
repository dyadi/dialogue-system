import json
import numpy as np
import copy

UNK = '<UNK>'


class StateTracker(object):
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 ontology, max_turn=10):
        self.intent_set = intent_set
        self.transitive_intent_set = transitive_intent_set
        self.slot_set = slot_set
        self.ontology = ontology
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
            if 'slot' in dialog_act:
                if 'filler' in dialog_act:
                    slot_val = dialog_act['filler']
                else:
                    slot_val = UNK
                from_slots[dialog_act['intent']][dialog_act['slot']] = slot_val
                # TODO remove requested to_slots
                # to_slots[dialog_act['intent']][dialog_act['slot']] = slot_val

        # NOTE: Ontology not work now
        self.avail_filler = self.ontology.retrieve(self.current_slots)

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
