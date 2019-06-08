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

    def _update_last_action(self, action):
        self.last_action[action['message_from']] = {}
        from_action = self.last_action[action['message_from']]
        for dialog_act in action['dialog_act']:
            intent = dialog_act['intent']
            if intent not in from_action:
                from_action[intent] = {}
            if 'slot' in dialog_act:
                slot = dialog_act['slot']
                if 'filler' in dialog_act:
                    from_action[intent][slot] = dialog_act['filler']
                else:
                    from_action[intent][slot] = UNK

    def update(self, action):
        if action['message_from'] == 'user':
            from_slots = self.current_slots['user']
            to_slots = self.current_slots['agent']
        elif action['message_from'] == 'agent':
            from_slots = self.current_slots['agent']
            to_slots = self.current_slots['user']
        else:
            raise KeyError(action['message_from'])

        self._update_last_action(action)

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
        self.intent_dict = {
                v: i for i, v in enumerate(self.intent_set)}
        self.slot_dict = {
                v: i for i, v in enumerate(self.slot_set)}
        self.transitive_intent_dict = {
                v: i for i, v in enumerate(self.transitive_intent_set)}
        self.reverse_intent_dict = {
                v: k for k, v in self.intent_dict.items()}
        self.reverse_slot_dict = {
                v: k for k, v in self.slot_dict.items()}
        self.reverse_transitive_intent_dict = {
                v: k for k, v in self.transitive_intent_dict.items()}

    @property
    def state_size(self):
        # TODO
        return 0

    @property
    def action_size(self):
        '''
        Only support for onehot action now(single intent, single slot).
        '''
        # TODO
        return 0

    def _encode_slots(self, current_slots):
        x = np.zeros(
                (
                    len(self.transitive_intent_dict),
                    len(self.slot_dict)
                )
            ).astype(np.float32)
        for intent, slots in current_slots.items():
            if intent not in self.transitive_intent_set:
                continue
            intent_idx = self.transitive_intent_dict[intent]
            for slot in slots.keys():
                slot_idx = self.slot_dict[slot]
                x[intent_idx][slot_idx] = 1.
        return x.reshape(-1)

    def _encode_intents(self, dialog_action):
        x = np.zeros((len(self.intent_dict)))
        for intent in dialog_action.keys():
            x[self.intent_dict[intent]] = 1.
        return x

    def _encoder_fillers(self, dialog_action):
        pass

    def encode_dialog_state(self, dialog_state):
        turn = np.array(
                dialog_state['turn'] / self.max_turn).astype(np.float32)
        user_slot = self._encode_slots(
                dialog_state['current_slots']['user'])
        agent_slot = self._encode_slots(
                dialog_state['current_slots']['agent'])
        user_action = self.encode_dialog_action(
                dialog_state['last_action']['user'],
                onehot=False)
        agent_action = self.encode_dialog_action(
                dialog_state['last_action']['agent'],
                onehot=False)
        x = np.hstack([turn, user_slot, agent_slot, user_action, agent_action])
        # TODO: encode filler
        return x

    def decode_dialog_state(self, x):
        pass

    def encode_dialog_action(self, dialog_action, onehot=True):
        if onehot:
            return self._encode_onehot_action(dialog_action)
        else:
            intents = self._encode_intents(dialog_action)
            slots = self._encode_slots(dialog_action)
            return np.hstack((intents, slots))

    def decode_dialog_action(self, y, onehot=True):
        if one_hot:
            return self._decode_onehot_action(y)
        else:
            pass


    def _encode_onehot_action(self, dialog_action):
        pass
    
    def _decode_onehot_action(self, y):
        pass
