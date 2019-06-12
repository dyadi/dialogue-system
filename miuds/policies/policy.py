from abc import ABCMeta, abstractmethod
import numpy as np


class Policy(metaclass=ABCMeta):
    '''
    Policy interface
    '''
    def __init__(self, intent_set, transitive_intent_set, slot_set):
        self.intent_set = intent_set
        self.transitive_intent_set = transitive_intent_set
        self.slot_set = slot_set

    @abstractmethod
    def make_action(self, state):
        pass


class RLPolicy(Policy, metaclass=ABCMeta):
    '''
    Reinforcement Learning Policy interface
    '''
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 max_turn=10):
        super(RLPolicy, self).__init__(
                intent_set, transitive_intent_set, slot_set)
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
        self.onehot_action_set = []
        for intent in self.intent_set:
            if intent in self.transitive_intent_set:
                for slot in self.slot_set:
                    self.onehot_action_set.append({
                        'intent': intent,
                        'slot': slot})
            else:
                self.onehot_action_set.append({'intent': intent})
        self.max_turn = max_turn
        self.train_mode = False

    @property
    def input_size(self):
        '''
        turn = 1
        slots = transitive_intent * slot
        action = intent + slots
        total = turn + user_slot + agent_slot + user_action + agent_action
              = 1 + 2 * slots + 2 * action
              = 1 + 2 * slots + 2 * (intent * slots)
              = 1 + 4 * slots + 2 * intent
        '''
        return (1
                + 4 * len(self.transitive_intent_set) * len(self.slot_set)
                + 2 * len(self.intent_set))

    @property
    def output_size(self):
        '''
        Only support for onehot action now(single intent, single slot).
        total = transitive_intent * slots + intent
        '''
        return len(self.onehot_action_set)

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
        x = np.zeros((len(self.intent_dict))).astype(np.float32)
        for intent in dialog_action.keys():
            x[self.intent_dict[intent]] = 1.
        return x

    def _encoder_fillers(self, dialog_action):
        raise NotImplementedError()

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

    def encode_dialog_action(self, dialog_action, onehot=True):
        if onehot:
            return self._encode_onehot_action(dialog_action)
        else:
            intents = self._encode_intents(dialog_action)
            slots = self._encode_slots(dialog_action)
            return np.hstack((intents, slots))

    def decode_dialog_action(self, y, onehot=True):
        if onehot:
            return [self._decode_onehot_action(y)]
        else:
            raise NotImplementedError()

    def _decode_onehot_action(self, y):
        return self.onehot_action_set[y]

    def set_train_mode(self):
        self.train_mode = True

    @abstractmethod
    def train_episode(self, *args, **kwargs):
        pass
