import miuds
from .policy import Policy
import random

# TODO: Replace hardcoding necessary_slots with configurable setting
necessary_slots = [
    'moviename',
    'theater',
    'date',
    'starttime',
    'numberofpeople']


class RulePolicy(Policy):
    def __init__(self, intent_set, transitive_intent_set, slot_set):
        super(RulePolicy, self).__init__(
                intent_set,
                transitive_intent_set,
                slot_set)

    def sample_action(self):
        """Sample a random action from action sapce"""
        action = {}
        intent = random.sample(self.intent_set, 1)[0]
        action['intent'] = intent
        if intent in self.transitive_intent_set:
            action['slot'] = random.sample(self.slot_set, 1)[0]
        return [action]

    def _is_complete(self, state):
        """Check if all necessary slot is filled"""
        user_informed = state['current_slots']['user']['inform']
        agent_informed = state['current_slots']['agent']['inform']
        for slot in necessary_slots:
            if slot not in user_informed and slot not in agent_informed:
                return False
        return True

    def _response_request(self, state):
        avail_filler = state['avail_filler']
        user_requested = state['current_slots']['user']['request']
        agent_informed = state['current_slots']['agent']['inform']

        for slot in user_requested:
            if slot in avail_filler:
                return [{
                    'intent': 'inform',
                    'slot': slot,
                    'filler': avail_filler[slot]
                    }]

        # No availalbe result
        # TODO: Replace with more suitable intent/slot
        return [{'intent': 'inform', 'slot': 'result', 'filler': '{}'}]

    def _request_slot(self, state):
        user_informed = state['current_slots']['user']['inform']
        agent_requested = state['current_slots']['agent']['request']
        for slot in necessary_slots:
            if slot not in agent_requested or slot not in user_informed:
                return [{'intent': 'request', 'slot': slot}]
        # Out of rules
        return [{'intent': 'deny'}]

    def make_action(self, state):
        last_user_action = state['last_action']['user']

        # Highest priority to response user request
        if 'request' in last_user_action:
            return self._response_request(state)

        # Fill all necessary slots
        if not self._is_complete(state):
            return self._request_slot(state)
        else:
            # NOTE: 'taskcomplete' should be a intent ???
            return [{'intent': 'inform', 'slot': 'taskcomplete'}]

        # TODO: Bellow conditions may not happen ??
        '''
        if 'closing' in last_user_action:
            return [{'intent': 'closing'}]
        if 'greeting' in last_user_action:
            return [{'intent': 'greeting'}]
        '''

        # Out of rules
        return [{'intent': 'deny'}]
