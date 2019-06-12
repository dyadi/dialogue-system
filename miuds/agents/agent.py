import miuds.policies
from .state_tracker import StateTracker


class Agent(object):
    '''Dialog agent
    The agent interact with user. Take the semantic user_action as input and
    output the agent_action.
    '''
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 ontology, policy='DQNPolicy', policy_args={}):
        self.state_tracker = StateTracker(
                intent_set=intent_set,
                transitive_intent_set=transitive_intent_set,
                slot_set=slot_set,
                ontology=ontology)
        policy_args['intent_set'] = intent_set
        policy_args['transitive_intent_set'] = transitive_intent_set
        policy_args['slot_set'] = slot_set
        if isinstance(policy, str):
            self.policy = getattr(miuds.policies, policy)(**policy_args)
        elif issubclass(policy, miuds.policies.Policy):
            self.policy = policy(**policy_args)
        else:
            raise TypeError(
                    'policy must be a subclass of miuds.policies.Policy.')

    def set_policy(self, policy):
        self.policy = policy

    def __call__(self, user_action):
        self.state_tracker.update(user_action, message_from='user')
        state = self.state_tracker.get_dialog_state()
        agent_action = self.policy.make_action(state)
        self.state_tracker.update(agent_action, message_from='agent')
        return agent_action
