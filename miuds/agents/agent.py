import miuds.policies


class Agent(object):
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 policy='DQNPolicy', policy_args={}, ontology):
        self.intent_set = intent_set
        self.transitive_intent_set = transitive_intent_set
        self.slot_set = slot_set
        if isinstance(policy, str):
            self.policy = getattr(miuds.policies, policy)(**policy_args)
        elif isinstance(policy, miuds.policies.Policy):
            self.policy = policy(**policy_args)
        else:
            raise TypeError(
                    'policy must be a subclass of miuds.policies.Policy.')
        self.ontology = ontology
        self.state_tracker = NeuralStateTracker(
                intent_set=intent_set,
                transitive_intent_set=transitive_intent_set,
                slot_set=slot_set)

    def __call__(self, dialog_action):
        
        self.state_tracker.update(dialog_action)
        dialog_state = self.state_tracker.get_dialog_state()

        # NOTE: This not work now
        dialog_state = self.ontology.retrieve(dialog_state)
        
        # Encode dialog state for neural network
        # TODO: feasible for rule-based policy
        state = self.state_tracker.encode_dialog_state(dialog_state)
        
        action = self.policy(state)
        
        # Decode neural network output
        # TODO: feasible for rule-based policy
        agent_action = self.state_tracker.decode_dialog_action(action)

        self.state_tracker.update(agent_action)
        return agent_action


