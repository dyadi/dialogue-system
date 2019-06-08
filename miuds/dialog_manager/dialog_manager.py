class DialogueManager(object):
    def __init__(self, agent, user, action_set, transitive_action_set, 
                 slot_set):
        self.agent = agent
        self.user = user
        self.action_set = action_set
        self.transitive_action_set = transitive_action_set
        self.slot_set = slot_set
        self.nlu = nlu
        self.nlg = nlg
        self.mode = 'text' # text or semantic

    def text_mode(self):
        self.mode = 'text'

    def semantic_mode(self):
        self.mode = 'semantic'
    
    def __call__(self, user_input):
        if self.mode == 'text':
            user_action = self.nlu(user_input)
        else if self.mode == 'semantic':
            user_action = user_input
        else:
            raise KeyError('Invalid mode: {}'.format(self.mode))

        agent_action = self.agent(user_action)

        if self.mode == 'text':
            output = self.nlg(agent_action)
        else:
            output = agent_action

        return output 

