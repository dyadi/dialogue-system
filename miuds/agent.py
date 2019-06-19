from miuds.dialogue_manager import DialogueManager
from miuds.ontologies import Ontology
from miuds.data import Dataset
import json

class Agent(object):
    def __init__(self, dataset, database, nlu=None, nlg=None):
        if isinstance(dataset, str):
            dataset = Dataset(open(dataset, 'r'))
        self.dm = DialogueManager(
                intent_set=dataset.intent_set,
                transitive_intent_set=dataset.transitive_intent_set,
                slot_set=dataset.slot_set,
                ontology=Ontology(dataset.slot_set, database))
        self.nlu = nlu
        self.nlg = nlg
        self.mode = 'text'  # text or semantic

    def text_mode(self):
        self.mode = 'text'

    def semantic_mode(self):
        self.mode = 'semantic'

    def __call__(self, user_input):
        if self.mode == 'text':
            user_action = self.nlu(user_input)
        elif self.mode == 'semantic':
            user_action = json.loads(user_input)
        else:
            raise KeyError('Invalid mode: {}'.format(self.mode))

        agent_action = self.dm(user_action)

        if self.mode == 'text':
            output = self.nlg(agent_action)
        else:
            output = json.dumps(agent_action)

        return output
