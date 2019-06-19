import json
from collections import namedtuple

DialogData = namedtuple(
    'DialogData',
    (
        'session_id',
        'message_id',
        'message_from',
        'text',
        'dialog_act'
    )
)


class Dataset(object):

    def __init__(self, file):
        self.data = json.load(file)
        self.intent_set = set()
        self.slot_set = set()
        self.transitive_intent_set = set()
        self.dialog_data = []
        for session in self.data:
            for message in session['message']:
                self.dialog_data.append(
                    DialogData(
                        session['session_id'],
                        message['id'],
                        message['from'],
                        message['text'],
                        message['dialog_act']
                    )
                )
                for action in message['dialog_act']:
                    self.intent_set.add(action['intent'])
                    if 'slot' in action:
                        self.slot_set.add(action['slot'])
                        self.transitive_intent_set.add(action['intent'])
