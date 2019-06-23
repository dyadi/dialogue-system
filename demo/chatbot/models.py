from django.db import models
from django.conf import settings
from .nlu import nlu as NLU
from .nlg import NLGPredict as NLG
import json
import miuds

nlu = NLU()
nlu.load_nlu_model(settings.NLU_MODEL_PATH)
nlg = NLG(settings.NLG_MODEL_DIR)
agent = miuds.Agent(settings.DATASET, settings.AGENT_DB)
agent.semantic_mode()

def parse_user_action(user_action):
    dialog_action = []
    intent = user_action['diaact']
    for slot, filler in user_action['inform_slots'].items():
        dialog_action.append({
            'intent': intent,
            'slot': slot,
            'filler': filler})
    for slot, filler in user_action['request_slots'].items():
        dialog_action.append({
            'intent': intent,
            'slot': slot,
            'filler': filler})
    return json.dumps(dialog_action)

def text_to_semantic(text):
    return nlu.generate_dia_act(text)

def semantic_to_text(semantic_frame):
    return nlg.predict(semantic_frame)

def reset():
    agent.dm.state_tracker.initial_episode()

def agent_response(user_text):
    user_action = parse_user_action(text_to_semantic(user_text))
    print(user_action)
    agent_action = agent(user_action)
    print(agent_action)
    agent_text = semantic_to_text(json.loads(agent_action))
    return {
            'user_text': user_text,
            'agent_text': agent_text,
            'user_action': json.dumps(user_action, indent=2).replace('\\"', ''),
            'agent_action': json.dumps(agent_action, indent=2).replace('\\"', '')
            }
