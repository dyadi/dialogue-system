from django.db import models
from django.conf import settings
from .nlu import nlu as NLU
from .nlg import NLGPredict as NLG
import json
import miuds
from miuds.data import Dataset

dataset = Dataset(open(settings.DATASET))
nlu = NLU()
nlu.load_nlu_model(settings.NLU_MODEL_PATH)
nlg = NLG(settings.NLG_MODEL_DIR)
agent_manager = {}

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

def reset_agent(sid):
    if sid in agent_manager:
        agent_manager[sid].dm.state_tracker.initial_episode()
    else:
        agent_manager[sid] = miuds.Agent(dataset, settings.AGENT_DB)
        agent_manager[sid].semantic_mode()

def get_timetable(sid):
    df = agent_manager[sid].dm.state_tracker.ontology.retrieve({}).df
    timetable = df.to_dict()
    ret = {'keys': timetable.keys(), 'values': []}
    for i in range(len(df)):
        tmp = {}
        for key in timetable.keys():
            tmp[key] = timetable[key][i]
        ret['values'].append(tmp)
    return ret

def agent_response(user_text, sid):
    user_action = parse_user_action(text_to_semantic(user_text))
    agent_action = agent_manager[sid](user_action)
    agent_text = semantic_to_text(json.loads(agent_action))
    return {
            'user_text': user_text,
            'agent_text': agent_text,
            'user_action': json.dumps(user_action, indent=2).strip('"').replace('\\"', ''),
            'agent_action': json.dumps(agent_action, indent=2).strip('"').replace('\\"', '')
            }
