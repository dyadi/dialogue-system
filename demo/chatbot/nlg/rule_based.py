import json


class Rule_Based():
    def __init__(self, domain):
        super(Rule_Based, self).__init__()

        self.diaact_nl_pairs = {}
        self.diaact_nl_pairs['dia_acts'] = {}
        path = "../data/dia_act_nl_pairs_" + domain + ".json"
        self.load_predefine_act_nl_pairs(path)

    def find_template_sentence(self, dia_act):
        sentence = ""
        for i in range(len(dia_act)):
            template_sentence = None
            if dia_act[i]["intent"] != "request" and dia_act[i]["intent"] != "inform":
                try:
                    template_sentence = self.diaact_nl_pairs['dia_acts'][dia_act[i]["intent"]][0]['nl']['agt']
                except BaseException:
                    template_sentence = None

            else:
                
                if dia_act[i]["intent"] == "inform":
                    dia_act[i]["inform_slots"] = [dia_act[i]["slot"]]
                    dia_act[i]["request_slots"] = []
                elif dia_act[i]["intent"] == "request":
                    dia_act[i]["request_slots"] = [dia_act[i]["slot"]]
                    dia_act[i]["inform_slots"] = []

                for ele in self.diaact_nl_pairs["dia_acts"][dia_act[i]["intent"]]:
                    find_template = True
                    if len(ele["request_slots"]) == len(dia_act[i]["request_slots"]) and len(ele["inform_slots"]) == len(dia_act[i]["inform_slots"]):
                        for req in dia_act[i]["request_slots"]:
                            if ele["request_slots"].count(req) < 1:
                                find_template = False
                        for inf in dia_act[i]["inform_slots"]:
                            if ele["inform_slots"].count(inf) < 1:
                                find_template = False
                    else:
                        find_template = False
                    if find_template:
                        template_sentence = ele["nl"]["agt"]
                        break
            if template_sentence != None:
                sentence += " " + template_sentence

        return sentence

    def diaact_to_nl_slot_filling(self, dicact, template_sentence):
        """ Replace the slots with its values """
        sentence = template_sentence
        for i in range(len(dicact)):
            try:
                filler = dicact[i]["filler"]
                slot = dicact[i]["slot"]
                slot = str(slot)
                sentence = sentence.replace('$' + slot + '$', filler, 1)
            except:
                pass

        return sentence

    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """

        self.diaact_nl_pairs = json.load(open(path, 'rb'))
