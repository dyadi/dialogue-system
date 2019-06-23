from .model_based import ModelBasedNLG


class NLGPredict(object):
    def __init__(self, model_dir, mode="model", domain="movie"):
        """  choose NLG mode: rule-based, model, mix, and choose NLG domain: movie, restaurant, taxi """
        super(NLGPredict, self).__init__()

        self.mode = mode
        self.domain = domain
        self.model = ModelBasedNLG(model_dir)

    def predict(self, dialog_act):
        sentence = ""
        domain = self.domain
        if self.mode == "model":
            sentence = self.model.predict(dialog_act)
        
        elif self.mode == "rule-based":
            from rule_based import Rule_Based
            RB = Rule_Based(domain)
            template_sentence = RB.find_template_sentence(dialog_act)
            sentence = RB.diaact_to_nl_slot_filling(dialog_act, template_sentence)
        
        elif self.mode == "mix":
            pass

        return sentence

if __name__ == '__main__':
    dicact = [
          {
            "intent": "inform",
            "slot": "moviename",
            "filler": "The Revenant"
          },
          {
            "intent": "inform",
            "slot": "date",
            "filler": "tomorrow"
          },
          {
            "intent": "inform",
            "slot": "starttime",
            "filler": "8pm"
          },
          {
            "intent": "inform",
            "slot": "theater",
            "filler": "Regal MacArthur Marketplace Stadium 16"
          },
          {
            "intent": "confirm_question"
          }
        ]
    nlg_predict = NLGPredict(mode='model')
    sentence = nlg_predict.predict(dicact)
    print(sentence)
