import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

bert_models = None
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

def get_bert_feature(text, word2ph, device, language):
    global bert_models
    if language != "zh":
        return torch.zeros(1024, sum(word2ph))

    if bert_models == None:
        bert_models = AutoModelForMaskedLM.from_pretrained(
            "hfl/chinese-roberta-wwm-ext-large"
        ).to(device)
        print('loaded bert model at rank', device)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_models(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

