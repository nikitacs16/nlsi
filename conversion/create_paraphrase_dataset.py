import json
from collections import defaultdict
import random
import os
import copy
from nlsi.interpret.utils import get_domain_kv
random.seed(42)


#creating the paraphrase dictionary where every templated instruction is matched with three paraphrases
fname = ['conversion/si_paraphrase/pp_v2_val.predictions.jsonl', 'conversion/si_paraphrase/pp_v3_val.predictions.jsonl', 'conversion/si_paraphrase/pp_v4_val.predictions.jsonl']
og_fname = json.load(open('conversion/base_paraphrase.json'))

def read_jsonl(fname):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(json.loads(line))
    return data


paraphrase_dict = defaultdict(list)
for f in fname:
    data = read_jsonl(f) 
    assert len(data) == len(og_fname)   
    for k in range(len(data)):
        for i in range(len(data[k]['applicable_standing_instructions'])): 
            paraphrase_dict[og_fname[k]['standing_instructions'][i]].append(data[k]['applicable_standing_instructions'][i]['nl_instruction'])
            paraphrase_dict[og_fname[k]['standing_instructions'][i].replace("preferance", "preference")].append(data[k]['applicable_standing_instructions'][i]['nl_instruction'])
            paraphrase_dict[og_fname[k]['standing_instructions'][i].replace("preferrance", "preference")].append(data[k]['applicable_standing_instructions'][i]['nl_instruction'])
            
for k in paraphrase_dict.keys():
    paraphrase_dict[k] = list(set(paraphrase_dict[k]))


#replacing the templates with the paraphrases per split
sentences = []
for split in ["train", "val", "test"]:
    data = json.load(open(f"conversion/template/{split}.json"))
    w = []
    for i in range(len(data)):  
        og_instructions = copy.deepcopy(data[i]['standing_instructions'])
        
        for j in range(len(data[i]['standing_instructions'])):
            
            if data[i]['standing_instructions'][j] in paraphrase_dict:
                data[i]['standing_instructions'][j] = random.choice(paraphrase_dict[data[i]['standing_instructions'][j]])
            else:
                sentences.append(data[i]['standing_instructions'][j])
        for j in range(len(data[i]['user_profile'])):    
            if data[i]["user_profile"][j] in og_instructions:
                idx = og_instructions.index(data[i]["user_profile"][j])
                data[i]["user_profile"][j] = data[i]["standing_instructions"][idx]
            else:
                if data[i]["user_profile"][j] in paraphrase_dict:
                    data[i]["user_profile"][j] = random.choice(paraphrase_dict[data[i]["user_profile"][j]])
        for j in data[i]["standing_instructions"]:
            try:
                assert j in data[i]["user_profile"]
            except:
                
                print(j)
                print(data[i]["user_profile"])
                print(data[i]["standing_instructions"])
                print()
        w = w + data[i]['standing_instructions']    

    os.makedirs(f'conversion/paraphrase', exist_ok=True)
    with open(f'conversion/paraphrase/{split}.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
    

#fixing some mistakes from prompts
        
c = 0
mapper = {'comic':'funny', 'NY': 'New York', 'NYC': 'New York City',  'SF': 'San Fransico', 'SD': 'San Diego', 'Philly': 'Philadelphia', 'Chi-town': 'Chicago', 
          'LA': 'Los Angeles', 'Biographical': 'biopic', 'KL': 'Kuala Lumpur' , 'phl international airport': 'Philadelphia International Airport'}
for split in ["train", "val", "test"]:
    data = json.load(open(f"conversion/paraphrase/{split}.json"))
    c = 0
    for example in data:
        all_si = "-".join(example['standing_instructions']).lower()
        flag = False
        new_api_calls = []
        for api in example['api_calls']:
            if "Denists" in api:
                api = api.replace("Denists", "Dentists")
            domain, kv = get_domain_kv(api)

            flag = False
            for k,v in kv.items():
                
                if v!= "True":
                    if v == "?":
                        continue
                    if v.lower() not in example['utterance'].lower() and v.lower() not in all_si:
                        if v.isdigit():
                            continue
                        if k in ['account_type', 'recipient_account_type', 'fare_type', 'category', 'event_type', 'seating_class', 'refundable', 'airlines', 'show_type', 'playback_device', 'type', 'car_type', 'price_range', 'ride_type']:
                            continue
                        if v in mapper:
                            new_api = api.replace(v, mapper[v])                
                            flag = True
            if flag:
                new_api_calls.append(new_api)
            else:
                new_api_calls.append(api)
     
        example['api_calls'] = new_api_calls
        example['example_id'] = split + '_' + str(c).zfill(4)
        c = c + 1
    with open(f'conversion/paraphrase/{split}.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
        print("added")