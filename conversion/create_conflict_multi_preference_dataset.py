import json
import numpy as np  
import random
import itertools
import pandas as pd
from collections import defaultdict
import copy
import re
from nlsi.interpret.utils import get_domain_kv
#from conversion.create_template_dataset import get_api_calls, update_api_calls

domain_dict = pd.read_csv("conversion/domain-value.tsv", sep='\t')
domain_dict = {k:v for k,v in zip(domain_dict["domain"], domain_dict["value"])}

instructions_dataset = json.load(open('conversion/dataset_instructions_apis_v1.json'))
slot_value_dict = defaultdict(lambda: defaultdict(set))
for example in instructions_dataset:
    if len(example['services']) > 1:
        continue
    service = example['services'][0].split('_')[0]
    for slot_name, slot_value in zip(example['frames']["state"][0]["slot_values"]["slot_name"], example['frames']["state"][0]["slot_values"]["slot_value_list"]):
        if slot_value[0] == "dontcare" or slot_value[0] == "?":
            continue
        slot_value_dict[service][slot_name].add(slot_value[0])

global new_dict
new_dict = {}
for service in slot_value_dict:
    new_dict[service] = {}
    for k,v in slot_value_dict[service].items():
        new_dict[service][k] = list(v)

global hierarchy
dataset_schema = json.load(open("conversion/schema_updated_v5.json")) #This was created manually
schema = {k["service_name"]:k for k in dataset_schema}
hierarchy = {}
for k in schema:
    hierarchy[k] = {}
    for id in range(len(schema[k]["hierarchy"])):
        hierarchy[k][schema[k]["slots"][id]["name"]] = schema[k]["hierarchy"][id] 


global domain_to_api 
domain_to_api = {}
for i in schema:
    domain_to_api[i] = "Get" + i.split('_')[0] 

domain_to_api["Hotels_2"] = "GetHouseStays"
domain_to_api["Services_1"] = "GetSalons"
domain_to_api["Services_2"] = "GetDentists"
domain_to_api["Services_3"] = "GetDoctors"

global intent_dict
def get_intent_dict(schema):
    intent_dict = {}
    for domain in schema:
        intent_dict[domain] = {}
        intent_is_instruction = {}
        slot_to_intent = {}
        all_slots = []
        for intent in schema[domain]['intents']:
            intent_is_instruction[intent['name']] = intent['is_instruction']
            all_slots = all_slots + list(set(list(intent["required_slots"]) + list(intent["optional_slots"]) + list(intent["result_slots"])))
            for slot in all_slots:
                if slot not in slot_to_intent:
                    slot_to_intent[slot] = []
                slot_to_intent[slot].append(intent['name'])
        intent_dict[domain]["intent_is_instruction"] = intent_is_instruction
        intent_dict[domain]["slot_to_intent"] = slot_to_intent
        intent_dict[domain]["slots"] = list(set(all_slots))
    return intent_dict

intent_dict = get_intent_dict(schema)


def get_api_calls(domains, slot_names, slot_values):
    
    api_calls_dict = defaultdict(list)
    #print(slot_names, slot_values)
    for domain in domains:
        #Every api call is saved as #({"slot_name":"slot_value"})
        for slot_name, slot_value in zip(slot_names, slot_values):
            if slot_name in intent_dict[domain]["slots"]:
                api_calls_dict[domain].append({slot_name: slot_value})             
    api_calls = []
    for domain in api_calls_dict:
        s = domain_to_api[domain] + "("
        for i in api_calls_dict[domain]:
            for k,v in i.items():
                if v=="dontcare":
                    v = "any"
                s = s + k + "=\"" + v + "\", "
        s = s[:-2] + ")"
        api_calls.append(s)
    return api_calls




def get_slot_value_list(example):
    slot_names = []
    slot_values = []
    for api_calls in example["api_calls"]:
        domain, key_value_pairs = get_domain_kv(api_calls)
        for k,v in key_value_pairs.items():
            slot_names.append(k)
            slot_values.append(v)
    return slot_names, slot_values

def update_api_calls(domains, old_slot_names, old_slot_values, new_slot_names, new_slot_values):
    new_api_calls = []
    #find domain where slot_name is old_slot_name
    slot_name_to_domain = {}
    for slot_name in new_slot_names:
        for domain in domains:
            if slot_name in intent_dict[domain]["slots"]:
                slot_name_to_domain[slot_name] = domain
            

    #get other slot names in the same domain
    domain_to_slot_names = defaultdict(list)
    domain_to_slot_names_hierarchy = defaultdict(list)
    
    for slot_name in old_slot_names:
        for domain in domains:
            if slot_name in intent_dict[domain]["slots"]:
                domain_to_slot_names[domain].append(slot_name)
                domain_to_slot_names_hierarchy[domain].append(hierarchy[domain][slot_name])
    
    for slot_name, slot_value in zip(new_slot_names, new_slot_values):
        sn = []
        sv = []
        
        curr_domain = slot_name_to_domain[slot_name]
        curr_hierarchy = hierarchy[curr_domain][slot_name]
        for slot_name_old, slot_value_old in zip(old_slot_names, old_slot_values):
            if slot_name_old in domain_to_slot_names[curr_domain]:
                if hierarchy[curr_domain][slot_name_old] == -1 or hierarchy[curr_domain][slot_name_old] == -10 or hierarchy[curr_domain][slot_name_old] >= curr_hierarchy:                
                    sn.append(slot_name_old)
                    if slot_name_old == slot_name:
                        sv.append(slot_value)
                    else:
                        sv.append(slot_value_old)
        new_api_calls  = new_api_calls + get_api_calls([curr_domain], sn, sv)
   
    return new_api_calls


def multiple_instructions(example):
    if len(example["services"]) > 2:
        return example, None    

    random.seed(len(example["utterance"]))
    remapper = {"HouseStays": "Hotels", "Doctors": "Services", "Dentists": "Services", "Salons": "Services", "Salon": "Services", "Dentist": "Services", "Doctor": "Services", "HouseStay": "Hotels" }
 
    new_example = copy.deepcopy(example)
    slot_names, slot_values = get_slot_value_list(example)
    valid_slot_names = ['number_stops', 'star_rating',  'dentist_name', 'average_rating', 'doctor_name', 'show_type', 'genre',  'theater_name', 'number_of_beds', 'venue', 'category', 'cuisine', 'type', 'event_type', 'number_of_baths', 'ride_type', 'car_type', 'artist', 'car_name', 'stylist_name', 'directed_by', 'rating', 'price_range', 'seating_class', 'airlines', 'hotel_name']
    keep_domains = {}
    keep_instructions = defaultdict(list)
    keep_values = defaultdict()
    c = 0
    all_slot_names = slot_names #[s for s in example["frames"]['state'][0]["slot_values"]["slot_name"]]
    flag = False
    for slot_name, slot_value in zip(slot_names, slot_values):
        keep_values[slot_name] = slot_value
        if slot_name not in valid_slot_names:
            continue
        split_slot_name = slot_name.replace("_", " ")   
        for instruction, label in zip(example["standing_instructions"], example["standing_instructions_labels"]):
                if label == "domain_conditional" or label == "simple_default":       
                    if split_slot_name in instruction:
                        keep_instructions[slot_name].append(instruction)
                        m = re.match(r"If I ask for (.+), my*", instruction)
                        if m:
                            domain = m.group(1)
                            keep_domains[slot_name] = domain.strip()
                            break
    
    if len(keep_instructions.keys()) == 0:
        return example, None
    if len(keep_instructions.keys()) == 1:
        chosen_idx = 0

    if len(keep_instructions.keys()) > 1:
        chosen_idx = random.randint(0, len(keep_instructions.keys())-1)
    
    slot_name = list(keep_instructions.keys())[chosen_idx]

    instructions = keep_instructions[slot_name]
    domain = keep_domains[slot_name]
    #weird combinations
    if domain.lower() == "doctor" and slot_name == "type":
        return example, None
    
    if domain.strip() in remapper:
        domain = remapper[domain.strip()]
    slot_value = keep_values[slot_name]

    # choosing a new slot value 
    if domain not in new_dict:
        return example, None
    while True:
        new_slot_value = random.sample(new_dict[domain][slot_name], 1)[0]
        if new_slot_value.lower() != slot_value.lower():
            break
    
    
    
    new_instructions = []
    new_instructions_labels = []
    updated_values = defaultdict(list)
    replace_instructions = {}
    for instruction in instructions:
        flag = False
        #complicated cases for the new slot are not used during updating. Especially when old slot is used for chains
        for slot_name2 in all_slot_names:
            if slot_name2.replace("_", " ") in instruction and slot_name2 != slot_name:
                flag = True
                break
        if flag:
            continue
        new_instruction = instruction.replace(slot_value, new_slot_value)
        if new_slot_value in new_instruction:
            updated_values[slot_name].append(new_slot_value)
            #coin toss to replace old instruction with "or"
            if random.random() > 0.4:
                new_instructions.append(new_instruction)
                new_instructions_labels.append("domain_conditional")
            else:
                replace_instructions[instruction] = instruction + " or " + new_slot_value
            break
    


    new_slot_names = []
    new_slot_values = []
    for slot_name in updated_values.keys():
        for sv in updated_values[slot_name]:
            if slot_name not in new_slot_names:
                new_slot_names.append(slot_name)
                new_slot_values.append(sv)
    try:
        new_api_calls = update_api_calls(example["services"], slot_names, slot_values,  new_slot_names, new_slot_values)
    except:
        return example, None    

            
    new_example["standing_instructions"] = new_example["standing_instructions"] + new_instructions
    new_example["standing_instructions_labels"] = new_example["standing_instructions_labels"] + new_instructions_labels
    new_example["user_profile"] = example["user_profile"] + new_instructions
    new_example["api_calls"] = example["api_calls"] + new_api_calls
   
    #updating instructions with or
    for idx, instruction in enumerate(new_example["standing_instructions"]):
        if instruction in replace_instructions:
            new_example["standing_instructions"][idx] = replace_instructions[instruction]
    
    for idx, instruction in enumerate(new_example["user_profile"]):
        if instruction in replace_instructions:
            new_example["user_profile"][idx] = replace_instructions[instruction]

    return new_example, updated_values         





#conflict
c = len(instructions_dataset)
new_examples = []
conflict_data = json.load(open('conversion/override_paraphrase.json')) 
for example in conflict_data:
    example["example_id"] = c
    c = c + 1
    new_examples.append(example)

#multipreference
v1_data = json.load(open('conversion/dataset_instructions_apis_user_profiles_v1.json'))
new_c = 0
all_slots = []
slot_count = defaultdict(int)
for example in v1_data:
    if len(example["services"]) > 2:
        continue #exclude multi domains more than two
    new_example, updated_values = multiple_instructions(example)
    if new_example["user_profile"]!= example["user_profile"]:   
        for k in updated_values:
            slot_count[k] += 1
            if slot_count[k] < 50:
                new_example["example_type"] = "multiple_instructions"
                new_example["example_id"] = c
                new_examples.append(new_example)
                c += 1
                new_c = new_c + 1
print(new_c)

with open('conversion/dataset_instructions_apis_user_profiles_v2.json', 'w') as f:
    json.dump(new_examples, indent=4, fp=f)
