from datasets import load_dataset
import json
import numpy as np  
import random
import itertools
import pandas as pd
from collections import defaultdict
import re
import itertools
import copy
from tqdm import tqdm
from nlsi.interpret.utils import get_domain_kv
random.seed(42)


  

#intial loading of the dataset
domain_dict = pd.read_csv("conversion/domain-value.tsv", sep='\t') #This was created manually
domain_dict = {k:v for k,v in zip(domain_dict["domain"], domain_dict["value"])}
dataset = load_dataset("schema_guided_dstc8",split="train")
dataset_schema = json.load(open("conversion/schema_updated_v5.json")) #This was created manually
schema = {k["service_name"]:k for k in dataset_schema}
hierarchy = {}
for k in schema:
    hierarchy[k] = {}
    for id in range(len(schema[k]["hierarchy"])):
        hierarchy[k][schema[k]["slots"][id]["name"]] = schema[k]["hierarchy"][id] 

#helper functions
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


global slot_instruction
def get_slot_instruction(schema):
    slot_instruction = {}
    for domain in schema:
        slot_instruction[domain] = {}
        for slot in schema[domain]['slots']:

            slot_instruction[domain][slot["name"]] = slot["is_instruction"]=="True"
    return slot_instruction

intent_dict = get_intent_dict(schema)
slot_instruction = get_slot_instruction(schema)




#code block to create api calls
global domain_to_api 
domain_to_api = {}
for i in schema:
    domain_to_api[i] = "Get" + i.split('_')[0] 

domain_to_api["Hotels_2"] = "GetHouseStays"
domain_to_api["Services_1"] = "GetSalons"
domain_to_api["Services_2"] = "GetDentists"
domain_to_api["Services_3"] = "GetDoctors"

def get_api_calls(domains, slot_names, slot_values):
    
    api_calls_dict = defaultdict(list)
    question_seen = []
    #print(slot_names, slot_values)
    for domain in domains:
        #Every api call is saved as #({"slot_name":"slot_value"})
        for slot_name, slot_value in zip(slot_names, slot_values):
            if slot_name in intent_dict[domain]["slots"]:
                if slot_value == "?" and slot_name in question_seen:
                    continue
                api_calls_dict[domain].append({slot_name: slot_value})  
                if slot_value == "?":
                    question_seen.append(slot_name)           
    api_calls = []
    for domain in api_calls_dict:
        s = domain_to_api[domain] + "("
        for i in api_calls_dict[domain]:
            for k,v in i.items():
                if v == "dontcare":
                    v = "any"
                s = s + k + "=\"" + v + "\", "
        s = s[:-2] + ")"
        api_calls.append(s)
    return api_calls


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

#retrieving slot names and slot values from api calls
def get_slot_value_list(example):
    slot_names = []
    slot_values = []
    for api_calls in example["api_calls"]:
        domain, key_value_pairs = get_domain_kv(api_calls)
        for k,v in key_value_pairs.items():
            slot_names.append(k)
            slot_values.append(v)
    return slot_names, slot_values

def remove_instructions(standing_instructions, standing_instructions_labels, exclude_slots, exclude_slot_values):
    """
    Post-prcessing to remove instructions that are not required either because of base dataset or weird design choices.
    """
    if len(exclude_slots) == 0:
        return standing_instructions, standing_instructions_labels
    remove_indices = []
    for i in range(len(standing_instructions)):
        for slot_name, slot_value in zip(exclude_slots, exclude_slot_values):
            slot_name = slot_name.replace("_", " ")
            if slot_name in standing_instructions[i].split("then")[-1]:
                if slot_name == "location":
                    if "from location" in standing_instructions[i] or "to location" in standing_instructions[i] or "address of location" in standing_instructions[i] or "pickup location" in standing_instructions[i]:
                        continue
                if slot_name == "city":
                    if "origin city" in standing_instructions[i] or "pickup city" in standing_instructions[i] or "destination city" in standing_instructions[i] or "city of event" in standing_instructions[i]:
                        continue
                if slot_name == "origin":
                    if "origin city" in standing_instructions[i] or "origin airport" in standing_instructions[i] or "origin station name" in standing_instructions[i]:
                        continue
            
                if slot_name == "destination":
                    if "destination city" in standing_instructions[i] or "destination airport" in standing_instructions[i] or "destination station name" in standing_instructions[i]:
                        continue
                    
                if slot_name == "category" and "subcategory" in standing_instructions[i]:
                    continue

                if slot_name == "rating" and ("average rating" in standing_instructions[i] or "star rating" in standing_instructions[i]):
                    continue

                if  slot_name == "type":
                    if "car type" in standing_instructions[i] or "account type" in standing_instructions[i] or "ride type" in standing_instructions[i] or "event type" in standing_instructions[i] or "show type" in standing_instructions[i]:
                        continue
                if standing_instructions_labels[i] =="chains_if_a_then_b_c":
                    if str(slot_value) == "True":
                        standing_instructions[i] = standing_instructions[i].replace(slot_name, "")
                    else:
                        standing_instructions[i] = standing_instructions[i].replace(slot_name + " as " + slot_value, "")
                else:
                    remove_indices.append(i)
                
    updated_standing_instructions = [standing_instructions[i] for i in range(len(standing_instructions)) if i not in remove_indices]
    updated_standing_instructions_labels = [standing_instructions_labels[i] for i in range(len(standing_instructions_labels)) if i not in remove_indices]
    return updated_standing_instructions, updated_standing_instructions_labels


#common text processing mistakes
def common_mistakes(text, label):
    text = text.replace(' , , , ', '')
    text = text.replace(' , , ', '')
    text = text.replace(' , ', '')
    text = text.strip()
    if "and" in text and not ("preferred" in text or "then" in text):
        text = "None"
        label = "None"

    if text.endswith(' and look for'):
        text = text.replace(' and look for', '')
    if text.endswith(' then look for'):
        text = text.replace(' then look for', '')
        label = "domain_conditional"
        if "and" in text and not ("preferred" in text or "then" in text):
            text = "None"
            label = "None"
    if text.endswith(','):
        text = text[:-1]
    #If genre occurs twice anywhere in the sentece, remove the second one
    if text.count('genre') > 1:
        text = text.split(', then')[0]
        text = text.replace(' with', 'then')
        label = 'chains_if_a_then_b_c'
    
    return text, label  

#templates for instructions

def convert_conditional_slot_to_template(conditional_slot_name, conditional_slot_value, slot_name, slot_value):
        """
        Instruction template for dependent slots. If cusine is italian, then look for restaurant name as "Pizza Kitchen"
        """
        conditional_slot_name = conditional_slot_name.replace("_", " ")
        slot_name = slot_name.replace("_", " ")

        if conditional_slot_value == "True":
              old_condition = "If preference " + conditional_slot_name
        else:
              old_condition = "If " + conditional_slot_name + " is " + conditional_slot_value
        if slot_value == "True":
            new_condition = " then preference " + slot_name
        else:
            new_condition = " then look for " + slot_name + " as " + slot_value
        return old_condition + new_condition


def convert_conditional_slot_with_service_to_template(service, slot_name, slot_value):
    """
    Plain instruction template. If I ask for Hotels, then look for hotel name as "The Grand Hotel"
    """
    if type(slot_value) == list:
        slot_value = slot_value[0]
    
    slot_name = slot_name.replace("_", " ")
    if slot_name == "service":
        slot_name = ""

    if service == None:
        s = ""
    else:
        service = service.split("_")[0]
        s = "If I ask for {}, ".format(service)
    if slot_value == "True":
        return  s + "my preference is that it {}".format(slot_name) #If I ask for movie service, my preferance is wheelchair access. (For attributes that are boolean)
    return s +  "my preferred {} is {}".format(slot_name, slot_value) #If I ask for movie service, my preferred genre is action. (For attributes that are not boolean)


def convert_conditional_slot_with_service_and_chain_to_template(service, conditional_slot_name, conditional_slot_value, slot_names, slot_values):
    """
    Template for variant of the chain instructions. If I ask for Hotels, and the price is moderate, then look for hotel name as "The Grand Hotel"
    """
    service = service.split("_")[0]
    s = "If I ask for {} ".format(service)
    conditional_slot_name = conditional_slot_name.replace("_", " ")
    
    if conditional_slot_value == "True":
            old_condition = "and " + conditional_slot_name
    else:
            old_condition = "and " + conditional_slot_name + " is " + conditional_slot_value
    
    new_condition = " then look for "
    for slot_name, slot_value in zip(slot_names, slot_values):
        slot_name = slot_name.replace("_", " ")
        if slot_value == "True":
            new_condition = new_condition  + slot_name + ", "
        else:
            new_condition = new_condition + slot_name + " as " + slot_value + ", "
    
    new_condition = new_condition[:-2]
        
    return s + old_condition + new_condition


def convert_multiple_conditional_slot_with_service_and_chain_to_template(service, conditional_slot_names, conditional_slot_values, second_slot_name, second_slot_value):
    """
    Template for variant of the chain instructions. If I ask for Hotels, and the price is moderate, and the location is downtown, then look for hotel name as "The Grand Hotel"
    """
    service = service.split("_")[0]
    s = "If I ask for {}, ".format(service)
    
    old_condition = "with "

    for slot_name, slot_value in zip(conditional_slot_names, conditional_slot_values):
        slot_name = slot_name.replace("_", " ")

        if slot_value == "True":
            old_condition = "preferance "  + slot_name + ", "
        else:
            old_condition = old_condition + slot_name + " as " + slot_value + ", "
    
    new_condition = "then " 
    second_slot_name = second_slot_name.replace("_", " ")
    if second_slot_value == "True":
        new_condition = new_condition + second_slot_name
    
    else:
        new_condition = new_condition + second_slot_name + " as " + second_slot_value
    return s + old_condition + new_condition


def convert_conditional_service_to_template(main_service, secondary_service):
    """
    Template for multi-domain instructions. If I ask for Travel, then also look at Hotels
    """
    main_service = main_service.split("_")[0]
    secondary_service = secondary_service.split("_")[0]
    return "If I ask for " + main_service + ", then also look at " + secondary_service 

def also_look_for(slot_name, slot_value, k):
    """
    Template for override. If I ask for Restaurants, look in San Mateo. Also, look for restaurant name as "Pizza Kitchen"
    """
    slot_name = slot_name.replace("_", " ")
    templates = [str("Also, look for " + slot_name +  " as " + slot_value), str("Besides that, ensure " +  slot_name+  " is " + slot_value) , "Keep in mind that " + slot_name + " is " + slot_value, "Also, make sure " + slot_name + " is " + slot_value]
    return templates[k]



#dataset construction functions
def get_chains(service, cluster, slot_names, slot_values, standing_instructions, standing_instructions_labels, user_slots):
    """
    This function is used to retrieve variables dependent on the already mentioned variables. The dependent variables are then used to create instructions.
    """

    def get_if_a_then_b_c(prev_cluster, curr_cluster, seen_slots, seen_values):
        nonlocal standing_instructions, standing_instructions_labels, slot_names, slot_values
        for slot_id in prev_cluster:
            if slot_names[slot_id] not in seen_slots:
                    standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[slot_id], slot_values[slot_id]))
                    standing_instructions_labels.append("domain_conditional")
                    seen_slots.append(slot_names[slot_id])
                    seen_values.append(slot_values[slot_id])
        
        curr_slot_names = [slot_names[slot_id] for slot_id in curr_cluster]
        curr_slot_values = [slot_values[slot_id] for slot_id in curr_cluster]
        seen_slots = seen_slots + curr_slot_names
        seen_values = seen_values + curr_slot_values
        
        slot_id = prev_cluster[0]
        standing_instructions.append(convert_conditional_slot_with_service_and_chain_to_template(service, slot_names[slot_id], slot_values[slot_id], curr_slot_names, curr_slot_values))
        standing_instructions_labels.append("chains_if_a_then_b_c")
            
        return seen_slots, seen_values, standing_instructions, standing_instructions_labels
    
    def get_if_a_and_b_then_c(prev_cluster, curr_cluster, seen_slots, seen_values):
        nonlocal standing_instructions, slot_values, slot_names
        curr_slot_names = [slot_names[slot_id] for slot_id in prev_cluster]
        curr_slot_vales = [slot_values[slot_id] for slot_id in prev_cluster]
        for slot_id in prev_cluster:
            if slot_names[slot_id] not in seen_slots:
                    standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[slot_id], slot_values[slot_id]))
                    standing_instructions_labels.append("domain_conditional")
        
        seen_slots = seen_slots + curr_slot_names
        seen_values = seen_values + curr_slot_vales
        for slot_id in curr_cluster:
            if slot_names[slot_id] not in seen_slots:
                standing_instructions.append(convert_multiple_conditional_slot_with_service_and_chain_to_template(service, curr_slot_names, curr_slot_vales, slot_names[slot_id], slot_values[slot_id]))
                seen_slots.append(slot_names[slot_id])
                seen_values.append(slot_values[slot_id])
                standing_instructions_labels.append("chains_if_a_and_b_then_c")
        


        return seen_slots, seen_values, standing_instructions, standing_instructions_labels  

    
    if len(cluster) == 0:
        return standing_instructions
    
    if len(cluster) == 1: #Only one cluster. So convert into standard format
        for i in cluster[0]:
            standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[i], slot_values[i]))
            standing_instructions_labels.append("domain_conditional")
        return standing_instructions, standing_instructions_labels
    
    prev_cluster = cluster[0]
    seen_slots = []
    seen_values = []
    k = 1
    for clus in cluster[1:]:
        if len(clus) > 1:
            seen_slots, seen_values, standing_instructions, standing_instructions_labels = get_if_a_then_b_c(prev_cluster, clus, seen_slots, seen_values) #merging of instructions
            continue
        if len(prev_cluster) > 1:
            seen_slots, seen_values, standing_instructions, standing_instructions_labels = get_if_a_and_b_then_c(prev_cluster, clus, seen_slots, seen_values) #merging of instructions
            continue
        if k>=2:
            curr_slot_names = [slot_names[slot_id] for slot_id in clus]
            if 'restaurant_name' in curr_slot_names or 'hotel_name' in curr_slot_names:
                new_clus = cluster[0] + prev_cluster
                seen_slots, seen_values, standing_instructions, standing_instructions_labels = get_if_a_and_b_then_c(new_clus, clus, seen_slots, seen_values)
                prev_cluster = clus
                k = k + 1
                continue

        #waterfall 
        choices = list(itertools.product(prev_cluster, clus))
        for choice in choices:
            if slot_names[choice[0]] not in seen_slots: #main attribute mentioned with service once
                standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[choice[0]], slot_values[choice[0]]))
                standing_instructions_labels.append("domain_conditional")
                seen_slots.append(slot_names[choice[0]])
                seen_values.append(slot_values[choice[0]])
            
            if slot_names[choice[0]] in user_slots: #were uttered by user
                random.seed(len(seen_slots))
                idx = seen_slots.index(slot_names[choice[0]])
                id_array = [i for i in range(len(seen_slots)) if i != idx]
                if len(id_array) == 0: #look up from default slots?
                    standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[choice[1]], slot_values[choice[1]]))
                    standing_instructions_labels.append("domain_conditional")
                    seen_slots.append(slot_names[choice[1]])
                    seen_values.append(slot_values[choice[1]])  
                    continue  

                if len(id_array) == 1:
                    idx = id_array[0]
                else:
                    idx = random.choice(id_array)
                old_slot_name = seen_slots[idx]
                old_slot_value = seen_values[idx]
                standing_instructions.append(convert_multiple_conditional_slot_with_service_and_chain_to_template(service, [old_slot_name, slot_names[choice[0]]], [old_slot_value, slot_values[choice[0]]], slot_names[choice[1]], slot_values[choice[1]]))
                #standing_instructions.append(convert_conditional_slot_to_template(slot_names[choice[0]], slot_values[choice[0]], slot_names[choice[1]], slot_values[choice[1]]))
                standing_instructions_labels.append("chains_waterfall")
            else:
                standing_instructions.append(convert_conditional_slot_to_template(slot_names[choice[0]], slot_values[choice[0]], slot_names[choice[1]], slot_values[choice[1]]))
                standing_instructions_labels.append("chains_waterfall")
            seen_slots.append(slot_names[choice[1]])
            seen_values.append(slot_values[choice[1]])
              
    
        prev_cluster = clus
        k += 1
    
    return standing_instructions, standing_instructions_labels

def merge_frames(all_frames, intent):
    new_frame = all_frames[0]
    new_frame["state"][0]['active_intent'] = [intent] #Not meddling with the state, only changing the future slot and values
    slot_names = []
    slot_values = []
    services = new_frame["service"]
    
    for frame in all_frames:
        
        for service in frame["service"]:
            if service not in services:
                services.append(service)
        
        for s in range(len(frame["state"])):
            for slot, value in zip(frame["state"][s]["slot_values"]["slot_name"], frame["state"][s]["slot_values"]["slot_value_list"]):
                if slot in slot_names:
                    continue

                slot_names.append(slot)
                slot_values.append(value)
        
    new_frame["service"] = services
    new_frame["state"][0]["slot_values"]["slot_name"] = slot_names
    new_frame["state"][0]["slot_values"]["slot_value_list"] = slot_values
    new_frame["state"] = [new_frame["state"][0]] #This may cause problems in backward compatibility of SGD

    return new_frame


#functions to convert SGD to NLSI
def convert_to_standing_instructions_single_domain(example):
    """Converts first utterance of a dialogue to a simple instruction. The instruction is based on the last frame before the intent changes."""
    """
    Consists of simple defaults and first level of chained instructions
    Does not consider examples that have slots which are non instructions
    Does not consider examples that have more than one domain
    """     

    service = example["services"]
    if len(service) > 1:
        return None #Not supported in this function as only single domain examples are considered
    
    dialogue_id = example["dialogue_id"]
    curr_intent = example["turns"]["frames"][0]["state"][0]["active_intent"]
    start_frame = example["turns"]["frames"][0]
    curr_utterance = example["turns"]["utterance"][0]
    standing_instructions = []
    standing_instructions_labels = []

    exclude_slot_names = []
    exclude_slot_values = []
    if len(start_frame["state"][0]["slot_values"]["slot_name"]) > 0:
        exclude_slot_names = [name for name in start_frame["state"][0]["slot_values"]["slot_name"]]
        exclude_slot_values = [value[0] for value in start_frame["state"][0]["slot_values"]["slot_value_list"]]

    for k in range(0, len(example["turns"]["utterance"]),2): #Find the last frame before the intent changes. Typically the examples follow search -> book. We currently only consider the search part
        if example["turns"]["frames"][k]["state"][0]["active_intent"]!=curr_intent:
            break
        curr_frame = example["turns"]["frames"][k]

    #all_intents = list(set([frame["state"][0]["active_intent"] for frame in example["turns"]["frames"]]))
    all_intents = [curr_intent]
    curr_indices = []   
    slot_dict = {}
    
    slot_names = [name for name in curr_frame["state"][0]["slot_values"]["slot_name"]]
    slot_values = [value[0] for value in curr_frame["state"][0]["slot_values"]["slot_value_list"]]
    
    #Storing the slot names from the given domain that are present in the current frame
    for slot_name, slot_value in zip(slot_names, slot_values):
        slot_dict[slot_name] = slot_value
        if slot_value != "":
            for i,k in enumerate(schema[service[0]]["slots"]):
                if k["name"] == slot_name:
                    curr_indices.append(i)
                    break
    

    for slot_name, slot_value in zip(start_frame["state"][0]["slot_values"]["slot_name"], start_frame["state"][0]["slot_values"]["slot_value_list"]):
        if slot_value[0] != slot_dict[slot_name]:
            curr_utterance = curr_utterance.replace(slot_value[0], slot_dict[slot_name])
    
    priority = [schema[service[0]]["hierarchy"][i] for i in curr_indices] #get the priority of the slots in the current frame. Used for chaining of instructions
    
    if -10 in priority:
        return None # at least one non-instructional slot is present => currently not supported. -10 is the priority of non-instructional slots

    if len(priority) == 0:
        return None #No slots in the frame
    

    #sort descending with priority and get indices
    sorted_indices = np.ravel(np.argsort(priority)[::-1]) #Higher priority is more generic. So "cuisine" has priority 1 while "name of restaurant" has priority 0. 
    sorted_priority = np.ravel(np.sort(priority)[::-1])
     
    if len(sorted_priority) == 1: #Only one slot
        if sorted_priority[0] == -10:
            return None #only non-instructional slots are present
        
        #Write an instruction for that single slot
        standing_instructions.append(convert_conditional_slot_with_service_to_template(service[0], slot_names[0], slot_values[0]))
        if sorted_priority[0] == -1:
            standing_instructions_labels.append("simple_default")
        else:
            standing_instructions_labels.append("domain_conditional")
            remove_indices = []
        standing_instructions, standing_instructions_labels = remove_instructions(standing_instructions, standing_instructions_labels, exclude_slot_names, exclude_slot_values)
        api_calls = get_api_calls(service, curr_frame["state"][0]["slot_values"]["slot_name"], [v[0] for v in curr_frame["state"][0]["slot_values"]["slot_value_list"]])
        new_example = {"services": service, "dialogue_id": dialogue_id, "utterance_id": 0, "utterance": curr_utterance, "standing_instructions": standing_instructions, "standing_instructions_labels": standing_instructions_labels, "frames": curr_frame, "api_calls": api_calls}
        return new_example
        
    
    #Simple defaults
    for i, value in enumerate(priority):
        if value == -1:
            standing_instructions.append(convert_conditional_slot_with_service_to_template(service[0], slot_names[i], slot_values[i]))
            standing_instructions_labels.append("simple_default")
    if max(sorted_priority) == -1: #No conditional defaults
            
            standing_instructions, standing_instructions_labels = remove_instructions(standing_instructions, standing_instructions_labels, exclude_slot_names, exclude_slot_values)
            api_calls = get_api_calls(service, curr_frame["state"][0]["slot_values"]["slot_name"], [v[0] for v in curr_frame["state"][0]["slot_values"]["slot_value_list"]])
            new_example = {"services": service, "dialogue_id": dialogue_id, "utterance_id": 0, "utterance": curr_utterance, "standing_instructions": standing_instructions, "standing_instructions_labels": standing_instructions_labels, "frames": curr_frame, "api_calls": api_calls}
            return new_example


    if sorted_priority[0] == -10: #Nothing found
        return None
    

    #Conditional defaults
    #We create clusters of slots that have the same priority. We then create a conditional instruction by combining one slot from first cluster with one slot from second cluster

    cluster = []
    curr_value = sorted_priority[0]
    curr_cluster = [sorted_indices[0]]
    for id, value in zip(sorted_indices[1:], sorted_priority[1:]):
        if value == -1:
            break
        if value == curr_value:
            curr_cluster.append(id)
        else:   
            cluster.append(curr_cluster)
            curr_cluster = [id]
            curr_value = value
    
    cluster.append(curr_cluster) 

    standing_instructions, standing_instructions_labels = get_chains(service[0], cluster, slot_names, slot_values, standing_instructions, standing_instructions_labels, exclude_slot_names)
    standing_instructions, standing_instructions_labels = remove_instructions(standing_instructions, standing_instructions_labels, exclude_slot_names, exclude_slot_values)

    #This snippet will also send examples without standing instructions
    api_calls = get_api_calls(service, curr_frame["state"][0]["slot_values"]["slot_name"], [v[0] for v in curr_frame["state"][0]["slot_values"]["slot_value_list"]])
    new_example = {"services": service, "dialogue_id": dialogue_id, "utterance_id": 0, "utterance": curr_utterance, "standing_instructions": standing_instructions, "standing_instructions_labels": standing_instructions_labels, "frames": curr_frame, "api_calls": api_calls}
    return new_example

#Code Block for Multi Domain instructions



def get_instructions_per_domain(curr_frame, priority, service, slot_names, slot_values, user_slots):
    standing_instructions = []
    sorted_indices = np.ravel(np.argsort(priority)[::-1]) #Higher priority is more generic. So "cuisine" has priority 1 while "name of restaurant" has priority 0. 
    sorted_priority = np.ravel(np.sort(priority)[::-1])
    standing_instructions_labels = []
    if len(sorted_priority) == 0:
        return standing_instructions, standing_instructions_labels
    if len(sorted_priority) == 1: #Only one slot
        if sorted_priority[0] == -10:
            return [], [] #only non-instructional slots are present
 
        #Write an instruction for that single slot
        standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[0], slot_values[0]))
        if sorted_priority[0] == -1:
            standing_instructions_labels.append("simple_default")
        else:
            standing_instructions_labels.append("domain_conditional")
        return standing_instructions, standing_instructions_labels

    #Simple defaults
    for i, value in enumerate(priority):
        if value == -1:
            standing_instructions.append(convert_conditional_slot_with_service_to_template(service, slot_names[i], slot_values[i]))
            standing_instructions_labels.append("simple_default")
    if max(sorted_priority) == -1: #No conditional defaults
        return standing_instructions, standing_instructions_labels

    

    #Conditional defaults
    #We create clusters of slots that have the same priority. We then create a conditional instruction by combining one slot from first cluster with one slot from second cluster

    cluster = []
    curr_value = sorted_priority[0]
    curr_cluster = [sorted_indices[0]]
    for id, value in zip(sorted_indices[1:], sorted_priority[1:]):
        if value == -1:
            break
        if value == curr_value:
            curr_cluster.append(id)
        else:   
            cluster.append(curr_cluster)
            curr_cluster = [id]
            curr_value = value
    
    cluster.append(curr_cluster)


    standing_instructions, standing_instructions_labels = get_chains(service, cluster, slot_names, slot_values, standing_instructions, standing_instructions_labels, user_slots)   


    return standing_instructions, standing_instructions_labels

def convert_to_standing_instructions_multiple_domain(example):
    
    services = example["services"]

    if len(services) == 1:
        return None #Not supported in this function as only multiple domain examples are considered
    
    if "-".join(sorted(services)) not in domain_dict:
        return None
    
    if domain_dict["-".join(sorted(services))] == "n":
        return None
    dialogue_id = example["dialogue_id"]
    
    curr_utterance = example["turns"]["utterance"][0]
    main_service = example["turns"]["frames"][0]["service"][0] #The service that is currently being requested
    start_frame = example["turns"]["frames"][0]
    
    
    exclude_slot_names = []
    exclude_slot_values = []
    if len(start_frame["state"][0]["slot_values"]["slot_name"]) > 0:
        exclude_slot_names = [name for name in start_frame["state"][0]["slot_values"]["slot_name"]]
        exclude_slot_values = [value[0] for value in start_frame["state"][0]["slot_values"]["slot_value_list"]]

    all_intents = [frame["state"][0]["active_intent"] for frame in example["turns"]["frames"]]
    
    standing_instructions = []
    standing_instructions_labels = []
    #Manipulate the frames
    all_frames = []
    
    curr_service = example["turns"]["frames"][0]["service"]
    curr_intent = example["turns"]["frames"][0]["state"][0]["active_intent"]


    #iterate over user turns
    for k in range(2, len(example["turns"]["utterance"]),2):
        if example["turns"]["frames"][k]["service"] != curr_service:
            all_frames.append(example["turns"]["frames"][k-2])
        curr_service = example["turns"]["frames"][k]["service"]
       
    curr_frame = merge_frames(all_frames, curr_intent)
    curr_services = curr_frame["service"]
    curr_indices = [[] for i in range(len(curr_services))]   
    curr_slots = [[] for i in range(len(curr_services))]
    curr_values = [[] for i in range(len(curr_services))]

    slot_dict = {}
    
    
    seen_slots = []
    
    #Storing the slot names from the given domain that are present in the current frame
    for c, service in enumerate(curr_services):
        for slot_name, slot_value in zip(curr_frame["state"][0]["slot_values"]["slot_name"], curr_frame["state"][0]["slot_values"]["slot_value_list"]):
            if slot_value[0] == "dontcare" and slot_name not in exclude_slot_names:
                continue
                
            slot_dict[slot_name] = slot_value[0]
                
            if slot_value != "":
                for i,k in enumerate(schema[service]["slots"]):
                    if k["name"] == slot_name and slot_name not in seen_slots:
                        curr_indices[c].append(i)
                        curr_slots[c].append(slot_name)
                        curr_values[c].append(slot_value[0])
                        seen_slots.append(slot_name) #avoid duplicates
                        break
    
    all_slots = []
    all_values = []
    for c, service in enumerate(curr_services):
        all_slots = all_slots + curr_slots[c]
        all_values = all_values + curr_values[c]

                
    
    for slot_name, slot_value in zip(start_frame["state"][0]["slot_values"]["slot_name"], start_frame["state"][0]["slot_values"]["slot_value_list"]):
        try:
            if slot_value[0] != slot_dict[slot_name]:
                curr_utterance = curr_utterance.replace(slot_value[0], slot_dict[slot_name])
        except:
            return None   #weird debug statement
    
    priority = [[] for i in range(len(curr_services))]
    for c, curr_index in enumerate(curr_indices):
        priority[c] = [schema[curr_services[c]]["hierarchy"][i] for i in curr_index] #priority per domain
        if -10 in priority[c]:
            return None # at least one non-instructional slot is present => currently not supported. -10 is the priority of non-instructional slots

    seen_services = []
    remove_services = []


    
    for c in range(len(curr_services)):
        if c==0:
            user_slots = exclude_slot_names
        else:
            user_slots = []
        curr_instructions, curr_instructions_labels = get_instructions_per_domain(curr_frame, priority[c], curr_services[c], curr_slots[c], curr_values[c], user_slots)
        standing_instructions = standing_instructions + curr_instructions
        standing_instructions_labels = standing_instructions_labels + curr_instructions_labels
        if len(curr_instructions) == 0:
            remove_services.append(curr_services[c]) 
        
        if len(curr_instructions) > 0 and curr_services[c].split("_")[0] != main_service.split("_")[0] and curr_services[c].split("_")[0] not in seen_services:
            standing_instructions.append(convert_conditional_service_to_template(main_service, curr_services[c]))
            standing_instructions_labels.append("chains_domains")
            seen_services.append(curr_services[c].split("_")[0])

    for service in remove_services: #still has bugs
        services.remove(service)
    
    standing_instructions, standing_instructions_labels = remove_instructions(standing_instructions, standing_instructions_labels, exclude_slot_names, exclude_slot_values)
    
    if len(standing_instructions) > 0:
        api_calls = get_api_calls(services, curr_frame["state"][0]["slot_values"]["slot_name"], [v[0] for v in curr_frame["state"][0]["slot_values"]["slot_value_list"]])
        new_example = {"services": services, "dialogue_id": dialogue_id, "utterance_id": 0, "utterance": curr_utterance, "standing_instructions": standing_instructions, "standing_instructions_labels": standing_instructions_labels,  "frames": curr_frame, "api_calls": api_calls}
        return new_example
    return None

#post processing of the dataset
#Code Block for Multi Domain Multi Utterance instructions
def convert_to_standing_instructions_multiple_domain_multiple_utterances(example):
    
    services = example["services"]
    if services[0] == "Calendar_1":
        return None #Some examples were weird
    
    if "-".join(sorted(services)) not in domain_dict:
        return None
    
    if domain_dict["-".join(sorted(services))] == "n":
        return None
    
    dialogue_id = example["dialogue_id"]
    
    curr_utterance = example["turns"]["utterance"][0]
    main_service = example["turns"]["frames"][0]["service"][0] #The service that is currently being requested
    start_frame = example["turns"]["frames"][0]
    standing_instructions = []
    standing_instructions_labels = []
    #Manipulate the frames
    all_frames = []
    curr_utterance = []
    curr_service = example["turns"]["frames"][0]["service"]
    start_intent = example["turns"]["frames"][0]["state"][0]["active_intent"]
    all_intents = [frame["state"][0]["active_intent"] for frame in example["turns"]["frames"]]
    start_slots = example["turns"]["frames"][0]["state"][0]["slot_values"]["slot_name"]
    curr_intent = start_intent
    k = 0
    start_id = 0

    exclude_slots = []
    exclude_slots_values = []
    
    while len(start_slots) == 0:
        curr_utterance.append(example["turns"]["utterance"][k])
        exclude_slots = exclude_slots + example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_name"]
        exclude_slots_values = exclude_slots_values + [v[0] for v in example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_value_list"]]
        curr_frame = example["turns"]["frames"][k]
        all_frames.append(curr_frame)
        k = k + 1
        curr_utterance.append(example["turns"]["utterance"][k])
        k = k + 1

        start_slots = example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_name"]
        start_id = k        

    found_flag = False
    for k in range(start_id, len(example["turns"])):
        curr_utterance.append(example["turns"]["utterance"][k])

        if k%2 == 0:
            curr_slots = example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_name"]
            new_slots = set(curr_slots) - set(start_slots)
            
            if len(new_slots) > 0:
                for slot in new_slots:
                    for c in curr_service:
                        for i in schema[c]["slots"]:
                            if i["name"] == slot:
                                if i["is_instruction"]:
                                    foung_flag = True
             
            curr_intent = example["turns"]["frames"][k]["state"][0]["active_intent"]
            
            if curr_intent != start_intent:
                found_flag = True
            exclude_slots = exclude_slots + example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_name"]
            exclude_slots_values = exclude_slots_values + [v[0] for v in example["turns"]["frames"][k]["state"][0]["slot_values"]["slot_value_list"]]
            curr_frame = example["turns"]["frames"][k]

            if found_flag:
                break
             
   
    if len(curr_frame["state"][0]["requested_slots"]) > 0:
        for slot in curr_frame["state"][0]["requested_slots"]:
            exclude_slots = exclude_slots + [slot]
            exclude_slots_values = exclude_slots_values + ["?"]
            curr_frame["state"][0]["slot_values"]["slot_name"].append(slot)
            curr_frame["state"][0]["slot_values"]["slot_value_list"].append(["?"])    
    
    if len(curr_utterance)%2 == 0:
        return None
        
    #curr_utterance = curr_utterance[:-1]      
    first_frame = copy.deepcopy(curr_frame)
    all_frames.append(curr_frame)
    curr_utterance_merged = " ## ".join(curr_utterance)


    #iterate over user turns
    for k in range(2, len(example["turns"]["utterance"]),2):
        if example["turns"]["frames"][k]["service"] != curr_service:
            all_frames.append(example["turns"]["frames"][k-2])
        curr_service = example["turns"]["frames"][k]["service"]
    
 

    curr_frame = merge_frames(all_frames, curr_intent)
    difference_keys = set(curr_frame["state"][0]["slot_values"]["slot_name"]) - set(first_frame["state"][0]["slot_values"]["slot_name"]) #just to avoid calling future non instructional keys
    #print(difference_keys)
    remove_slots = []
 
    for key in difference_keys:
        for domain in curr_frame["service"]:
            for i in schema[domain]["slots"]:
                if i["name"] == key:
                    if i["is_instruction"]!="True":
                        remove_slots.append(key)
    
    """
    for slot_name, slot_value in zip(curr_frame["state"][0]["slot_values"]["slot_name"], curr_frame["state"][0]["slot_values"]["slot_value_list"]):
        if slot_value[0] in curr_utterance_merged:
            exclude_slots.append(slot_name)
            exclude_slots_values.append(slot_value[0])
    """
    
    
    curr_services = curr_frame["service"]
    curr_indices = [[] for i in range(len(curr_services))]   
    curr_slots = [[] for i in range(len(curr_services))]
    curr_values = [[] for i in range(len(curr_services))]
    slot_dict = {}
    
    seen_slots = []
    
  

    #Storing the slot names from the given domain that are present in the current frame
    for c, service in enumerate(curr_services):

        for slot_name, slot_value in zip(curr_frame["state"][0]["slot_values"]["slot_name"], curr_frame["state"][0]["slot_values"]["slot_value_list"]):
            if slot_name in remove_slots:
                continue
            if slot_value[0] == "dontcare" and slot_name not in exclude_slots:
                remove_slots.append(slot_name)
                continue
            if slot_value[0] in exclude_slots_values and slot_name not in exclude_slots and str(slot_value[0])!=True:
                continue #exclude locations if already in context
                
            slot_dict[slot_name] = slot_value[0]
            if slot_value[0] != "":
                for i,k in enumerate(schema[service]["slots"]):
                    
                    if k["name"] == slot_name and slot_name not in seen_slots:
                        curr_indices[c].append(i)
                        curr_slots[c].append(slot_name)
                        curr_values[c].append(slot_value[0])
                        seen_slots.append(slot_name) #avoid duplicates
                        break
    
    #removing don't care slots
 
    remove_ids = []
    for slot_name in remove_slots:
        idx = curr_frame["state"][0]["slot_values"]["slot_name"].index(slot_name)
        remove_ids.append(idx)
    
    new_slot_names = []
    new_slot_values = []
    for i in range(len(curr_frame["state"][0]["slot_values"]["slot_name"])):
        if i not in remove_ids:
            new_slot_names.append(curr_frame["state"][0]["slot_values"]["slot_name"][i])
            new_slot_values.append(curr_frame["state"][0]["slot_values"]["slot_value_list"][i])
    
    curr_frame["state"][0]["slot_values"]["slot_name"] = new_slot_names
    curr_frame["state"][0]["slot_values"]["slot_value_list"] = new_slot_values


    all_slots = []
    all_values = []
    for c, service in enumerate(curr_services):
        all_slots = all_slots + curr_slots[c]
        all_values = all_values + curr_values[c]


 
    for slot_name, slot_value in zip(exclude_slots, exclude_slots_values):
        if slot_name in remove_slots:
            continue
        if slot_value != slot_dict[slot_name]:
            curr_utterance_merged = curr_utterance_merged.replace(slot_value, slot_dict[slot_name])
    
    
    priority = [[] for i in range(len(curr_services))]
    for c, curr_index in enumerate(curr_indices):
        priority[c] = [schema[curr_services[c]]["hierarchy"][i] for i in curr_index] #priority per domain
        
        if -10 in priority[c] and c > 0: #second domain, likely to have not seen the non-instructional slot in the first domain
            return None # at least one non-instructional slot is present => currently not supported. -10 is the priority of non-instructional slots
        
        if -10 in priority[c]:
            for i in curr_index:
                if schema[curr_services[c]]["hierarchy"][i] == -10:
                    slot_name = schema[curr_services[c]]["slots"][i]["name"]
                    idx = curr_frame["state"][0]["slot_values"]["slot_name"].index(slot_name)
                    slot_value = curr_frame["state"][0]["slot_values"]["slot_value_list"][idx][0]
                    if slot_value not in curr_utterance_merged and slot_value != "?":
                        return None #non instructional slot is not in the utterance => hence example is excluded
        



    seen_services = []
    remove_services = []


    
    for c in range(len(curr_services)):
        if c==0:
            user_slots = exclude_slots
        else:
            user_slots = []
 
        curr_instructions, curr_instructions_labels = get_instructions_per_domain(curr_frame, priority[c], curr_services[c], curr_slots[c], curr_values[c], user_slots)
        standing_instructions = standing_instructions + curr_instructions
        standing_instructions_labels = standing_instructions_labels + curr_instructions_labels
        if len(curr_instructions) == 0:
            remove_services.append(curr_services[c]) 
        
        if len(curr_instructions) > 0 and curr_services[c].split("_")[0] != main_service.split("_")[0] and curr_services[c].split("_")[0] not in seen_services:
            standing_instructions.append(convert_conditional_service_to_template(main_service, curr_services[c]))
            standing_instructions_labels.append("chains_domains")
            seen_services.append(curr_services[c].split("_")[0])

    for service in remove_services: #still has bugs
        services.remove(service)
    
    if len(services) == 0:
        services = [main_service]
    
    standing_instructions, standing_instructions_labels = remove_instructions(standing_instructions, standing_instructions_labels, exclude_slots, exclude_slots_values)
    curr_utterance = ""
    for c,i in enumerate(curr_utterance_merged.split(" ## ")):
        if c%2 == 0:
            curr_utterance = curr_utterance + "User: " + i + "\n"
        else:
            curr_utterance = curr_utterance + "Agent: " + i + "\n"
    
    api_calls = get_api_calls(services, curr_frame["state"][0]["slot_values"]["slot_name"], [v[0] for v in curr_frame["state"][0]["slot_values"]["slot_value_list"]])
    new_example = {"services": services, "dialogue_id": dialogue_id, "utterance_id": start_id, "utterance": curr_utterance, "standing_instructions": standing_instructions, "standing_instructions_labels": standing_instructions_labels,  "frames": curr_frame, "api_calls": api_calls}
    return new_example



global slot_value_dict
instructions_dataset = json.load(open("conversion/old_instructions_dataset.json", "r")) #this needs a better one in final code
slot_value_dict = defaultdict(lambda: defaultdict(set))

for example in instructions_dataset:
    if len(example['services']) > 1:
        continue
    service = example['services'][0]
    for slot_name, slot_value in zip(example['frames']["state"][0]["slot_values"]["slot_name"], example['frames']["state"][0]["slot_values"]["slot_value_list"]):
        if slot_value[0] == "dontcare" or slot_value[0] == "?":
            continue
        if slot_name not in slot_instruction[service]:
            continue
        if slot_instruction[service][slot_name]:
            slot_value_dict[service][slot_name].add(slot_value[0])

def check_forbidden_keys(first_keys, second_keys):
        #first key from main domain
        #second key from other domain
    remove_keys = []
    all_slot_names = [["city", "area", "location", "to_location", "from_location", "origin_city", "pickup_city", "address_of_location", "pickup_location", "destination_city", 
    "origin", "destination", "where_to", "city_of_event"],
    ["category", "subcategory"],
    ["rating", "average_rating", "star_rating"],
    ["type", "car_type", "account_type", "ride_type", "event_type", "show_type"]]
    


    for key_1 in first_keys:
        for key_2 in second_keys:
            for slot_names in all_slot_names:
                if key_1 in slot_names and key_2 in slot_names:
                    remove_keys.append(key_2)
            if key_2 in key_1:
               remove_keys.append(key_2)
            if key_2 == key_1:
                remove_keys.append(key_2)
    remove_keys = list(set(remove_keys))
    return remove_keys

def updating_multi_domain(example):
    if len(example["services"]) > 2:
        return example
    if "Hotels_2" in "".join(example["services"]):
        return example
        
    new_example = copy.deepcopy(example)
    standing_instructions = example["standing_instructions"]
    standing_instructions_labels = example["standing_instructions_labels"]
    main_domain = example["services"][0].split('_')[0]
    other_domains = []
    under_score_domains = [services.split('_')[0] for services in example["services"]]

    for instruction in standing_instructions:
        m = re.match(r"If I ask for (.+), then also look at (.+)", instruction)
        if m:
            other_domains.append(m.group(2))
    api_calls = example["api_calls"]
    domain_api_calls = {}
    remapper = {"HouseStays": "Hotels", "Doctors": "Services", "Dentists": "Services", "Salons": "Services", "Salon": "Services", "Dentist": "Services", "Doctor": "Services", "HouseStay": "Hotels" }
    
    for api_call in api_calls:
        domain, kv = get_domain_kv(api_call)
        if domain in remapper:
            domain = remapper[domain]
        domain_api_calls[domain] = kv
    

    main_keys = list(domain_api_calls[main_domain].keys())
    main_keys = [k.replace(" ", "_") for k in main_keys]
    drop_keys = {}
    random.seed(len(example["utterance"]))
    add_keys = {}
    remaining_keys = {}
    all_instructions = []
    remove_ids = []
    remove_len = 0

    for domain in other_domains:
        kv = domain_api_calls[domain]
        keys_ = list(kv.keys())
        remove_keys =  check_forbidden_keys(main_keys, keys_)
        if len(remove_keys) == 0:
            remove_len = remove_len + 1
    
    if remove_len == len(other_domains):
        return example #no need to update the example

    new_slots = []
    new_values = []
    new_instructions = []
    for k,v in domain_api_calls[main_domain].items():
        new_slots.append(k)
        new_values.append(v)

    for domain in other_domains:
        kv = domain_api_calls[domain]
        keys_ = list(kv.keys())
        remove_keys =  check_forbidden_keys(main_keys, keys_)
        drop_keys[domain] = remove_keys
        idx = under_score_domains.index(domain)
        sample_domain = example["services"][idx]
        for instruction in standing_instructions:
            for r in remove_keys:
                if domain in instruction and r.replace("_", " ") in instruction:
                    remove_ids.append(standing_instructions.index(instruction))    

        curr_slot_names = list(slot_value_dict[sample_domain].keys())
        other_remove_keys = check_forbidden_keys(main_keys, curr_slot_names)
        possible_keys = list(set(curr_slot_names)-set(other_remove_keys + keys_))
    
        choose_from = random.randint(1,3)
        random.shuffle(possible_keys)
        
        add_keys[domain] = {}
        for k in possible_keys[:choose_from]:
            value = random.sample(slot_value_dict[sample_domain][k],1)
            add_keys[domain][k] = value[0]
        remaining_keys[domain] = {}
        for k,v in kv.items():
            if k not in remove_keys and k not in new_slots:
                remaining_keys[domain][k] = v


    for domain in add_keys:
        for k,v in add_keys[domain].items():
            new_values.append(v)
            new_slots.append(k)
            new_instructions.append(convert_conditional_slot_with_service_to_template(domain, k, v))
    for domain in remaining_keys:    
        for k,v in remaining_keys[domain].items():
            new_values.append(v)
            new_slots.append(k)

    new_instructions_labels = ["domain_conditional" for i in new_instructions]
   
    standing_instructions = [standing_instructions[i] for i in range(len(standing_instructions)) if i not in remove_ids]
    standing_instructions_labels = [standing_instructions_labels[i] for i in range(len(standing_instructions_labels)) if i not in remove_ids]
    standing_instructions = standing_instructions + new_instructions
    standing_instructions_labels = standing_instructions_labels + new_instructions_labels


    kv = domain_api_calls[main_domain]
    for k,v in kv.items():
        new_slots.append(k)
        new_values.append(v)

    updated_new_slots = []
    updated_new_values = []
    assert len(new_slots) == len(new_values)
    for k,v in zip(new_slots, new_values):
        if k in updated_new_slots:
            continue
        updated_new_slots.append(k)
        updated_new_values.append(v)
     
    
    new_slots = updated_new_slots
    new_values = updated_new_values
    new_example["standing_instructions"] = standing_instructions
    new_example["standing_instructions_labels"] = standing_instructions_labels
    new_example["api_calls"] = get_api_calls(example["services"], new_slots, new_values)
    new_example["frames"]['state'][0]["slot_values"]["slot_name"] = new_slots
    new_example["frames"]['state'][0]["slot_values"]["slot_value_list"] = [[v] for v in new_values]
   
    return new_example


#create slot value list across domains-slots
slot_value_dict = defaultdict(lambda: defaultdict(set))
for example in instructions_dataset:
    if len(example['services']) > 1:
        continue
    service = example['services'][0].split('_')[0]
    for slot_name, slot_value in zip(example['frames']["state"][0]["slot_values"]["slot_name"], example['frames']["state"][0]["slot_values"]["slot_value_list"]):
        if slot_value[0] == "dontcare" or slot_value[0] == "?":
            continue
        slot_value_dict[service][slot_name].add(slot_value[0])
new_dict = {}
for service in slot_value_dict:
    new_dict[service] = {}
    for k,v in slot_value_dict[service].items():
        new_dict[service][k] = list(v)

def override_simple(example):
    """
    This function is deprecated but kept for future dataset construction.
    We used an older version to create the overide dataset and directly added it to the final dataset.
    Essentially, this function takes an exisiting example, finds an attribute that can be overriden in the original example's standing instructions. 
    Then we sample a new value for the attribute and add that as an additional sentence to the original utterance to create a new example.
    We keep the old instruction in the new example's user profile but remove it from applicable standing instructions.    
    """
    random.seed(len(example["utterance"]))
    def removing_instructions_with_slot(instructions, instructions_labels, remove_names,  simp_slot_names):
        new_standing_instructions = []
        new_standing_instructions_labels = []
        new_remove_names = []
        keep_indices = []
        for i in range(len(instructions)):
            if "chains_domains" in instructions_labels[i]:
                continue
            flag = True
            for slot_name in remove_names:
                if slot_name not in instructions[i]:
                    pass
                else:
                    for sn in simp_slot_names:
                        if sn!=slot_name and sn in instructions[i]:
                            new_remove_names.append(sn)
                    flag = False
            if flag:
                keep_indices.append(i)
                    
        
        keep_indices = list(set(keep_indices))
        for i in keep_indices:
            new_standing_instructions.append(instructions[i])
            new_standing_instructions_labels.append(instructions_labels[i])
        remove_names = remove_names + new_remove_names
        
        remove_names = list(set(remove_names))
        return new_standing_instructions, new_standing_instructions_labels, remove_names

    new_example = copy.deepcopy(example)

    replace_domains = {"Hotels_2": {"Hotels":"HouseStays"}, "Services_1":{"Services":"Salon"}, "Services_2": {"Services": "Dentist"}, "Services_3": {"Services": "Doctor"}}
    
    if "chains_domains" in example["standing_instructions_labels"]:
        return example, None
    if "domain_conditional" not in example["standing_instructions_labels"] or "simple_default" not in example["standing_instructions_labels"]:
        return example, None
    
    main_service = example["services"][0]
    old_domain = main_service.split("_")[0]
    if main_service in replace_domains:
        main_service = replace_domains[main_service][example["services"][0].split("_")[0]]
    else:
        main_service = old_domain
    order_id = [i for i in range(len(example["standing_instructions"]))]
    random.shuffle(order_id) #To avoid cities getting picekd up first
    flag = True

    remove_values = []
    for i in order_id:
        instruction = example["standing_instructions"][i]
        label = example["standing_instructions_labels"][i]
        if main_service in instruction and (label == "domain_conditional" or label == "simple_default"):
            #If I ask for Houses, my preferred where to is LAX
            #If I ask for Restaurants, my preferred city is Palo Alto

            m = re.match(r"If I ask for (.+), my preferred (.+) is (.+)", instruction)
            if m:
                domain = m.group(1)
                chosen_slot = m.group(2).strip().replace(" ", "_")
                chosen_value = m.group(3)
                if chosen_slot in new_dict[old_domain].keys():
                    new_slot_value = random.sample(new_dict[old_domain][chosen_slot], 1)[0]
                    if new_slot_value.lower() != chosen_value.lower():
                        flag = False
                        break

    if flag:
        return example, None
    updated_utterance = example["utterance"] + " " + also_look_for(chosen_slot, new_slot_value, random.randint(0,3))
    
    #include any other instructions that contain the chosen value, so these could be removed.
    remove_slot_names = []
    for slot_name, slot_value in zip(example["frames"]['state'][0]["slot_values"]["slot_name"], example["frames"]['state'][0]["slot_values"]["slot_value_list"]):
       for sv in slot_value:
            if sv.lower() == chosen_value.lower():
                remove_slot_names.append(slot_name)


    simp_slot_names = [sn.replace("_", " ") for sn in example["frames"]['state'][0]["slot_values"]["slot_name"]] 

    

    #Removing older instructions
    standing_instructions = []
    standing_instructions_labels = []
    #replace_slot_names = [chosen_slot]
    remove_names = copy.deepcopy(remove_slot_names)
    remove_names = [sn.replace("_", " ") for sn in remove_names]
    
    curr_remove_names = copy.deepcopy(remove_names)
    new_standing_instructions, new_standing_instructions_labels, remove_names = removing_instructions_with_slot(example["standing_instructions"], example["standing_instructions_labels"], remove_names, simp_slot_names)
    
    
    while curr_remove_names != remove_names: #This will fail when no instructions that need updates are removed
        standing_instructions = copy.deepcopy(new_standing_instructions)
        standing_instructions_labels = copy.deepcopy(new_standing_instructions_labels)
        new_standing_instructions, new_standing_instructions_labels, remove_names = removing_instructions_with_slot(standing_instructions, standing_instructions_labels, remove_names, simp_slot_names)
        curr_remove_names = copy.deepcopy(remove_names)   
        
    remove_names = list(set(remove_names))

    updated_standing_instructions = new_standing_instructions
    updated_standing_instructions_labels = new_standing_instructions_labels
    remove_names = [sn.replace(" ", "_") for sn in remove_names]
    if chosen_slot in remove_names:
        remove_names.remove(chosen_slot)
    
    #Updating the frames
    c = 0
    remove_ids = []
    for slot_name, slot_value in zip(example["frames"]['state'][0]["slot_values"]["slot_name"], example["frames"]['state'][0]["slot_values"]["slot_value_list"]):
        if slot_name == chosen_slot:
            example["frames"]['state'][0]["slot_values"]["slot_value_list"][c] = [new_slot_value]
            
        if slot_name!= chosen_slot and slot_name in remove_names:
            remove_ids.append(c)
        c = c + 1
    new_slot_names = []
    new_slot_values = []    
    for i in range(len(example["frames"]['state'][0]["slot_values"]["slot_name"])):
        if i not in remove_ids:
            new_slot_names.append(example["frames"]['state'][0]["slot_values"]["slot_name"][i])
            new_slot_values.append(example["frames"]['state'][0]["slot_values"]["slot_value_list"][i])
        
    #new_api_calls = get_api_calls(example["api_calls"], remove_names, chosen_slot, new_slot_value) 
    new_example["api_calls"] = get_api_calls(example["services"], new_slot_names, [v[0] for v in new_slot_values])



    new_example["frames"]['state'][0]['slot_values']['slot_name'] = new_slot_names
    new_example["frames"]['state'][0]['slot_values']['slot_value_list'] = new_slot_values
    new_example["utterance"] = updated_utterance
    new_example["standing_instructions"] = updated_standing_instructions
    new_example["standing_instructions_labels"] = updated_standing_instructions_labels
  
    return new_example, chosen_slot



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


#dataset creation
    
new_dataset = []
c = 0
for i in tqdm(range(len(dataset))):
    w = convert_to_standing_instructions_single_domain(dataset[i])
    if w is not None:
        w["example_id"] = c
        w["og-id"] = i
        w["utterance"] = "User: " + w["utterance"]
        w["example_type"] = "single-domain-single-utterance"
        new_dataset.append(w)
        c = c + 1

    w = convert_to_standing_instructions_multiple_domain(dataset[i])
    if w is not None:
        if "chains_domains" in w["standing_instructions_labels"]:
            w = updating_multi_domain(w)
            
        w["example_id"] = c
        w["og-id"] = i
        w["utterance"] = "User: " + w["utterance"]
        if len(w["services"]) > 1:
            w["example_type"] = "multiple-domain-single-utterance"
        else:
            w["example_type"] = "single-domain-single-utterance"
        new_dataset.append(w)
        
        c = c + 1

    w = convert_to_standing_instructions_multiple_domain_multiple_utterances(dataset[i])
    if w is not None:
        
        if "chains_domains" in w["standing_instructions_labels"]:
            #print(i)
            w = updating_multi_domain(w)
    
        w["example_id"] = c
        w["og-id"] = i
        w["example_type"] = "multiple-domain-multiple-utterances"
        new_dataset.append(w)
        
        c = c + 1

#post-processing of the dataset
        
#Removing duplicate instructions
for example in new_dataset:
    if len(set(example["standing_instructions"])) != len(example["standing_instructions"]):
        idx = []
        sentences = []
        for i, sentence in enumerate(example["standing_instructions"]):
            if sentence not in sentences:
                sentences.append(sentence)
                idx.append(i)
        example["standing_instructions"] = [example["standing_instructions"][i] for i in idx]
        example["standing_instructions_labels"] = [example["standing_instructions_labels"][i] for i in idx]

for example in new_dataset:
    assert len(set(example["standing_instructions"])) == len(example["standing_instructions"])
 
#Filtering based on domain combinations
check_domain = {}

with open('conversion/domain.txt', 'r') as f:
    for line in f.readlines():
        w = line.strip().split("\t")
        check_domain[w[0] + w[1]] = w[2]

keep_examples = []
for example in new_dataset:
    flag = True
    for k in range(len(example["standing_instructions"])):
        if example["standing_instructions_labels"][k] == "chains_domains":
            w = example["standing_instructions"][k].split(', ')
            d1 = w[0].split("If I ask for ")[1]
            d2 = w[1].split("then also look at ")[1]
            try:
                if check_domain[d1 + d2] !='yes':
                    flag = False
                    break
            except:
                flag = False
                break
        
    if flag:
        keep_examples.append(example)

random.shuffle(keep_examples)

#Replacing some domain names:
data_id = [k.strip() for k in open('conversion/valid_dial_id.txt', 'r').readlines()]
replace_domains = {"Hotels_2": {"Hotels":"HouseStays"}, "Services_1":{"Services":"Salon"}, "Services_2": {"Services": "Dentist"}, "Services_3": {"Services": "Doctor"}}
c = 0
replace_examples = []
for example in keep_examples:
    update = []
    for k in range(len(example["standing_instructions"])):
        example["standing_instructions"][k], example["standing_instructions_labels"][k] = common_mistakes(example["standing_instructions"][k], example["standing_instructions_labels"][k])
      
    if "None" in example["standing_instructions"]:
        example["standing_instructions"] = [i for i in example["standing_instructions"] if i != "None"]
        example["standing_instructions_labels"] = [i for i in example["standing_instructions_labels"] if i != "None"]
    
    
    for domain in example["services"]:
        if domain in replace_domains:
            update.append(domain)
    for domain in update:
        key = list(replace_domains[domain].keys())[0]
        for k in range(len(example["standing_instructions"])):
            if key in example["standing_instructions"][k]:
                example["standing_instructions"][k] = str(example["standing_instructions"][k]).replace(key, replace_domains[domain][key])
    if example['dialogue_id'] in data_id:
        replace_examples.append(example)
    elif c < 100:
        replace_examples.append(example)
        c = c + 1

#Saving the intermediate dataset
#This dataset only consists of oracle standing instructions and api calls for "none", "simple", "multihop", and "multidomain"
with open('conversion/dataset_instructions_apis_v1.json', 'w') as outfile:
    json.dump(replace_examples, outfile, indent=4)


