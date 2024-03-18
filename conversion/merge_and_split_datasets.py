import json
import random
from collections import defaultdict
import os
import re
import pandas as pd
from nlsi.interpret.utils import get_domain_kv

random.seed(42)


def update_example_type(example):
    if example["example_type"] == "override" or example["example_type"] == "multiple_instructions":
        return example["example_type"]
    curr_ground_truth_labels = example["standing_instructions_labels"]
    if len(curr_ground_truth_labels) == 0:
        return "no_instructions"
    if "chains_domains" in curr_ground_truth_labels:
        return "chains_domains" 
    if "chains_waterfall" in curr_ground_truth_labels:
        return "chains_simple"

    return "simple"  

def get_unique_domains(api_calls):
    domains = set()
    for api in api_calls:
        #write an re to get domain name from GetBus(x=1, y=2) or GetTrain(x=1, y=2)
        domain = re.search(r'Get(\w+)', api)
        if domain:
            domains.add(domain.group(1))
    return domains


def create_api_string(domain, kv_pairs):
    api_string = "Get" + domain + "("
    for k, v in kv_pairs.items():
        if v == "dontcare":
            v = "any"
        api_string += k + "=\"" + v + "\", "
    api_string = api_string[:-2] + ")"
    return api_string

def write_to_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)


data_1 = json.load(open('conversion/dataset_instructions_apis_user_profiles_v1.json'))
data_2 = json.load(open('conversion/dataset_instructions_apis_user_profiles_v2.json'))


for example in data_1:
    example["example_type"] = update_example_type(example)

unique_domains = defaultdict(list)
for example in data_1:
    if example["example_type"] == "chains_domains":
        services = "-".join([x.split("_")[0] for x in example["services"]])
        unique_domains[services].append(example)

data = data_1 + data_2
random.shuffle(data)

df=pd.read_csv('conversion/common-domain-slots.tsv',sep='\t',index_col=False)
df=df[df['value']=='y']
common_single_domain = defaultdict(lambda: defaultdict(str))
for row in df.iterrows():
    row = row[1]
    common_single_domain[row['domain']][row['slot_1']] = row['slot_2']

#updating the examples 
for example in data:
    new_api_calls = []
    for api in example["api_calls"]:
        domain, key_value_pairs = get_domain_kv(api)
        updated_kv = {}
        if domain in ['Banks', 'Events', 'RentalCars', 'Flights', 'RideSharing', 'Hotels', 'Buses']:
            for k, v in key_value_pairs.items():
                if k in common_single_domain[domain]:
                    updated_kv[common_single_domain[domain][k]] = v
                else:
                    updated_kv[k] = v
            new_api_calls.append(create_api_string(domain, updated_kv))

        else:
            new_api_calls.append(api)
    example["api_calls"] = new_api_calls


df = pd.read_csv('conversion/common-slots.tsv', sep='\t', index_col=False)

df = df[df['value']=='y']

slot_domain_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str))) 
mapper = {"Hotels_2": "Houses", "Services_1": "Salons", "Services_2": "Dentists", "Services_3": "Doctors"}



for row in df.iterrows():
    row = row[1]
    domain_1 = row['domain_1']
    domain_2 = row['domain_2']
    slot_1 = row['slot_1']
    slot_2 = row['slot_2']

    if domain_1 in mapper:
        domain_1 = mapper[row['domain_1']]
    domain_1 = domain_1.split('_')[0]
    if row['domain_2'] in mapper:
        domain_2 = mapper[row['domain_2']]
    domain_2 = domain_2.split('_')[0]
    if domain_1 in common_single_domain:
        if slot_1 in common_single_domain[domain_1]:
            slot_1 = common_single_domain[domain_1][slot_1]
    if domain_2 in common_single_domain:
        if slot_2 in common_single_domain[domain_2]:
            slot_2 = common_single_domain[domain_2][slot_2]
    if domain_2 in slot_domain_dict[domain_1][slot_1]:
       assert slot_2 == slot_domain_dict[domain_1][slot_1][domain_2]
    else:
        slot_domain_dict[domain_1][slot_1] [domain_2] = slot_2
    if domain_1 in slot_domain_dict[domain_2][slot_2]:
        assert slot_1 == slot_domain_dict[domain_2][slot_2][domain_1]
        
    else:
        slot_domain_dict[domain_2][slot_2][domain_1] = slot_1


for example in data:
    update_flag = False
    unique_domains = get_unique_domains(example["api_calls"])
    if example["example_type"] == "chains_domains" or len(unique_domains) > 1:
        if example["example_type"] == "chains_domains":
            for instruction, label in zip(example["standing_instructions"], example["standing_instructions_labels"]):
                if label == "chains_domains":
                    second_domain = instruction.split('then also look at ')[1]
                    second_domain = second_domain.strip()
                    if second_domain not in unique_domains:
                        update_flag = True # no api call was ever made to this domain
                        unique_domains.add(second_domain)
        
        domain_api_calls = defaultdict(list)
        update_key_values_domain = defaultdict(lambda: defaultdict(str))
        for api in example["api_calls"]:
            domain, key_value_pairs = get_domain_kv(api)
            for domain_2 in unique_domains:
            
                if domain == domain_2:
                    continue
                for k, v in key_value_pairs.items():
                    if k in slot_domain_dict[domain]:
                        if domain_2 in slot_domain_dict[domain][k]:
                            k_2 = slot_domain_dict[domain][k][domain_2]
                            #if k_2 in update_key_values_domain[domain_2]:
                            update_key_values_domain[domain_2][k_2] = v
        #updating api calls
        new_api_calls = []
        for api in example["api_calls"]:
            domain, key_value_pairs = get_domain_kv(api)
            if domain in update_key_values_domain:
                api_string = ""
                for k,v in update_key_values_domain[domain].items():
                    if k not in key_value_pairs:
                        if v == "dontcare":
                            v = "any"
                        api_string += ", " + k + "=\"" + v + "\", "
                if len(api_string) > 0:
                    api_string = api[:-1] + api_string[:-2] + ")"
                else:
                    api_string = api
                new_api_calls.append(api_string)
            else:
                new_api_calls.append(api)
        if update_flag:
            all_api_calls = "-".join(example['api_calls'])
            for domain in unique_domains:
                if domain not in all_api_calls and len(update_key_values_domain[domain]) > 0:
                    api_string = "Get" + domain + "("
                    for k, v in update_key_values_domain[domain].items():
                        if v == "dontcare":
                            v = "any"
                        api_string += k + "=\"" + v + "\", "
                    api_string = api_string[:-2] + ")"
                    new_api_calls.append(api_string)


        example["api_calls"] = new_api_calls

#Removing weird multiple instructions examples
c = 0
idx = []
for k, example in enumerate(data):
    unique_domains = get_unique_domains(example["api_calls"])
    if 'RideSharing' in unique_domains:
         idx.append(k)
         continue
    if 'RideSharing' in "-".join(example['services']):
        idx.append(k)
        continue
    #remove examples with more than 3 domains
    if len(unique_domains) > 3:
        idx.append(k)
    if len(example['user_profile']) < 3:
         idx.append(k)
    #remove examples with weird instructions
    if len(example["standing_instructions"]) != len(set(example["standing_instructions"])):
        idx.append(k)
    
    #remove examples where second domain does not have any standing instructions
    if example["example_type"] == "chains_domains" and len(example["standing_instructions"]) > 1:
            unique_domains = get_unique_domains(example["api_calls"])
      
            for si, name in zip(example["standing_instructions"], example["standing_instructions_labels"]):
                if name == "chains_domains":
                    second_domain = si.split('then also look at ')[1].strip()   
                    if second_domain not in unique_domains:
                        print(si)

                        print(example['example_id'])
                        print("Issue detected")
                        print(example["api_calls"])
                        idx.append(k)
                        break
            
    #remove examples with dontcare in the standing instructions
    s = "-".join(example["standing_instructions"])
    if "dontcare" in s :
        idx.append(k)

data = [x for k, x in enumerate(data) if k not in idx]
#Fixing Dentists typo

for example in data:
    unique_domains = get_unique_domains(example["api_calls"])
    if 'Denists' in unique_domains:
        new_api_calls = []
        for api in example["api_calls"]:
            if 'Denists' in api:
                api = api.replace('Denists', 'Dentists')
            new_api_calls.append(api)
            example["api_calls"] = new_api_calls
        unique_domains = [u.replace('Denists', 'Dentists') for u in unique_domains]
    example["services"] = list(unique_domains)


#Spliting the dataset to train-val-test
    
example_count = defaultdict(int)
for example in data:
    example_count[example['example_type']] += 1

test_examples = []
remaining_examples = []
example_count = defaultdict(int)
main_ids = []
mapper = {"Hotels_2": "HouseStays", "Services_1": "Salons", "Services_2": "Dentists", "Services_3": "Doctors"}
c = 0
for example in data:
    if example_count[example["example_type"]] < 340:
        test_examples.append(example)
        example_count[example["example_type"]] += 1
        main_ids.append(example["dialogue_id"])
    else:
        remaining_examples.append(example)
    unique_domains = get_unique_domains(example["api_calls"])
    if "RideSharing" in unique_domains:
        c+=1
print('Number of RideSharing examples', c)

print(len(test_examples))
domain_count = defaultdict(int)
for example in test_examples:
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    domain_count[main_service]+= 1

print(domain_count)
domain_count = defaultdict(int)
for example in data:
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    domain_count[main_service]+= 1

print(domain_count)
all_domains = list(domain_count.keys())
#Test set looks stratified in terms of domains and uniform for example types
print(len(domain_count))


val_data = []
example_count_main = defaultdict(int)
val_id = []
rest_data = []
random.shuffle(remaining_examples)
print(len(remaining_examples))

train_domain_examples = defaultdict(lambda: defaultdict(list))
val_domain_examples = defaultdict(lambda: defaultdict(list))
all_domain_examples = defaultdict(list)

for example in remaining_examples:
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    all_domain_examples[main_service].append(example)

small_domains = defaultdict(list)
for domain in all_domain_examples:
    if len(all_domain_examples[domain]) < 50:
        small_domains[domain] = len(all_domain_examples[domain])


domain_count = defaultdict(int)
other_examples = []
val_data = []
example_ids = []
for domain in all_domain_examples:
    val_data.extend(all_domain_examples[domain][:2]) #ensure that there are at least 2 examples per domain in val
    example_ids.extend([example["dialogue_id"] for example in all_domain_examples[domain][:2]])
    domain_count[domain] += 2
    for example in all_domain_examples[domain][:2]:
        example_count_main[example["example_type"]] += 1
    val_id.extend([example["dialogue_id"] for example in all_domain_examples[domain][:2]])

for example in remaining_examples:
    if example["example_id"] in example_ids:
        continue
    
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    main_service = example["services"][0]
    
    if main_service in small_domains and domain_count[main_service] > small_domains[main_service]/2:
        other_examples.append(example)
        continue

    #if example["example_type"] == "chains_domains" and example_count_main[example["example_type"]] > 16:
     #   other_examples.append(example)
      #  continue
    
    if example["example_type"] == "override" and example_count_main[example["example_type"]] > 25:
        other_examples.append(example)
        continue
    

    if example_count_main[example["example_type"]] < 45:
        val_data.append(example)
        example_count_main[example["example_type"]] += 1
        val_id.append(example["dialogue_id"])
    else:
        other_examples.append(example)

example_count = defaultdict(int)
for example in val_data:
    example_count[example["example_type"]] += 1
print(example_count)

domain_count = defaultdict(int)

for example in val_data:
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    domain_count[main_service]+= 1






train_data = []
domain_count = defaultdict(int)
mapper = {"Hotels_2": "Houses", "Services_1": "Salons", "Services_2": "Dentists", "Services_3": "Doctors"}
example_count = defaultdict(int)
#domain_examples = defaultdict(lambda: defaultdict(list))
domain_examples= defaultdict(list)

train_data = []
for example in other_examples:
    if example["dialogue_id"] in main_ids or example["dialogue_id"] in val_id:
        continue
    if example["example_type"] == "chains_domains":
        train_data.append(example)
        continue
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    domain_examples[main_service].append(example)

max_count = 10
k = 0
for domain, examples in domain_examples.items():
    random.shuffle(examples)
    train_data = train_data + examples[:max_count]
    k = k + 1


domain_count = defaultdict(int)
for example in train_data:
    unique_domains = list(get_unique_domains(example["api_calls"]))
    main_service = unique_domains[0]
    domain_count[main_service] += 1

example_count = defaultdict(int)
for example in train_data:
    example_count[example["example_type"]] += 1

print("Train data length", len(train_data))
print("val data length", len(val_data))
print("Test data length", len(test_examples))
#Saving the data
os.makedirs(f"conversion/template/", exist_ok=True)
write_to_json(train_data, f"conversion/template/train.json")
write_to_json(val_data, f"conversion/template/val.json")
write_to_json(test_examples, f"conversion/template/test.json")

