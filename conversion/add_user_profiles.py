import json
import numpy as np
import random
import copy
random.seed(42)


def forbidden_slot_name(slot_name, instruction):

    slot_names = ["city", "area", "location", "to location", "from location", "origin city", "pickup city", "address of location", "pickup location", "destination city", "origin", "destination"]
    if slot_name in slot_names:
        for sn in slot_names:
            if sn in instruction:
                return True
    return False

def create_user_profiles(dataset):
    domain_dataset = {}
    for i in range(len(dataset)):
        if len(dataset[i]["services"]) > 1:
            continue
        for domain in dataset[i]["services"]:
            domain = domain.split('_')[0]
            if domain not in domain_dataset:
                domain_dataset[domain] = []
            domain_dataset[domain].append(i)
    

    domain_names = list(domain_dataset.keys())

    
    
    
    new_examples = []
    domain_k = []

    for example in dataset:
        new_example = copy.deepcopy(example)
        curr_example_domains = [s.split('_')[0] for s in example["services"]]
        remaining_domains = list(set(domain_names) - set(curr_example_domains))
        domain_k.append(len(remaining_domains))
        n_domains = len(remaining_domains)#random.randint(6, len(remaining_domains))
        other_domains = random.sample(remaining_domains, n_domains)
        slot_names = [k.replace("_"," ") for k in example["frames"]["state"][0]["slot_values"]["slot_name"]]
        other_standing_instructions = []
        other_standing_instructions_labels = []
        for domain in other_domains:
            idx = random.randint(0, len(domain_dataset[domain]) - 1)
            other_standing_instructions = other_standing_instructions + dataset[domain_dataset[domain][idx]]["standing_instructions"]
            other_standing_instructions_labels = other_standing_instructions_labels + dataset[domain_dataset[domain][idx]]["standing_instructions_labels"]
        remove_instructions = []
        for i in range(len(other_standing_instructions)):
            for slot_name in slot_names:
                if slot_name in other_standing_instructions[i]:
                    remove_instructions.append(i)
                    continue
                if forbidden_slot_name(slot_name, other_standing_instructions[i]):
                    remove_instructions.append(i)
        other_standing_instructions = [other_standing_instructions[i] for i in range(len(other_standing_instructions)) if i not in remove_instructions]

        user_profile = example["standing_instructions"] + other_standing_instructions
        random.shuffle(user_profile)
        new_example["user_profile"] = user_profile
        new_examples.append(new_example)
    print("Average number of domains: ", sum(domain_k)/len(domain_k))
    return new_examples


dataset = json.load(open("conversion/dataset_instructions_apis_v1.json"))
schema = json.load(open("conversion/schema_updated_v5.json"))
domains = []
domains = [dataset[i]["services"][0].split('_')[0] for i in range(len(dataset))] #We don't need to count Restaurant_1 and Restaurant_2 as different domains
unique_domains, domain_counts = np.unique(domains, return_counts=True)

new_examples = create_user_profiles(dataset)
with open("conversion/dataset_instructions_apis_user_profiles_v1.json", "w") as f:
    json.dump(new_examples, f, indent=4)