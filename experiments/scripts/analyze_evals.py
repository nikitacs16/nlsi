import json
from nlsi.data.datum import NLSIDatum
from typing import List, Tuple
import argparse
import os

from nlsi.evaluation.eval import Evaluation
from nlsi.interpret.simple_sv_parser import get_domain_kv, convert_domain_kv_to_array

from collections import defaultdict

def load_data(
        fname:str,
    ) -> List[NLSIDatum]:
        data: List[NLSIDatum] = []
        with open(fname, "r") as f:
            for line in f:
                datum = NLSIDatum.from_json(json.loads(line))
                data.append(datum)
        return data

def get_selection_data(data: List[NLSIDatum]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Returns a list of lists of selections, and a list of lists of gold selections
    """
    predicted_selections = []
    gold_selections = []
    profiles = []
    example_ids = []
    for datum in data:
        gold_selections.append(datum.list_of_applicable_standing_instructions)
        predicted_selections.append(datum.list_of_pred_standing_instructions)
        profiles.append(datum.user_profile)
        example_ids.append(datum.example_id)
    return gold_selections, predicted_selections, profiles, example_ids

def get_example_data_by_labels(data: List[NLSIDatum]):
    """
    Returns a list of examples by example type
    """
    example_type_data = defaultdict(list)
    for datum in data:
        example_type_data[datum.metadata['example_type']].append(datum)
    return example_type_data
        
def get_slots_data(data: List[NLSIDatum]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Returns a list of lists of slots, and a list of lists of gold slots
    """
    predicted_slots = []
    gold_slots = []
    for datum in data:
        gold_slots.append(datum.list_of_slot_values)
        predicted_slots.append(datum.list_of_pred_slot_values)
    return gold_slots, predicted_slots

def get_api_data(data: List[NLSIDatum]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Returns a list of lists of slots, and a list of lists of gold slots
    """
    predicted_slots = []
    gold_slots = []
    example_ids = []
    lengths = []
    for datum in data:
        example_ids.append(datum.example_id)
        curr_gold_slots = []
        curr_gold_domain = defaultdict(list)
        gold_length = len(datum.api_calls)
        versus = None
        for api in datum.api_calls:
            domain, kv = get_domain_kv(api)
            for k,v in kv.items():
                if 'versus' in v or 'vs' in v:
                    versus = v

                curr_gold_domain[domain].append({k:v})
            curr_gold_slots = curr_gold_slots + convert_domain_kv_to_array(domain, kv)            
        gold_slots.append(curr_gold_slots)  
        
        curr_predicted_slots = []
        c = 0
        for api in datum.pred_api_calls:
            if api.startswith('Get'):
                domain, kv = get_domain_kv(api)
                ## handling corner cases
                category_flag = False
                event_type_flag = False
                if domain in curr_gold_domain: 
                    kv_list = curr_gold_domain[domain]
                    curr_keys = [list(k.keys())[0] for k in kv_list]
                    for k,v in kv.items():
                        if 'versus' in v.lower() or 'vs' in v.lower() and versus is not None:
                            kv[k] = versus
                        if 'time' in k or 'date' in k:                    
                            if k in curr_keys:
                                idx = curr_keys.index(k) 
                                gold_value = curr_gold_domain[domain][idx][k].lower()
                                if v.lower() == gold_value:
                                    continue
                                value_tokens = v.lower().split()
                                if len(value_tokens) == 1:
                                    continue

                                gold_value_tokens = gold_value.split()
                
                                #difference between gold_value_tokens and tokens
                                diff = set(gold_value_tokens) - set(value_tokens)
                                
                                if len(diff) == 1:
                                    kv[k] = gold_value
                                if v.lower() in gold_value.lower():
                                    kv[k] = gold_value

                        if k=="event_type"  and "subcategory" not in curr_keys:
                            if "category" in curr_keys:
                                category_flag = True
                        if k=="category" and "subcategory" not in curr_keys:
                            if "event_type" in curr_keys:
                               
                                event_type_flag = True
                if category_flag:
                    v = kv["event_type"]
                    kv["category"] = v
                    del kv["event_type"]
                
                if event_type_flag:
                    v = kv["category"]
                    kv["event_type"] = v
                    del kv["category"]
      
                curr_predicted_slots = curr_predicted_slots + convert_domain_kv_to_array(domain, kv) 
                c += 1
        lengths.append(c != gold_length)
  
        predicted_slots.append(curr_predicted_slots)
    return gold_slots, predicted_slots, example_ids, lengths




# use argparse to accept experiment name as input
# and fname, which has default value of test.predictions.jsonl
# metrics output prefix, which has default value of test.metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--fname", type=str, default="test.predictions.jsonl")
    parser.add_argument("--metrics_prefix", type=str, default="test.metrics")
    parser.add_argument("--compute_selection_metrics", action="store_true")
    parser.add_argument("--compute_slot_metrics", action="store_true")
    parser.add_argument("--remove_hallucinations_in_selection", action="store_true")
    parser.add_argument("--dsv_f1", action="store_true")

    parser.add_argument("--compute_selection_metrics_example_type", action="store_true")
    parser.add_argument("--compute_slot_metrics_example_type", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment
    fname = args.fname
    output_metrics_prefix = args.metrics_prefix

    data = load_data(f"{experiment}/{fname}")

    aggregate_metrics_all = {}
    instance_wise_metrics_all = {}
    # aggregate_metrics_all_by_type = {}

    if args.compute_selection_metrics:
        ground_truth, predictions, profiles, datum_ids = get_selection_data(data)
        evals = Evaluation(ground_truth=ground_truth, predictions=predictions, datum_ids=datum_ids, profiles=profiles, preprocess=True, remove_hallucinations=args.remove_hallucinations_in_selection)
        aggregate_metrics, instance_wise_metrics = evals.get_metrics(prefix='selection_', is_selection_exp=True)
        aggregate_metrics_all.update(aggregate_metrics)
        instance_wise_metrics_all.update(instance_wise_metrics)
        # aggregate_metrics_type, _ = evals.get_metrics(prefix='selection_', is_selection_exp=True, get_by_example_type=True)
        # aggregate_metrics_all_by_type.update(aggregate_metrics_type)

    # compute slot metrics
    if args.compute_slot_metrics:
        ground_truth, predictions, datum_ids, lengths = get_api_data(data)
        
        evals = Evaluation(ground_truth, predictions, datum_ids=datum_ids, preprocess=True)
        aggregate_metrics, instance_wise_metrics = evals.get_metrics(prefix='slot_')
        exact_match_scores = instance_wise_metrics['slot_em_scores']
        for i, l in enumerate(lengths):
            if l:
                exact_match_scores[i] = 0.0
        new_exact_match = sum(exact_match_scores) / len(exact_match_scores)
        if args.dsv_f1:
            d_f1, s_f1, v_f1 =evals.get_dsv_f1_score(predictions, ground_truth)
            aggregate_metrics['d_f1'] = d_f1
            aggregate_metrics['s_f1'] = s_f1
            aggregate_metrics['v_f1'] = v_f1

        aggregate_metrics['slot_em_score'] = new_exact_match
        instance_wise_metrics['slot_em_scores'] = exact_match_scores
        aggregate_metrics_all.update(aggregate_metrics)
        instance_wise_metrics_all.update(instance_wise_metrics)

    with open(f"{experiment}/{output_metrics_prefix}.aggregate_metrics.json", "w") as f:
        json.dump(aggregate_metrics_all, f, indent=4)
        # print aggregate metrics
        for k, v in aggregate_metrics_all.items():
            print(f"{k}: {v}")

    with open(f"{experiment}/{output_metrics_prefix}.instance_wise_metrics.json", "w") as f:
        json.dump(instance_wise_metrics_all, f, indent=4)

    # compute correlations between length of profile and metrics like exact match and precision and recall
    # use instance_wise_metrics_all for this analysis
    if args.compute_selection_metrics:
        # create {experiment}/correlations/ if it doesn't exists
        if not os.path.exists(f"{experiment}/correlations/"):
            os.makedirs(f"{experiment}/correlations/")
        print("instance_wise_metrics_all: ", instance_wise_metrics_all.keys())
        for accuracy_metric in ['em_scores', 'f1_scores', 'prec_scores']:
            for feature_val in ['selection_profile_len', 'selection_pred_len', 'selection_gt_len']:
                feature_vals = instance_wise_metrics_all[feature_val]
                accuracy_vals = instance_wise_metrics_all[f'selection_{accuracy_metric}']
                corr = Evaluation.get_correlation_score(feature_vals, accuracy_vals, feature_val, accuracy_metric, plot_vals=True, plot_save_path=f"{experiment}/correlations/{accuracy_metric}_vs_{feature_val}.png")

    if args.compute_selection_metrics_example_type:
        label_wise_data = get_example_data_by_labels(data)
       
        aggregate_metrics_all_by_type = {}
        for example_type, example_data in label_wise_data.items():
            ground_truth, predictions, profiles, datum_ids = get_selection_data(example_data)
            evals = Evaluation(ground_truth=ground_truth, predictions=predictions, datum_ids=datum_ids, profiles=profiles, preprocess=True, remove_hallucinations=args.remove_hallucinations_in_selection)
            aggregate_metrics, instance_wise_metrics = evals.get_metrics(prefix='selection_', is_selection_exp=True)
            exact_match = aggregate_metrics['selection_em_score']
            for i, l in enumerate(lengths):
                if l:
                    exact_match[i] = 0.0
            aggregate_metrics["selection_em_score"] = exact_match
            aggregate_metrics_all_by_type[example_type] = aggregate_metrics
    
    

       
        with open(f"{experiment}/{output_metrics_prefix}.aggregate_metrics_by_type.json", "w") as f:
            json.dump(aggregate_metrics_all_by_type, f, indent=4)
            # print aggregate metrics
            for example_type in aggregate_metrics_all_by_type.keys():
                for k, v in aggregate_metrics_all_by_type[example_type].items():
                    if 'em_score' in k:
                        print(f"{example_type}\t{v}")
    
    if args.compute_slot_metrics_example_type:
        label_wise_data = get_example_data_by_labels(data)
        aggregate_metrics_all_by_type = {}
        for example_type, example_data in label_wise_data.items():
            ground_truth, predictions, datum_ids, lengths = get_api_data(example_data)
            evals = Evaluation(ground_truth, predictions, datum_ids=datum_ids, preprocess=True)
            aggregate_metrics, instance_wise_metrics = evals.get_metrics(prefix='slot_')

            aggregate_metrics_all_by_type[example_type] = aggregate_metrics
        
        with open(f"{experiment}/{output_metrics_prefix}.aggregate_metrics_by_type.json", "w") as f:
            json.dump(aggregate_metrics_all_by_type, f, indent=4)
            # print aggregate metrics
            for example_type in aggregate_metrics_all_by_type.keys():
                for k, v in aggregate_metrics_all_by_type[example_type].items():
                    if 'em_score' in k:
                        print(f"{example_type}\t{v}")

    
    

