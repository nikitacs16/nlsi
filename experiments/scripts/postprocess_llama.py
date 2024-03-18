import json
import argparse
import dataclasses
from nlsi.data.datum import StandingInstruction

def read_jsonl(data):
    new_data = []
    with open(data, 'r') as f:
        for line in f:
            new_data.append(json.loads(line))
    return new_data

def write_jsonl(data,fname):
    with open(fname, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
    return    

def clean_multipass_llama(example):
    user_profile = [i['nl_instruction'] for i in example['all_standing_instructions']]
    output = example['metadata']['output'].split(example['metadata']['prompt'])[1]
    instructions = example['metadata']['prompt'].split('Applicable Standing Instructions')[-1].split('Remaining')[0]
    instructions = instructions.split('\n')
    output = output.split('<EOS>')[0]
    output = output.split('\n')
    output = output + instructions
    output = [i.strip() for i in output if i.strip()]
    output = [i for i in output  if i in user_profile]
    output = list(set(output))
    output = [dataclasses.asdict(StandingInstruction(standing_instruction, instruction_id=f"{i}"))
                            for i, standing_instruction in enumerate(output)]

    return output



def clean_cot_llama(example):
    user_profile = [i['nl_instruction'] for i in example['all_standing_instructions']]
    output = example['metadata']['output'].split(example['metadata']['prompt'])[1]
    output = output.split('<EOS>')[0]
    temp = output.split('API Calls:\n')
    instructions = temp[0]
    instructions = instructions.split('\n')
    instructions = [i.strip() for i in instructions if i.strip()]
    instructions = [i for i in instructions if i in user_profile]
    instructions = list(set(instructions))
    api_calls = "".join(temp[1:])
    api_calls = api_calls.split('\n')
    api_calls = [i.strip() for i in api_calls if i.strip()]
    api_calls = list(set(api_calls))
    return instructions, api_calls


def clean_selection_llama(example):
    user_profile = [i['nl_instruction'] for i in example['all_standing_instructions']]
    output = example['metadata']['output'].split(example['metadata']['prompt'])[1]
    if '<EOS>' not in output:
        print('Error in output')
    output = output.split('<EOS>')[0]
    if 'Dialogue:' in output:
        output = output.split('Dialogue:')[0]
    instructions = output.split('\n')
    instructions = [i.strip() for i in instructions if i.strip()]
    instructions = [i for i in instructions if i in user_profile]
    instructions = list(set(instructions))
    #instruction_list.append(StandingInstruction(nl_instruction=instruction, instruction_id=k))
    return instructions

def clean_parsing_llama(example):
    output = example['metadata']['output'].split(example['metadata']['prompt'])[1]
    output = output.split('<EOS>')[0]
    output = output.split('\n')
    output = list(set(output))
    output = [i for i in output if len(i) > 0]
    return output

parser = argparse.ArgumentParser(description="post-processing")
parser.add_argument('--fname')
parser.add_argument('--type', choices=['parsing', 'selection', 'both', 'multipass'])
#parsing: interpretation experiments
#selection: selection experiments
#both: select-and-interpret experiments
#multipass: multipass selection experiments
args = parser.parse_args()

data = read_jsonl(args.fname)

if args.type == "parsing":
    for example in data:
        new_api_calls = clean_parsing_llama(example)
        example['ref_api_calls'] = new_api_calls
elif args.type =="selection":
    for example in data:
        new_si = clean_selection_llama(example)
        new_si_functions = [dataclasses.asdict(StandingInstruction(standing_instruction, instruction_id=f"{i}"))
                            for i, standing_instruction in enumerate(new_si)]
        example['ref_applicable_standing_instructions'] = new_si_functions
elif args.type =="both":
    for example in data:
        new_si, new_api_calls = clean_cot_llama(example)
        new_si_functions = [dataclasses.asdict(StandingInstruction(standing_instruction, instruction_id=f"{i}"))
                            for i, standing_instruction in enumerate(new_si)]
        example['ref_applicable_standing_instructions'] = new_si_functions
        example['ref_api_calls'] = new_api_calls

else:
    for example in data:
        new_si_functions = clean_multipass_llama(example)
        example['ref_applicable_standing_instructions'] = new_si_functions

fname = args.fname.split('.jsonl')[0] + '-up' + '.jsonl'
write_jsonl(data, fname)           