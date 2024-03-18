import dataclasses
from typing import Optional, List
from nlsi.data.datum import NLSIDatum, StandingInstruction
import os
import json
from pprint import pprint




class SGDNLSIBasicDataProcessor:

  def __init__(self):
    pass

  def get_dialog_examples(self, data_path:str, max_cnt:Optional[int] = None) -> List[NLSIDatum]:
    data = json.load(open(data_path, "r"))
    examples = []
    for dialog_idx, row in enumerate(data):
      if max_cnt is not None and dialog_idx >= max_cnt:
        break
      applicable_standing_instructions  = [ StandingInstruction(nl_instruction=nl_instruction, instruction_src=label, instruction_id=str(j) ) for j,(nl_instruction,label) in enumerate(zip(row["standing_instructions"],
      row["standing_instructions_labels"])) ]
      all_standing_instructions =  [StandingInstruction(nl_instruction=nl_instruction, instruction_src=None, instruction_id=str(j) ) for j,nl_instruction in enumerate(row["user_profile"]) ]
      # TODO: check carefully
      api_calls = row["api_calls"]
      example = NLSIDatum(
        example_id = row["example_id"],
        user_utterance = row["utterance"],
        applicable_standing_instructions = applicable_standing_instructions, 
        all_standing_instructions = all_standing_instructions,
        api_calls=row["api_calls"],
        metadata = {"dialogue_id": row["dialogue_id"], "example_id": row["example_id"], "example_type": row["example_type"]},
        )
      examples.append(example)
    return examples

if __name__ == "__main__":
    
    sgd_nlsi_data_processor = SGDNLSIBasicDataProcessor()
    
    
    for split in ["train", "val", "test"]:   
      data_path = f"conversion/v2.9_paraphrase/{split}.json" #replace with the path to the data
      all_examples = sgd_nlsi_data_processor.get_dialog_examples(data_path)
      
      os.makedirs(f"data", exist_ok=True)
      output_path = f"data/{split}.jsonl" 
      
      with open(output_path, "w") as fw:
          for split_data_j in all_examples:
              d = json.dumps(dataclasses.asdict(split_data_j))
              fw.write(d + "\n")
 

