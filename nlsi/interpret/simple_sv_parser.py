import re
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from nlsi.data.datum import NLSIDatum, StandingInstruction
from nlsi.evaluation.eval import Evaluation
from nlsi.interpret.generic_model import GenericModel
from nlsi.interpret.utils import get_domain_kv, convert_domain_kv_to_array




###LLM based parsing code


class SVParserModel(GenericModel):
    

    @classmethod
    def _parse_and_update_generated_output(
        self,
        complete_prompts: List[str],
        test_instances: List[NLSIDatum],
        outputs: List[str],
    ) -> Dict[str, Any]:

        all_instances = []
        for complete_prompt, test_instance, output in zip(
            complete_prompts, test_instances, outputs
        ):
            lines = output.split("\n")
            api_calls = []
            standing_instructions = []
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith("API"):
                    continue
                if not line.startswith("Get"):
                    standing_instructions.append(line)
                api_calls.append(line)
            new_meta_data = test_instance.metadata
            new_meta_data["prompt"] = complete_prompt
            new_meta_data["output"] = output
            if len(standing_instructions) > 0:
                standing_instructions = [
                    StandingInstruction(
                        nl_instruction=instruction,
                        instruction_id=k,
                        instruction_src=None,
                    )
                    for k, instruction in enumerate(standing_instructions)
                ]
                current_instance = replace(
                    test_instance,
                    pred_api_calls=api_calls,
                    pred_applicable_standing_instructions=standing_instructions,
                    metadata=new_meta_data,
                )
            else:
                current_instance = replace(
                    test_instance, pred_api_calls=api_calls, metadata=new_meta_data
                )

            all_instances.append(current_instance)

        return all_instances

    @staticmethod
    def _assemble_one_datum(
        d: NLSIDatum, is_test: bool = False, exp_type: str = "simple_sv_parser"
    ) -> str:

        prompt_list: List[str] = []
        prompt_list.append(f"Dialogue:\n{d.user_utterance}\n\n")

        if (
            exp_type == "simple_sv_parser"
        ):  # Input with rewritten utterance or general utterance
            if not is_test:
                instructions = [
                    k.nl_instruction for k in d.applicable_standing_instructions
                ]
                prompt_list.append(
                    ". ".join(instructions)
                )  # Pretending that the instructions are a single sentence

        if exp_type == "direct_sv_parser":  # User profile is given as is
            prompt_list.append("Standing Instructions:\n")
            standing_instructions = [
                k.nl_instruction for k in d.all_standing_instructions
            ]
            for standing_instruction in standing_instructions:
                prompt_list.append(f"{standing_instruction}\n")

        if exp_type == "augmented_sv_parser":  # standing instructions are provided
            if not is_test:

                instructions = [
                    k.nl_instruction for k in d.applicable_standing_instructions
                ]
            else:
                all_instructions = [
                    k.nl_instruction for k in d.all_standing_instructions
                ]
                instructions = [
                    k.nl_instruction
                    for k in d.pred_applicable_standing_instructions
                    if k.nl_instruction in all_instructions
                ]
            prompt_list.append(
                "Applicable Standing Instructions:\n"
            )  # Assumes Standing Instructions were selected in Previous step
            for standing_instruction in instructions:
                prompt_list.append(f"{standing_instruction}\n")

        if exp_type == "oracle_sv_parser":
            instructions = [
                k.nl_instruction for k in d.applicable_standing_instructions
            ]
            prompt_list.append("Applicable Standing Instructions:\n")
            for standing_instruction in instructions:
                prompt_list.append(f"{standing_instruction}\n")

        if exp_type == "joint_sv_parser":
            prompt_list.append("User Profile:\n")
            instructions = [k.nl_instruction for k in d.all_standing_instructions]
            for standing_instruction in instructions:
                prompt_list.append(f"{standing_instruction}\n")
            prompt_list.append("\n\nApplicable Standing Instructions:\n")

            if not is_test:
                instructions = [
                    k.nl_instruction for k in d.applicable_standing_instructions
                ]
                for standing_instruction in instructions:
                    prompt_list.append(f"{standing_instruction}\n")

        # schema is in prompt
        if not is_test:
            prompt_list.append("\n\nAPI Calls:\n")
            for api_call in d.api_calls:
                prompt_list.append(api_call + "\n")
            prompt_list.append("<EOS>\n\n")

        elif exp_type != "joint_sv_parser":
            prompt_list.append("\n\nAPI Calls: \n")

        return "".join(prompt_list)

    @staticmethod
    def _evaluation_(predicted_outcomes: List[NLSIDatum]) -> Dict[str, Any]:
        ground_truth = []
        predictions = []
        for d in predicted_outcomes:
            gold = []
            for api_call in d.api_calls:
                domain, key_value_pairs = get_domain_kv(api_call)
                gold = gold + convert_domain_kv_to_array(domain, key_value_pairs)
            ground_truth.append(gold)
            pred = []
            for api_call in d.pred_api_calls:
                domain, key_value_pairs = get_domain_kv(api_call)
                pred = pred + convert_domain_kv_to_array(domain, key_value_pairs)
            predictions.append(pred)
        evals = Evaluation(ground_truth, predictions)
        return {
            "exact_match": evals.exact_match(),
            "f1": evals.sample_f1(),
        }  # currently F1 is soft F1
