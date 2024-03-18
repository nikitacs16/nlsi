import random
from dataclasses import replace
from typing import Any, Dict, List

from nlsi.data.datum import NLSIDatum, StandingInstruction
from nlsi.evaluation.eval import Evaluation
from nlsi.interpret.generic_model import GenericModel

random.seed(42)
##Selector code based on simple_parser.py
## We simply ask the LLMs to select the applicable standing instructions and potentially rewrite them!


class SelectorModel(GenericModel):
    @classmethod
    def _parse_and_update_generated_output(
        self,
        complete_prompts: List[str],
        test_instances: List[NLSIDatum],
        outputs: List[str],
    ) -> Dict[str, Any]:
        # example output
        # Applicable Standing Instruction:
        # If I ask for Movies, my preferred theater name is Regal Cinemas Crow Canyon 6

        # parse the generated output
        all_instances = []
        c = True
        for complete_prompt, test_instance, output in zip(
            complete_prompts, test_instances, outputs
        ):
            lines = output.split("\n")
            ret = []

            for line in lines:
                if line.startswith("None") or line.startswith("No"):
                    ret.append("None")

                line = line.strip()

                if (
                    line.startswith("Applicable Standing Instructions")
                    or len(line) == 0
                    or line.startswith("Remaining Applicable Standing Instructions")
                ):
                    continue
                ret.append(line)

            generated_instructions = list(set(ret))
            # if c:
            print("[generated_instructions] = ", "\n".join(generated_instructions))
            # c = False
            # updating with NLSIInstructionFormat

            generated_instructions = [
                StandingInstruction(
                    nl_instruction=instruction, instruction_id=k, instruction_src=None
                )
                for k, instruction in enumerate(generated_instructions)
            ]
            if test_instance.pred_applicable_standing_instructions is not None:
                generated_instructions = (
                    generated_instructions
                    + test_instance.pred_applicable_standing_instructions
                )
            new_meta_data = test_instance.metadata
            new_meta_data["prompt"] = complete_prompt
            new_meta_data["output"] = output
            current_instance = replace(
                test_instance,
                pred_applicable_standing_instructions=generated_instructions,
                metadata=new_meta_data,
            )
            all_instances.append(current_instance)
        return all_instances

    @staticmethod
    def _assemble_one_datum(
        d: NLSIDatum, is_test: bool = False, exp_type: str = "selector"
    ) -> str:

        prompt_str: str = ""

        curr_profile = [k.nl_instruction for k in d.all_standing_instructions]

        prompt_str = (
            prompt_str + "Standing Instructions:\n" + "\n".join(curr_profile) + "\n"
        )
        prompt_str = prompt_str + "\n\nDialogue:\n" + d.user_utterance + "\n"

        # for idx, instruction in enumerate(curr_profile):
        #    prompt_str = prompt_str + str(idx+1) + ". " + instruction + "\n"

        if not is_test:
            curr_applicable_standing_instructions = [
                k.nl_instruction for k in d.applicable_standing_instructions
            ]
            k = len(curr_applicable_standing_instructions) / 2
            # curr_applicable_standing_instructions = [str(curr_profile.index(k.nl_instruction)+1) for k in d.applicable_standing_instructions]
            if len(curr_applicable_standing_instructions) == 0:
                curr_applicable_standing_instructions = ["None"]
                k = 1
            if exp_type == "selector_multi_pass":
                prompt_str = (
                    prompt_str
                    + "\n\nApplicable Standing Instructions:\n"
                    + "\n".join(curr_applicable_standing_instructions[: int(k)])
                    + "\n"
                )
                prompt_str = (
                    prompt_str
                    + "\nRemaining Applicable Standing Instructions:\n"
                    + "\n".join(curr_applicable_standing_instructions[int(k) :])
                    + "\n"
                )
                if curr_applicable_standing_instructions[int(k) :] == []:
                    prompt_str = prompt_str + "None\n"
            else:
                prompt_str = (
                    prompt_str
                    + "\n\nApplicable Standing Instructions:\n"
                    + "\n".join(curr_applicable_standing_instructions)
                    + "\n"
                )
            prompt_str = prompt_str + "<EOS>\n\n"
        else:
            prompt_str = prompt_str + "\n\nApplicable Standing Instructions:\n"
            if exp_type == "selector_multi_pass":
                other_applicable_standing_instructions = [
                    k.nl_instruction for k in d.pred_applicable_standing_instructions
                ]
                prompt_str = (
                    prompt_str
                    + "\n".join(other_applicable_standing_instructions)
                    + "\n"
                )
                prompt_str = (
                    prompt_str + "\nRemaining Applicable Standing Instructions:\n"
                )
        return prompt_str

    @staticmethod
    def _evaluation_(predicted_outcomes: List[NLSIDatum]) -> Dict[str, Any]:
        ground_truth = []
        predictions = []
        # Read jsonl file
        for d in predicted_outcomes:
            ground_truth.append(
                [k.nl_instruction.lower() for k in d.applicable_standing_instructions]
            )
            pred = []
            all_standing_instructions = [
                k.nl_instruction.lower() for k in d.all_standing_instructions
            ]  # We remove instructions not present in user profile
            for k in d.pred_applicable_standing_instructions:
                if k.nl_instruction.lower() in all_standing_instructions:
                    pred.append(k.nl_instruction.lower())
            predictions.append(pred)

        evals = Evaluation(ground_truth, predictions)
        return {"exact_match": evals.exact_match(), "f1": evals.sample_f1()}
