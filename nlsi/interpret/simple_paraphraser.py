import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlsi.data.datum import NLSIDatum, StandingInstruction
from nlsi.interpret.generic_model import GenericModel

##Selector code based on simple_parser.py
## We simply ask the LLMs to select the applicable standing instructions and potentially rewrite them!
## StandAlone code for paraphrasing


class ParaphraserModel(GenericModel):
    @classmethod
    def _parse_and_update_generated_output(
        self, complete_prompts: List[str], test_instances: NLSIDatum, outputs: str
    ) -> List[str]:
        # example output
        # Applicable Standing Instruction:
        # If I ask for Movies, my preferred theater name is Regal Cinemas Crow Canyon 6

        # parse the generated output
        all_instances = []
        for complete_prompt, test_instance, output in zip(
            complete_prompts, test_instances, outputs
        ):
            lines = output.split("\n")
            ret = ""
            current_instance = test_instance
            for line in lines:
                if line.startswith("Here are") or "paraphrase" in line:
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                all_instances.append(line)  # Anything after this is garbage
                break
        return all_instances

    # @staticmethod
    async def predict_instruction(
        self,
        test_data: List[NLSIDatum],
        prompt_instruction: Optional[str] = None,  # instruction at beginning of prompt
        output_writer_path: Optional[
            Path
        ] = None,  # write each output to file as it is generated
    ) -> List[NLSIDatum]:
        # assert output_writer_path is not None
        model_args = self.model_args
        seed_examples = self.seed_examples
        if model_args.debug_mode:
            print(
                f"[predict_instruction] Number of seed_examples: {len(seed_examples)}"
            )
        results = []
        for idx in range(0, len(test_data)):
            test_instances = [
                k.nl_instruction
                for k in test_data[idx].applicable_standing_instructions
            ]  # use entire profile as a batch
            print(test_instances)
            all_instances = []
            for k in range(
                0, len(test_instances), 20
            ):  # batch size of 20 because substrate error

                temp = await self._gen_using_gpt3(
                    prompt_instruction=prompt_instruction,
                    test_instances=test_instances[k : k + 20],
                    model_args=model_args,
                )  # this is paraphrased user profile
                all_instances.extend(temp)

            test_instance = test_data[idx]  # this is now NLSI data instance
            curr_standing_instructions = [
                k.nl_instruction for k in test_instance.applicable_standing_instructions
            ]
            curr_standing_instructions_idx = [
                test_instances.index(k) for k in curr_standing_instructions
            ]
            new_standing_instructions = [
                all_instances[k] for k in curr_standing_instructions_idx
            ]
            standing_instructions_labels = [
                k.instruction_src
                for k in test_instance.applicable_standing_instructions
            ]
            updated_standing_instructions = []
            c = 0
            for label, new_si in zip(
                standing_instructions_labels, new_standing_instructions
            ):
                u_new_si = StandingInstruction(
                    nl_instruction=new_si, instruction_src=label, instruction_id=c
                )
                c = c + 1
                updated_standing_instructions.append(u_new_si)
            print("*************")
            print(len(test_instance.applicable_standing_instructions))
            print(len(updated_standing_instructions))

            current_instance = replace(
                test_instance,
                applicable_standing_instructions=updated_standing_instructions,
                metadata={"prompt": "None", "rewritten": True},
            )

            results.extend([current_instance])
            if output_writer_path is not None:
                print(f"************* Writing to {output_writer_path}")
                with open(output_writer_path, "a") as fa:
                    for result in [current_instance]:
                        fa.write(json.dumps(dataclasses.asdict(result)) + "\n")

        return results

    @staticmethod
    def _assemble_one_datum(
        d: str, is_test: bool = False, exp_type: str = "paraphraser"
    ) -> str:

        prompt_str: str = ""
        prompt_str = prompt_str + d + "\n"

        return prompt_str

    @staticmethod
    def _evaluation_(predicted_outcomes: List[NLSIDatum]) -> Dict[str, Any]:
        # No evaluation required here!
        return {}
