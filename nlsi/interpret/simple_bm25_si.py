import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlsi.common.config_args import ModelArguments
from nlsi.data.datum import NLSIDatum, StandingInstruction
from nlsi.evaluation.eval import Evaluation
from nlsi.interpret.bm25_index import GeneralBM25Retriever
from nlsi.interpret.generic_model import GenericModel

##Selector code based on simple_parser.py
## We simply ask the LLMs to select the applicable standing instructions and potentially rewrite them!


class BM25SIModel(GenericModel):
    @classmethod
    def __init__(
        self,
        model_args: ModelArguments,
        exp_type: str,
        seed_examples: Optional[List[NLSIDatum]] = None,
    ):
        # super().__init__(model_args, exp_type, seed_examples)
        self.model_args = model_args
        self.seed_examples = seed_examples
        self.exp_type = exp_type

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
        seed_instructions = []
        for example in self.seed_examples:
            seed_instructions = seed_instructions + [
                k.nl_instruction for k in example.all_standing_instructions
            ]
        seed_instructions = list(set(seed_instructions))
        # print(len(seed_instructions))
        results = []
        for idx in range(len(test_data)):
            all_seed_instructions = seed_instructions + [
                k.nl_instruction for k in test_data[idx].all_standing_instructions
            ]

            # print(len(seed_examples))
            query = test_data[idx].user_utterance
            retriever = GeneralBM25Retriever(
                train_data=all_seed_instructions,
                top_k=self.model_args.max_examples_in_prompt,
            )
            predictions = await retriever.retrieve(query)
            generated_instructions = [
                StandingInstruction(
                    nl_instruction=instruction, instruction_id=k, instruction_src=None
                )
                for k, instruction in enumerate(predictions)
            ]
            current_instance = replace(
                test_data[idx],
                pred_applicable_standing_instructions=generated_instructions,
            )
            results.append(current_instance)
        if output_writer_path is not None:
            with open(output_writer_path, "a") as fa:
                for result in results:
                    print("Writing to file")
                    fa.write(json.dumps(dataclasses.asdict(result)) + "\n")
        return results

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
