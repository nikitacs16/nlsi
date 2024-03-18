import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

from nlsi.common.config_args import ModelArguments
from nlsi.data.datum import NLSIDatum, StandingInstruction
from nlsi.evaluation.eval import Evaluation
from nlsi.interpret.generic_model import GenericModel

"""
code for computing selection experiments using sentence embeddings
"""


class SelectorEmbeddingModel(GenericModel):
    @classmethod
    def __init__(
        self,
        model_args: ModelArguments,
        exp_type: str,
        seed_examples: Optional[List[NLSIDatum]] = None,
    ):
        # super().__init__(model_args, exp_type, seed_examples)
        self.model_args = model_args
        self.model = SentenceTransformer(
            self.model_args.model_st_type
        )  # Reusing for now
        self.seed_examples = seed_examples
        self.exp_type = exp_type

    def predict_instruction(
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
        for idx in range(0, len(test_data), self.model_args.batch_size):
            test_instances = test_data[idx : idx + self.model_args.batch_size]
            batch_result = self._get_similartiy_scores(
                test_instances, self.model_args.max_examples_in_prompt
            )
            results.extend(batch_result)
            if output_writer_path is not None:
                print(f"************* Writing to {output_writer_path}")
                with open(output_writer_path, "a") as fa:
                    for result in batch_result:
                        fa.write(json.dumps(dataclasses.asdict(result)) + "\n")
        return results

    def _get_similartiy_scores(self, test_instances, max_examples_in_prompt):
        """
        Returns a list of NLSIDatum objects
        """
        results = []

        for test_instance in test_instances:
            sentences, query = self._assemble_one_datum(test_instance)

            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            sentence_embeddings = util.normalize_embeddings(sentence_embeddings)
            query_embedding = util.normalize_embeddings(query_embedding)
            scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
            top_k = min(max_examples_in_prompt, len(sentences))
            top_results = torch.topk(scores, k=top_k)
            top_results = top_results[1].tolist()
            top_results = [sentences[idx] for idx in top_results]
            top_results = [
                StandingInstruction(
                    nl_instruction=instruction, instruction_id=k, instruction_src=None
                )
                for k, instruction in enumerate(top_results)
            ]

            results.append(
                replace(test_instance, pred_applicable_standing_instructions=top_results)
            )
        return results

    @staticmethod
    def _assemble_one_datum(
        d: NLSIDatum, is_test: bool = False, exp_type: str = "selector"
    ) -> str:
        sentences = [k.nl_instruction for k in d.all_standing_instructions]
        query = d.user_utterance
        return sentences, query

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
