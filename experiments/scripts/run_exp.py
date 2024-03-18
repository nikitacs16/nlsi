import asyncio
import dataclasses
import json
import random
import os
import datetime
import sys
from typing import List, Optional, Tuple, Dict, Any
from transformers import HfArgumentParser
from datasets import load_dataset
from pathlib import Path
from nlsi.common.config_args import ModelArguments, DataArguments, ExperimentArguments
from nlsi.data.datum import NLSIDatum
from nlsi.interpret.simple_selector import SelectorModel
from nlsi.interpret.simple_embedding_similarity import SelectorEmbeddingModel
from nlsi.interpret.simple_bm25_si import BM25SIModel
from nlsi.interpret.simple_paraphraser import ParaphraserModel
from nlsi.interpret.simple_sv_parser import SVParserModel
from nlsi.interpret.generic_model import GenericModel
from nlsi.common import intercept_logger
import time

class SGDNLSIExperiment:
    """
    """

    model_args: ModelArguments
    data_args: DataArguments
    experiment_args: ExperimentArguments
    train_data: List[NLSIDatum]
    dev_data: Optional[List[NLSIDatum]] = None
    test_data: Optional[List[NLSIDatum]] = None
    parser_model: Optional[GenericModel] = None

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        experiment_args: ExperimentArguments,
    ):

        self.data_args = data_args
        self.experiment_args = experiment_args
        self.model_args = model_args
        self.train_data, self.val_data, self.test_data = self.load_data()
        if self.experiment_args.exp_type == "selector" or self.experiment_args.exp_type == "selector_multi_pass":
            self.parser_model = SelectorModel(
                model_args=self.model_args,
                seed_examples=self.train_data,
                exp_type=self.experiment_args.exp_type
            )
        elif self.experiment_args.exp_type == "selector_embedding":
            self.parser_model = SelectorEmbeddingModel(
                model_args=self.model_args,
                seed_examples=self.train_data,
                exp_type=self.experiment_args.exp_type
            )
        elif self.experiment_args.exp_type == "selector_bm25":
            self.parser_model = BM25SIModel(
                model_args=self.model_args,
                seed_examples=self.train_data,
                exp_type=self.experiment_args.exp_type
            )
        elif self.experiment_args.exp_type == "paraphraser":
            self.parser_model = ParaphraserModel(
                model_args=self.model_args,
                seed_examples=self.train_data,
                exp_type=self.experiment_args.exp_type
            )
        

        else:
            self.parser_model = SVParserModel(
                model_args=self.model_args,
                seed_examples=self.train_data,
                exp_type=self.experiment_args.exp_type
            )

    def generate_predictions(
        self,
        eval_data: List[NLSIDatum],
        output_filename: Path
    ) -> List[NLSIDatum]:
        
        assert self.parser_model is not None
        if self.model_args.prompt_instruction.endswith(".txt"):
            with open(self.model_args.prompt_instruction, "r") as f:
                prompt_instruction = f.read()
        else:
            prompt_instruction = self.model_args.prompt_instruction
        
        if self.experiment_args.exp_type == "selector_embedding":
            ret: List[NLSIDatum] = self.parser_model.predict_instruction(
                eval_data,
                prompt_instruction=prompt_instruction,
                output_writer_path=output_filename,
            )
            return ret
        ret: List[NLSIDatum] = asyncio.run(
            self.parser_model.predict_instruction(
                test_data = eval_data,
                prompt_instruction=prompt_instruction,
                output_writer_path=output_filename
            )
        )
        return ret

    def get_automated_evaluations(self, predicted_outcomes) -> Dict[str, Any]:
        assert self.parser_model is not None
        return self.parser_model._evaluation_(predicted_outcomes)

    def load_data(
        self,
    ) -> Tuple[List[NLSIDatum], Optional[List[NLSIDatum]], Optional[List[NLSIDatum]]]:
        """
        Returns train, dev, test data
        """
        print(">>"*25)
        print(">>>> Loading data")
        #data_vers = self.data_args.data_version
        ret = {}
        data = load_dataset('nikitam/nlsi')

        for split in ["train", "validation", "test"]:
            ret[split] = [NLSIDatum.from_json(datum) for datum in data[split]]

        if self.data_args.custom_file is not None: 
            print(f"Loading {self.data_args.custom_file}")
            with open(self.data_args.custom_file, "r") as f:
                tmp = [NLSIDatum.from_json(json.loads(line)) for line in f.readlines()]
                #print(tmp[0].ref_applicable_standing_instructions)
                print("Setting custom_file as "  + self.data_args.split_to_use_for_eval)
                ret[self.data_args.split_to_use_for_eval] = tmp
                print(len(ret[self.data_args.split_to_use_for_eval]))

        if self.experiment_args.num_examples_to_use_train != -1:
            random.shuffle(ret["train"])
            print(f"Adjusting: #train_data earlier: {len(ret['train'])}")
            ret["train"] = ret["train"][: self.experiment_args.num_examples_to_use_train]
            print(
                f"#train_data after applying num_examples_to_use_train: {len(ret['train'])}"
            )
        return ret["train"], ret["validation"], ret["test"]

    def run_exp(self):

        print(">>"*25)
        print(">>>> Saving config files")
        self.exp_folder = exp_folder = f"tmp/{self.experiment_args.exp_name}/"
        os.makedirs(exp_folder, exist_ok=True)
        # make a copy of the configs to the exp_folder
        with open(f"{exp_folder}/model_args.json", "w") as fw:
            json.dump(dataclasses.asdict(self.model_args), fw)
        with open(f"{exp_folder}/data_args.json", "w") as fw:
            json.dump(dataclasses.asdict(self.data_args), fw)
        # dump experiment_args with timestamp in the filename
        with open(f"{exp_folder}/experiment_args.{str(datetime.datetime.now())}.json", "w") as fw:
            json.dump(dataclasses.asdict(self.experiment_args), fw)

        print(">>"*25)
        print(">>>> Collecting eval data")
        split_to_use_for_eval = self.data_args.split_to_use_for_eval
        assert split_to_use_for_eval in [ "val", "test"]
        eval_data_to_use: Optional[List[NLSIDatum]] = {"val": self.val_data, "test": self.test_data}[split_to_use_for_eval]
        if eval_data_to_use is not None:
            print(f"#eval_data_to_use: {len(eval_data_to_use)}")
            if self.experiment_args.num_examples_to_use != -1:
                eval_data_to_use = eval_data_to_use[
                    : self.experiment_args.num_examples_to_use
                ]
                print(
                    f"#eval_data_to_use after applying num_examples_to_use: {len(eval_data_to_use)}"
                )
        
        print("\n>>>> Getting predictions from examples")
        output_filename = self.output_filename = Path(f"tmp/{self.experiment_args.exp_name}/{split_to_use_for_eval}.predictions.jsonl")
        if self.experiment_args.continue_from_last_point:
            if output_filename.exists():
                print(f"WARNING: {output_filename} already exists. Will be continued from here...")
                with open(output_filename, "r") as f:
                    previous_predictions = [NLSIDatum.from_json(json.loads(line)) for line in f.readlines()]
                    previous_predictions_datum_ids = set([datum.example_id for datum in previous_predictions])
                    print(f"#eval_data_to_use before removing previously predicted examples: {len(eval_data_to_use)}")
                    # combine the previous predictions with the eval_data_to_use
                    eval_data_to_use = [datum for datum in eval_data_to_use if datum.example_id not in previous_predictions_datum_ids]
                    # print how many examples are left to be predicted
                    print(f"#eval_data_to_use after removing previously predicted examples: {len(eval_data_to_use)}")
            else:
                with open(output_filename, "w") as fw:
                    fw.write("")
        elif output_filename.exists():
            print(f"WARNING: {output_filename} already exists. Will be overwritten...")
        else:
            with open(output_filename, "w") as fw:
                fw.write("")
        predictions = self.generate_predictions(eval_data_to_use, output_filename)         

        print("\n>>>> Running evaluations")
        metrics = self.get_automated_evaluations(predictions)
        print(f"metrics = {metrics}")
        results_filename = Path(f"tmp/{self.experiment_args.exp_name}/{split_to_use_for_eval}.results.json")    
        with open(results_filename, "w") as fw:
            json.dump(metrics, fw, indent=4)        
        


        

def main():
    random.seed(0)

    parser = HfArgumentParser((ModelArguments, DataArguments, ExperimentArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # pylint: disable=unbalanced-tuple-unpacking
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args: ModelArguments = args[0]
        data_args: DataArguments = args[1]
        experiment_args: ExperimentArguments = args[2]
    else:
        # pylint: disable=unbalanced-tuple-unpacking
        args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments = args[0]
        data_args: DataArguments = args[1]
        experiment_args: ExperimentArguments = args[2]

    now = str(datetime.datetime.now())
    
    os.makedirs(f"tmp/{experiment_args.exp_name}", exist_ok=True)
    with intercept_logger.intercept_output(
        Path(f"tmp/{experiment_args.exp_name}/stdout.{now}"), Path(f"tmp/{experiment_args.exp_name}/stderr.{now}")
    ):
        experiment = SGDNLSIExperiment(
            model_args=model_args, experiment_args=experiment_args, data_args=data_args
        )
        experiment.run_exp()


if __name__ == "__main__":
    main()

