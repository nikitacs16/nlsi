# Interpreting User Requests in the Context of Natural Language Standing Instructions
Users of natural language interfaces, generally powered by Large Language Models (LLMs),often must repeat their preferences each time they make a similar request. To alleviate this, we propose including some of a user's preferences and instructions in natural language -- collectively termed standing instructions -- as additional context for such interfaces. For example, when a user states I'm hungry, their previously expressed preference for Persian food will be automatically added to the LLM prompt, so as to influence the search for relevant restaurants. We develop NLSI, a language-to-program dataset consisting of over 2.4K dialogues spanning 17 domains, where each dialogue is paired with a user profile (a set of users specific standing instructions) and corresponding structured representations (API calls). A key challenge in NLSI is to identify which subset of the standing instructions is applicable to a given dialogue. NLSI contains diverse phenomena, from simple preferences to interdependent instructions such as triggering a hotel search whenever the user is booking tickets to an event. We conduct experiments on NLSI using prompting with large language models and various retrieval approaches, achieving a maximum of 44.7% exact match on API prediction. Our results demonstrate the challenges in identifying the relevant standing instructions and their interpretation into API calls.

[Paper](https://arxiv.org/abs/2311.09796)

## Installation

Run the following commands:
```bash
git clone https://github.com/nikitacs16/nlsi
cd nlsi
poetry install
```

## Data

The processed data is available on [huggingface](https://huggingface.co/datasets/nikitam/nlsi)




## Experiments




Typically, any experiment is run with the following script from the main folder:
```
    env PYTHONPATH=. python experiments/scripts/run_exp.py 
        --num_examples_to_use  -1  
        --num_examples_to_use_train -1 
        --exp_name name_of_experiment 
        --prompt_instruction  experiments/prompts/DIRECT.txt
        --prompt_example_similarity_method bm25 
        --max_examples_in_prompt 5 
        --batch_size 1 
        --exp_type direct_sv_parser 
        --split val
```

We elaborate on options within exp_type:

```   
      selector_embedding: Contriever 
      selector_bm25: BM25
      selector: Prompt based selection experiments. Use prompts/ICL.txt and prompts/ICL-Dynamic.txt respectively.
      selector_multipass: Prompt based multipass experiment. Use prompts/MULTIPASS.txt
      direct_sv_parser: Direct interpretation. Use prompts/DIRECT.txt
      augmented_sv_parser: Select-then-Interpret experiments. Use prompts/DIRECT.txt
      joint_sv_parser: Select-And-Interpret experiments. Use prompts/Select-And-Interpret.txt
      oracle_sv_parser: Gold standing instructions. Use prompts/DIRECT.txt


```

Selection experiments produce a ```split.predictions.jsonl``` file which should be used as an input to the interpretation experiments with the ```--custom_file /path/to/split.predictions.jsonl``` argument in the command for simulating the Select-Then-Interpret experiments.

We provide a ```experiments/scripts/demo_exp.sh``` file containing the commands for all the experiments.

### Prompts
The prompts for all the experiments are stored in ```experiments/prompts/```


### Experiments with LLaMA-2

For the Selection experiments, following arguments are added to the command: 

```
--use_llama --use_huggingface --model_type meta-llama/Llama-2-7b-hf 
```

For the Interpretation experiments, following arguments are added to the command:
```
--model_type codellama/CodeLlama-7b-Instruct-hf --use_huggingface --custom_file /path/to/predictions.jsonl
```

Please also run the following command before evaluation

```
env PYTHONPATH= python experiments/scripts/postprocess_llama.py --fname /path/to/predictions.jsonl --type selection/parsing/both/multipass
```

## Evaluation

For Selection Experiments

```
env PYTHONPATH=. python experiments/scripts/analyze_evals.py --experiment /path/to/experiment/folder --fname split.predictions.jsonl --metrics_prefix split.metrics --compute_selection_metrics --remove_hallucination
```

For Interpretation Experiments

```
env PYTHONPATH=. python experiments/scripts/analyze_evals.py --experiment /path/to/experiment/folder --fname split.predictions.jsonl --metrics_prefix split.metrics --compute_slot_metrics --compute_slot_metrics_example_type

```
Replace split with val/test depending on the evaluation

## Citation

```sql
@inproceedings{moghe2024interpreting,
      title="Interpreting User Requests in the Context of Natural Language Standing Instructions", 
      author="Nikita Moghe and Patrick Xia and Jacob Andreas and Jason Eisner and Benjamin Van Durme and Harsh Jhamtani",
      year="2024",
      booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
      publisher = "Association for Computational Linguistics"
}
```
