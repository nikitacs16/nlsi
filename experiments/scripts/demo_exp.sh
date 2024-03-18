#bm25
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name selector_bm25 --max_examples_in_prompt 5   --batch_size 20 --exp_type selector_bm25 --split val 
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name interpreter_si_bm25 --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type augmented_sv_parser --split val    --custom_file tmp/selector_bm25/val.predictions.jsonl 


#contriever
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name selector_contriever --prompt_example_similarity_method bm25 --max_examples_in_prompt 5   --batch_size 32 --exp_type selector_embedding --model_st_type facebook/contriever --split val
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name interpreter_si_contriever --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type augmented_sv_parser --split val    --custom_file tmp/selector_contriever/val.predictions.jsonl

# ICL
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name selector_icl --prompt_instruction  experiments/prompts/ICL.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 0     --batch_size 1 --exp_type selector --split val 
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name interpreter_si_llm_icl --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type augmented_sv_parser --split val    --custom_file tmp/selector_icl/val.predictions.jsonl

# ICL-Dynamic
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name selector_icl_dynamic_5 --prompt_instruction  experiments/prompts/ICL-Dynamic.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5     --batch_size 1 --exp_type selector --split val 
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name interpreter_si_llm_icl_dynamic_5 --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type augmented_sv_parser --split val    --custom_file tmp/selector_icl_dynamic_5/val.predictions.jsonl

# Multipass
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name selector_multipass_static_second_pass --prompt_instruction  experiments/prompts/MULTIPASS.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5     --batch_size 1 --exp_type selector_multi_pass --split val  --custom_file tmp/selector_icl/val.predictions.jsonl
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name interpreter_si_llm_multipass_static_second_pass --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type augmented_sv_parser --split val    --custom_file tmp/selector_multipass_static_second_pass/val.predictions.jsonl


# Direct
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name direct --prompt_instruction  experiments/prompts/DIRECT.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type direct_sv_parser --split val     

# Select-And-Interpret
env PYTHONPATH=. python experiments/scripts/run_exp.py --num_examples_to_use  -1   --num_examples_to_use_train -1 --exp_name parser_cot --prompt_instruction  experiments/prompts/Select-And-Interpret.txt --prompt_example_similarity_method bm25 --max_examples_in_prompt 5    --batch_size 1 --exp_type joint_sv_parser --split val 