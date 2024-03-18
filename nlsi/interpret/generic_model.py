import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from openai import OpenAI
import tiktoken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from nlsi.common.config_args import ModelArguments
from nlsi.data.datum import NLSIDatum
from nlsi.interpret.bm25_index import BM25Retriever

tokenizer_mapper = {'gpt-4':'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base'}


class GenericModel:

    model_args: ModelArguments
    seed_examples: Optional[List[NLSIDatum]]
    retriever: BM25Retriever

    def __init__(
        self,
        model_args: ModelArguments,
        exp_type: str,
        seed_examples: Optional[List[NLSIDatum]] = None,
    ):
        self.model_args = model_args


        self.seed_examples = seed_examples
        self.retriever = BM25Retriever(train_data=seed_examples, top_k=5)
        self.exp_type = exp_type
        if self.model_args.use_huggingface:
            if self.model_args.use_llama:
                tokenizer = LlamaTokenizer.from_pretrained(self.model_args.model_type)
                tokenizer.pad_token = tokenizer.eos_token
                self.tokenizer = tokenizer
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_args.model_type, device_map="auto"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_type)

                tokenizer.pad_token = tokenizer.eos_token
                self.tokenizer = tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_args.model_type,
                    device_map="auto",
                    torch_dtype="auto"#self.model_args.torch_dtype,
                )

        else:
            self.tokenizer = self.tokenizer = tiktoken.get_encoding(
                    tokenizer_mapper[self.model_args.gpt_engine]
                )
            self.gpt3_client = OpenAI() #the API access key must be set on command line

    async def predict_instruction(
        self,
        test_data: List[NLSIDatum],
        prompt_instruction: Optional[str] = None,  # instruction at beginning of prompt
        output_writer_path: Optional[
            Path
        ] = None,  # write each output to file as it is generated
    ) -> List[NLSIDatum]:
        # assert output_writer_path is not None
        seed_examples = self.seed_examples
        if self.model_args.debug_mode:
            print(
                f"[predict_instruction] Number of seed_examples: {len(seed_examples)}"
            )
        results = []
        if self.model_args.batch_size == 1:
            for test_instance in test_data:
                batch_result = await self._gen_using_gpt3_or_hf(
                    prompt_instruction=prompt_instruction,
                    test_instances=[test_instance],
                    model_args=self.model_args,
                )
                results.extend(batch_result)
                if output_writer_path is not None:
                    with open(output_writer_path, "a") as fa:
                        for result in batch_result:
                            fa.write(json.dumps(dataclasses.asdict(result)) + "\n")
            return results
        for idx in range(0, len(test_data), self.model_args.batch_size):
            test_instances = test_data[idx : idx + self.model_args.batch_size]

            batch_result = await self._gen_using_gpt3_or_hf(
                prompt_instruction=prompt_instruction,
                test_instances=test_instances,
                model_args=self.model_args,
            )
            results.extend(batch_result)
            if output_writer_path is not None:
                print(f"************* Writing to {output_writer_path}")
                with open(output_writer_path, "a") as fa:
                    for result in batch_result:
                        fa.write(json.dumps(dataclasses.asdict(result)) + "\n")
        return results

    @classmethod
    def _parse_and_update_generated_output(
        self, complete_prompts, test_instance, output: str
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def _gen_using_gpt3_or_hf(
        self,
        test_instances: List[NLSIDatum],
        model_args: ModelArguments,
        prompt_instruction: Optional[str] = None,
    ) -> NLSIDatum:
        """generation using GPT3"""

        complete_prompts, complete_prompts_tokenized, _ = await self._create_prompt(
            prompt_instruction=prompt_instruction,
            test_instances=test_instances,
            use_hf=model_args.use_huggingface,
            use_llama=model_args.use_llama,
        )


        if model_args.use_huggingface:
            print("[complete_prompt] = ", complete_prompts[0])
            generated_ids = self.model.generate(
                input_ids=complete_prompts_tokenized["input_ids"],
                attention_mask=complete_prompts_tokenized["attention_mask"],
                max_length=model_args.max_length,
                temperature=model_args.temperature,
                num_return_sequences=1,
                repetition_penalty=1.1,
            )

            generation = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )



        else:
            assert model_args.batch_size == 1 #We donot want to give older example as context to GPT-3.
            print("[complete_prompt] = ", complete_prompts[0])
            
            messages = [{"role":"user","content":complete_prompts[0]}]
            try:
                response = self.gpt3_client.chat.completions.create(
                            model = model_args.gpt_engine,
                            messages=messages,
                            temperature=model_args.temperature,
                            max_tokens=model_args.max_length,)
                generation = [response.choices[0].message.content]

            except Exception as e:
                print(f"Error in getting response from GPT-3: {e}")
                generation = ["Error in getting response from GPT-3: {e}"]

        print("[generation] = ", generation[0])

        # parse the generated output
        batch_instances = self._parse_and_update_generated_output(
            complete_prompts, test_instances, generation
        )

        return batch_instances

    

    async def _create_prompt(
        self,
        prompt_instruction: str,
        test_instances: List[NLSIDatum],
        use_hf: bool = False,
        use_llama: bool = False, #
    ) -> Tuple[str, Dict[str, Any]]:

        # create prompt
        all_prompts = []
        instruction_token_length = len(self.tokenizer.encode(prompt_instruction))
        for test_instance in test_instances:
            token_count = instruction_token_length
            complete_prompt_list: List[str] = []
            #if use_llama:
            #    prompt_instruction = "<<SYS>>" + prompt_instruction + "\n<</SYS>>" #we resorted to using base models hence the template is not required, but for future use, uncomment this
            if prompt_instruction is not None:
                complete_prompt_list.append(prompt_instruction)
                complete_prompt_list.append("\n")

            chosen_prompt_examples = []
            num_examples_in_prompt = self.model_args.max_examples_in_prompt
            chosen_prompt_examples = await self._gather_prompt_examples(
                test_instance, num_examples_in_prompt=num_examples_in_prompt
            )
            #if use_llama:
            #   complete_prompt_list.append("\n[INST]")

            # examples in prompt
            if not self.model_args.is_most_to_least_similar:
                chosen_prompt_examples = chosen_prompt_examples[::-1]
            
            
            #test_instance_str = self._assemble_one_datum(
            #    test_instance, is_test=True, exp_type=self.exp_type
            #) #For future use, also consider including the length of test instance in the prompt when removing examples that exceed prompt length
                
            #removing examples that exceed prompt length
            for example in chosen_prompt_examples:
                turn_str = self._assemble_one_datum(example, exp_type=self.exp_type)
                turn_str_token_length = len(self.tokenizer.encode(turn_str))
                if token_count + turn_str_token_length  > self.model_args.max_length:
                    break #Don't add more examples if we're going to exceed the max length
                complete_prompt_list.append(turn_str)
                token_count += turn_str_token_length
            
            # add test instance
            prompt_prefix = len("".join(complete_prompt_list))
            turn_str = self._assemble_one_datum(
                test_instance, is_test=True, exp_type=self.exp_type
            )

            complete_prompt_list.append(turn_str)
            #if use_llama:
            #    complete_prompt_list.append("[/INST]")
            complete_prompt = "".join(complete_prompt_list)
            all_prompts.append(complete_prompt)
        if use_hf:
            all_prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt").to(
                self.model_args.device
            )  
        else:
            all_prompts_tokenized = [] #We do not need tokenized prompts for GPT

        return (
            all_prompts,
            all_prompts_tokenized,
            {"prompt_prefix": complete_prompt[prompt_prefix:]},
        )

    async def _gather_prompt_examples(
        self,
        test_instance: Optional[NLSIDatum] = None,
        num_examples_in_prompt: int = 10,
    ) -> List[NLSIDatum]:
        """Subselect seed examples to construct the prompt"""
        if num_examples_in_prompt == 0:
            return []
        seed_examples = self.seed_examples
        assert seed_examples is not None, "seed_examples are not set"
        if len(seed_examples) == 0:
            print("WARNING! seed_examples is empty.")
            return []
        similarity_method = self.model_args.prompt_example_similarity_method
        if similarity_method == "random":
            if len(seed_examples) > num_examples_in_prompt:
                return random.sample(seed_examples, num_examples_in_prompt)
            return seed_examples
        elif similarity_method == "bm25":
            selected_data = await self._retrieve_similar(
                query=test_instance.user_utterance, top_k=num_examples_in_prompt
            )
            return selected_data  # type: ignore
        elif similarity_method == "firstk":
            return seed_examples[:num_examples_in_prompt]
        else:
            raise NotImplementedError(
                f"Similarity method {similarity_method} not implemented."
            )

    async def _retrieve_similar(self, query: str, top_k: int = 5) -> List[NLSIDatum]:
        """Retrieve similar examples to the query"""
        assert self.retriever is not None
        print("[_retrieve_similar] query = ", query)
        data = await self.retriever.retrieve(query, top_k)
        return data

    @staticmethod
    # Prompt creation
    def _assemble_one_datum(d: NLSIDatum, is_test: bool = False) -> str:

        raise NotImplementedError

    # Evaluation is optional
    @staticmethod
    def _evaluation_(predicted_outcomes: List[NLSIDatum]) -> Dict[str, Any]:
        raise NotImplementedError
