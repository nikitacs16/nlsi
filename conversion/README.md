## Dataset Creation
Our dataset creation process is outlined in Section 3 of the paper. 

Following files should be run in that order to create the dataset:

```create_template_dataset.py``` Generates the templated version of the dataset for the following reasoning types: NoneApplicable, Plain, MultiHop, and MultiDomain.

```add_user_profiles.py``` updates the raw dataset with user profiles.

```create_conflict_multi_preference_dataset.py ``` The above dataset containing the four types of instructions is used to create MutliPreference and Conflict examples. 

```merge_and_split_datasets.py`` This file combines all the reasoning types into one dataset, fixes some common errors in mapping SGD to NLSI, and then creates the train/dev/test split.

```create_paraphrase_dataset.py``` This file creates the final version of the dataset which replaces templated standing instructions with paraphrases.

```load_process_data.py```

All json/.txt files were created/generated during runing of the above files. 
Disclaimer: We found some reproducibility issues with these files which are no longer recoverable. We recommend to use the dataset uploaded in huggingface. Use the files in this folder for creation of newer standing instructions dataset.


## Dataset Format

The attributes are as follows:
```
example_id: unique id for the example

user_utterance: dialogue for which the standing instructions may need to be invoked

all_standing_instructions: collection of standing instructions for this example, referred to as user profile

applicable_standing_instructions: ground truth standing instructions for this example

api_calls: list of the corresponding API calls

metadata: dictionary consisting of example id, mapping id to the SGD dataset, and reasoning type of the example
```


