env PYTHONPATH=. python conversion/create_template_dataset.py
env PYTHONPATH=. python conversion/add_user_profiles.py
env PYTHONPATH=. python conversion/create_conflict_multi_preference_dataset.py
env PYTHONPATH=. python conversion/merge_and_split_datasets.py
env PYTHONPATH=. python conversion/create_paraphrase_dataset.py
