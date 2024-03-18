import re
from typing import Tuple
def get_domain_kv(input_string: str) -> Tuple[str, dict]:
    domain_pattern = r"Get(\w+)\((.+)\)"  # defined as our API call
    matches = re.findall(domain_pattern, input_string)
    domain = ""
    if matches:
        domain = matches[0][0]
    if "True" in input_string:
        idx = input_string.index("True")
        if input_string[idx - 1] != '"':
            input_string = input_string.replace("True", '"True"')
    pattern = r'(\w+)="([^"]+)"'

    matches = re.findall(pattern, input_string)
    if not matches:

        return domain, {}

    key_value_pairs = {}
    for match in matches:
        key = match[0]
        value = match[1]
        key_value_pairs[key] = value

    return domain, key_value_pairs


# The two functions are decoupled if we want something new in evaluation
def convert_domain_kv_to_array(domain: str, key_value_pairs: dict) -> str:
    arr = []
    for key, value in key_value_pairs.items():
        s = domain.lower() + "_" + key.lower() + "=" + str(value).lower()
        arr.append(s)
    return arr