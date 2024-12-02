import yaml
import json


def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML file and returns its content as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}
    
def printv(message, verbosity=0):
    """
    Prints the message if verbosity is set to 1.

    Parameters:
    - message (str): The message to print.
    - verbosity (int): Controls whether the message is printed. Default is 0.
    """
    if verbosity == 1:
        print(message)


def json_to_dict(json_string):
    """
    Converts a JSON string to a Python dictionary.

    Args:
        json_string (str): A string containing JSON data.

    Returns:
        dict: A Python dictionary if the conversion is successful.
        None: If the JSON string is invalid.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None