import yaml

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