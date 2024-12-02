from dataclasses import dataclass, field, asdict, fields
from typing import Literal, List, Dict, Optional, Any
from itertools import product
import os
import json


from helpers import load_yaml, printv, json_to_dict


@dataclass
class Geo:
    country: str
    state: str
    city: str

    def generate_prompt(self) -> str:
        return (
            f"\t\tCountry: {self.country}\n"
            f"\t\tState: {self.state}\n"
            f"\t\tCity: {self.city}"
        )

    def location_str(self) -> str:
        return f"{self.city}, {self.state}, {self.country}"


@dataclass
class Persona:
    age: int
    sex: str
    geo: Geo
    political_leaning: Optional[Literal['conservative', 'liberal']] = None
    lang: str = field(default=None)

    def __post_init__(self):
        pass 
        # if self.political_leaning not in ['conservative', 'liberal', None]:
        #     raise ValueError("Invalid political leaning")

    def generate_prompt(self) -> str:
        return (
            f"\n\tAge: {self.age}"
            f"\n\tSex: {self.sex}"
            f"\n\tLanguage: {self.lang}"
            f"\n\tNationality:\n{self.geo.generate_prompt()}"
            # f"\n\tPolitical leaning: {self.political_leaning}"
        )

@dataclass
class Task:
    modality: Literal['text2image', 'image2text', 'text2text', 'image2image'] = field()
    lang: str = field(default=None)
    domain: str = field(default=None)
    persona: Persona = field(default=None)
    intent: str = field(default=None)
    optimal_response: str = field(default=None)
    cultural_nuances: str = field(default=None)
    external_files: List[str] = field(default_factory=list)
    internal_system_prompt: str = field(default=None)
    external_system_prompt: str = field(default=None)
    task_id: Optional[int] = field(default=None)

    # Class variable for task ID counter
    _id_counter: int = 0

    def __post_init__(self):
        if self.task_id is None:
            type(self)._id_counter += 1
            self.task_id = type(self)._id_counter
        else:
            # When loading from JSON, task_id is provided
            # Update the counter if necessary
            if self.task_id > type(self)._id_counter:
                type(self)._id_counter = self.task_id
        if self.modality == 'image2image':
            raise NotImplementedError("image2image modality is not supported yet")

    def update_intent(self, intent: dict):
        if isinstance(intent, str):
            intent = json_to_dict(intent)

        try: 
            self.intent = intent['intent']
        except KeyError:
            print(f"Error: 'intent' key not found in response: {intent}")
        try:
            self.optimal_response = intent['optimal_response']
        except KeyError:
            print(f"Error: 'optimal_response' key not found in response: {intent}")
        try:
            self.cultural_nuances = intent['cultural_nuances']
        except KeyError:
            print(f"Error: 'cultural_nuances' key not found in response: {intent}")

    def update_external_files(self, external_files: List[str]):
        self.external_files = external_files

    def prompt_intent_generation(self) -> str:
        # Prompt for generating intents
        prompt = (
            f"{self.internal_system_prompt}"
            f"Modality: {self.modality}\n"
            f"Domain: {self.domain}\n"
            f"Personal details: {self.persona.generate_prompt()}"
        )
        return prompt

    def prompt_external(self) -> str:
        # Prompt for communicating with another agent
        # Should only include information we're okay sharing publicly
        prompt = self.external_system_prompt.format(
            age=self.persona.age,
            sex=self.persona.sex,
            location=self.persona.geo.location_str(),
            question=self.intent
        )
 
        return prompt
    
    def prompt_with_persona(self) -> str: 
        prompt = "Given the following details about the user, please respond to the question below:"
        prompt += f"\n\nUSER INFORMATION:\n{self.persona.generate_prompt()}\n"
        prompt += f"\nQUESTION:\n{self.intent}"
        return prompt

@dataclass
class PromptProtocol:
    langs: List[str] = field(default_factory=lambda: ['en', 'ru', 'de'])
    country_specific_dict: Dict[str, List[Task]] = field(default_factory=dict)
    country_info_path: str = field(default='data/prompts/{}.yaml')

    # Task parameters
    modalities: List[Literal['text2image', 'image2text', 'text2text']] = field(
        default_factory=lambda: ['text2image', 'text2text']
    )

    # Persona parameters
    age_list: List[int] = field(default_factory=lambda: [25, 44, 66])

    # System prompt paths
    internal_system_prompts: Dict[str, str] = field(default_factory=dict)
    internal_system_prompt_path: str = field(default='data/prompts/system_prompt_intent.yaml')

    external_system_prompts: Dict[str, str] = field(default_factory=dict)
    external_system_prompt_path: str = field(default='data/prompts/system_prompt_user.yaml')

    verbosity: int = 0

    def __post_init__(self):
        temp_system_prompt = load_yaml(self.internal_system_prompt_path)
        temp_system_external = load_yaml(self.external_system_prompt_path)
        printv(temp_system_prompt, verbosity=self.verbosity)

        for lang in self.langs:
            yaml_data = load_yaml(self.country_info_path.format(lang))
            self.internal_system_prompts[lang] = temp_system_prompt.get(lang, "")
            self.external_system_prompts[lang] = temp_system_external.get(lang, "")
            self.country_specific_dict[lang] = self.explode_combinations(yaml_data)

    def explode_combinations(self, data: Dict) -> List[Task]:
        lang = data.get("Lang", "en")
        domains = data.get("Domains", [])
        persona_data = data.get("Persona", {})
        sexes = persona_data.get("sex", ['unspecified'])
        # political_leanings = persona_data.get("political_leaning", [None])
        ages = self.age_list

        geo_combinations = [
            (country, region, city)
            for country, regions in data.get("Geo", {}).items()
            for region, cities in regions.items()
            for city in cities
        ]

        modalities = self.modalities
        combinations = product(modalities, domains, sexes, ages, geo_combinations)
        result = []

        for modality, domain, sex, age, geo in combinations:
            # Create Geo object
            geo_obj = Geo(country=geo[0], state=geo[1], city=geo[2])
            # Create Persona object
            persona_obj = Persona(age=age, sex=sex, geo=geo_obj, lang=lang)
            # Create Task object
            task_obj = Task(
                modality=modality,
                domain=domain,
                persona=persona_obj,
                lang=lang,
                internal_system_prompt=self.internal_system_prompts.get(lang, ""),
                external_system_prompt=self.external_system_prompts.get(lang, "")
            )
            result.append(task_obj)

        return result

    def save_tasks(self, output_dir: str = 'data/tasks', langs: Optional[List[str]] = None):
        """
        Save the country_specific_dict to JSON files, one per language.

        Args:
            output_dir (str): The directory where the task files will be saved.
                              Defaults to 'data/tasks'.
        """

        os.makedirs(output_dir, exist_ok=True)
        for lang, tasks in self.country_specific_dict.items():
            if langs and lang not in langs:
                continue
            data_to_save = [asdict(task) for task in tasks]
            file_path = os.path.join(output_dir, f'{lang}_tasks.json')
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
                printv(f"Tasks for language '{lang}' saved to '{file_path}'", verbosity=self.verbosity)
            except IOError as e:
                printv(f"Failed to save tasks for language '{lang}' to '{file_path}': {e}", verbosity=self.verbosity)


    def load_tasks(self, input_dir: str = 'data/tasks'):
        """
        Load the tasks from JSON files back into the country_specific_dict.

        Args:
            input_dir (str): The directory from where the task files will be loaded.
                             Defaults to 'data/tasks'.
        """
        for lang in self.langs:
            file_path = os.path.join(input_dir, f'{lang}_tasks.json')
            if not os.path.exists(file_path):
                printv(f"No task file found for language '{lang}' at '{file_path}'", verbosity=self.verbosity)
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_loaded = json.load(f)
                tasks = [self._dict_to_task(task_dict) for task_dict in data_loaded]
                self.country_specific_dict[lang] = tasks
                printv(f"Tasks for language '{lang}' loaded from '{file_path}'", verbosity=self.verbosity)
            except (IOError, json.JSONDecodeError) as e:
                printv(f"Failed to load tasks for language '{lang}' from '{file_path}': {e}", verbosity=self.verbosity)

    def _dict_to_task(self, data: Dict[str, Any]) -> Task:
        """
        Helper method to convert a dictionary back into a Task dataclass instance.

        Args:
            data (Dict[str, Any]): The dictionary representation of the Task.

        Returns:
            Task: The reconstructed Task instance.
        """
        # Reconstruct Geo object
        geo_data = data['persona']['geo']
        geo_obj = Geo(**geo_data)

        # Reconstruct Persona object
        persona_data = data['persona']
        persona_data['geo'] = geo_obj
        persona_obj = Persona(**persona_data)

        # Reconstruct Task object
        data['persona'] = persona_obj
        task_fields = {field.name for field in fields(Task)}
        task_init_data = {key: data[key] for key in data if key in task_fields}
        task_obj = Task(**task_init_data)

        return task_obj
    

def main():
    pp = PromptProtocol(verbosity=1)
    pp.save_tasks()
    tasks = pp.country_specific_dict['ru']
    print(f"Total tasks generated: {len(tasks)}")
    sample_tasks = tasks[:3]
    for task in sample_tasks:
        print(f"Task ID: {task.task_id}")
        print(task.prompt_intent_generation())
        print()

    pp.load_tasks()
    tasks = pp.country_specific_dict['ru']
    print(f"Total tasks loaded: {len(tasks)}")


if __name__ == "__main__":
    main()