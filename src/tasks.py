from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional
from itertools import product

from helpers import load_yaml, printv


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

    def __post_init__(self):
        pass 
        # if self.political_leaning not in ['conservative', 'liberal', None]:
        #     raise ValueError("Invalid political leaning")

    def generate_prompt(self) -> str:
        return (
            f"\n\tAge: {self.age}"
            f"\n\tSex: {self.sex}"
            f"\n\tGeographic information:\n{self.geo.generate_prompt()}"
            f"\n\tPolitical leaning: {self.political_leaning}"
        )


@dataclass
class Task:
    modality: Literal['text2image', 'image2text', 'text2text', 'image2image'] = field()
    lang: str = field(default=None)
    domain: str = field(default=None)
    persona: Persona = field(default=None)
    intent: str = field(default=None)
    external_files: List[str] = field(default_factory=list)
    internal_system_prompt: str = field(default=None)
    external_system_prompt: str = field(default=None)

    def __post_init__(self):
        if self.modality == 'image2image':
            raise NotImplementedError("image2image modality is not supported yet")

    def update_intent(self, intent: str):
        self.intent = intent

    def update_external_files(self, external_files: List[str]):
        self.external_files = external_files

    def prompt_intent_generation(self) -> str:
        # Prompt for generating intents
        prompt = (
            f"{self.internal_system_prompt}"
            f"Modality: {self.modality}\n"
            f"Domain: {self.domain}\n"
            f"Persona: {self.persona.generate_prompt()}"
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
        prompt += (
            f"Domain: {self.domain}\n"
            f"Persona:\n{self.persona.generate_prompt()}\n"
            f"Intent: {self.intent}\n"
            f"Image file: {self.external_files}"
        )
        return prompt

@dataclass
class PromptProtocol:
    langs: List[str] = field(default_factory=lambda: ['en', 'ru', 'de'])
    country_specific_dict: Dict[str, List[Task]] = field(default_factory=dict)
    country_info_path: str = field(default='data/prompts/{}.yaml')

    # Task parameters
    modalities: List[Literal['text2image', 'image2text', 'text2text']] = field(
        default_factory=lambda: ['text2image', 'image2text', 'text2text']
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
        political_leanings = persona_data.get("political_leaning", [None])
        ages = self.age_list

        geo_combinations = [
            (country, region, city)
            for country, regions in data.get("Geo", {}).items()
            for region, cities in regions.items()
            for city in cities
        ]

        modalities = self.modalities
        combinations = product(modalities, domains, sexes, ages, political_leanings, geo_combinations)
        result = []

        for modality, domain, sex, age, pl, geo in combinations:
            # Create Geo object
            geo_obj = Geo(country=geo[0], state=geo[1], city=geo[2])
            # Create Persona object
            persona_obj = Persona(age=age, sex=sex, geo=geo_obj, political_leaning=pl)
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


def main():
    pp = PromptProtocol(verbosity=1)
    tasks = pp.country_specific_dict['ru']
    print(f"Total tasks generated: {len(tasks)}")
    sample_tasks = tasks[:3]
    for task in sample_tasks:
        print(task.prompt_intent_generation())
        print()


if __name__ == "__main__":
    main()