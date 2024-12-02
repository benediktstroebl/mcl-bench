from litellm import completion 
from tqdm import tqdm
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict, fields
import os
import json
from datetime import datetime
from collections import defaultdict 
from typing import Dict, Any
import base64 

load_dotenv()

from helpers import load_yaml, json_to_dict
from tasks import Task, Persona, Geo

@dataclass 
class JudgeResponses: 
    judge_model_name: str = field(default='gpt-4o')
    base_path: str = field(default='output')
    system_prompt: str = field(default=None)
    system_prompt_path: str = field(default="data/prompts/llm_as_a_judge_prompt.yaml")

    images_path: str = 'output/dall-e-3/'

    def __post_init__(self):
        print('loaded') 
        self._load_system_prompt()

    def _load_system_prompt(self):
        self.system_prompt=load_yaml(self.system_prompt_path)['en']

    def generate_verdict(self, history):
        prompt = self.system_prompt
        prompt += self._convert_chat_history_to_messages(history)
        task = self._dict_to_task(history['task'])
        prompt += "==============================="
        prompt += "\nUSERS SYSTEM PROMPT: \n"
        prompt += task.prompt_external()
        prompt += "==============================="
        prompt += "Here is the user's optimal response: \n"
        prompt += task.optimal_response if task.optimal_response else "No optimal response provided."
        prompt += "==============================="
        prompt += "Here is the cultural nuances the user was thinking about for the generation: \n"
        prompt += task.cultural_nuances if task.cultural_nuances else "No cultural nuances provided."
        prompt += "==============================="
        images = history.get('image_files', [])
        messages = [
            {"role": "user", "content": prompt}
        ]
        if images:
            all_imgs = []
            for img in images: 
                base64img = self.encode_image(os.path.join(self.images_path, img))
                all_imgs.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{base64img}"}})
            
            messages[0]['content'] = [{'type': 'text', 'text': prompt}] + all_imgs
        response = completion(
            model=self.judge_model_name,
            messages=messages,
            temperature=1,
            response_format={ "type": "json_object" },
        )
        return response.choices[0].message.content
    
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
    
    @staticmethod   
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _convert_chat_history_to_messages(self, data):
        """Formats messages into a nicely readable string with role mappings."""
        role_mapping = {'user': 'User', 'assistant': 'AI Assistant'}
        formatted_messages = []
        
        for message in data.get('messages', []):
            role = role_mapping.get(message['role'], message['role'])
            content = message['content']
            formatted_messages.append(f"{role}:\n{content}\n")
        
        return "\n".join(formatted_messages)

    def load_latest_output(self, model_name):
        base_path = self.base_path
        model_path = os.path.join(base_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")
        
        # Get all subdirectories in the model path
        timestamp_dirs = [
            d for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamped directories found in '{model_path}'.")
        
        # Sort directories by timestamp (latest first)
        latest_dir = sorted(timestamp_dirs, reverse=True)[0]
        latest_path = os.path.join(model_path, latest_dir)
        
        output_dirs = ["intent_only", "with_agent", "with_persona"]
        data = defaultdict(list)
        for subdir in output_dirs:
            subdir_path = os.path.join(latest_path, subdir)
            if os.path.exists(subdir_path):
                # Load the first JSON file found in this subdirectory
                json_files = [
                    f for f in os.listdir(subdir_path)
                    if f.endswith(".json")
                ]
                if json_files:
                    for file in json_files:
                        latest_file_path = os.path.join(subdir_path, file)
                        with open(latest_file_path, "r") as f:
                            data[subdir].append(json.load(f))
        
        return data
    
    def generate_all_verdicts(self, data):
        save_path = os.path.join(self.base_path, 'verdict', self.judge_model_name)

        for key, value in data.items():
            save_path_temp = os.path.join(save_path, key)
            os.makedirs(save_path_temp, exist_ok=True)
            for _, item in enumerate(value):
                idx = item['task_id']
                save_path_temp2 = os.path.join(save_path_temp, f"{idx}.json")
                if os.path.exists(save_path_temp2):
                    continue
                try: 
                    resp = json_to_dict(self.generate_verdict(item))
                    item['faithfulness'] = resp['faithfulness']
                    item['stereotypicality'] = resp['stereotypicality']
                    item['helpfulness'] = resp['helpfulness']
                except:
                    item['faithfulness'] = "Error"
                    item['stereotypicality'] = "Error"
                    item['helpfulness'] = "Error"
                item['task']['persona'] = asdict(item['task']['persona'])

                with open(save_path_temp2, "w") as f:
                    json.dump(item, f, indent=4)

    
def main():
    jr = JudgeResponses()
    data = jr.load_latest_output('gpt-4o')
    # print(data['with_persona'][1])
    verdict = jr.generate_verdict(data['with_persona'][1])
    # print(verdict)
    jr.generate_all_verdicts(data)

if __name__ == '__main__':
    main()