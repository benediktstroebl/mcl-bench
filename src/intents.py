import os 
from litellm import completion
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from tasks import PromptProtocol

def main():
    model_name = 'gpt-4o'
    # langs = ['ru', 'de', 'en']
    langs = ['en']
    pp = PromptProtocol(verbosity=1)
    pp.load_tasks()
    
    for lang in tqdm(langs, desc="Generating intents"):
        tasks = pp.country_specific_dict[lang]
        if os.path.exists(f'data/tasks_with_intents/{model_name}/{lang}.json'):
            print(f"Tasks with intents for {lang} already exist, skipping...")
            continue
        for task in tqdm(tasks, leave=False, desc=f"Generating intents for {lang}"):
            prompt = task.prompt_intent_generation()
            print(prompt)
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = completion(
                model=model_name,
                messages=messages,
                temperature=1,
                response_format={ "type": "json_object" },

            )
            print(response.choices[0].message.content)
            task.update_intent(response)
    
    pp.save_tasks(f'data/tasks_with_intents/{model_name}')

if __name__ == "__main__":
    main()