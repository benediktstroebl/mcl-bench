import os
from litellm import completion
from tqdm import tqdm
from dotenv import load_dotenv
from multiprocessing.dummy import Pool as ThreadPool
import functools

load_dotenv()

from tasks import PromptProtocol

def process_task(model_name, lang, task):
    """Process a single task to generate intents."""
    prompt = task.prompt_intent_generation()
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = completion(
        model=model_name,
        messages=messages,
        temperature=1,
        response_format={"type": "json_object"},
    )
    print(response.choices[0].message.content)
    task.update_intent(response.choices[0].message.content)

def process_language(model_name, lang):
    """Process all tasks for a specific language."""
    pp = PromptProtocol(verbosity=1)
    pp.load_tasks()
    tasks = pp.country_specific_dict[lang]

    if os.path.exists(f'data/tasks_with_intents/{model_name}/{lang}.json'):
        print(f"Tasks with intents for {lang} already exist, skipping...")
        return

    with ThreadPool(25) as pool:
        pool.map(functools.partial(process_task, model_name, lang), tasks)
    
    pp.save_tasks(f'data/tasks_with_intents/{model_name}', langs=[lang])

def main():
    model_name = 'gpt-4o'
    langs = ['de', 'ru', 'en']  # Add more languages if needed

    for lang in tqdm(langs, desc="Generating intents"):
        process_language(model_name, lang)

if __name__ == "__main__":
    main()