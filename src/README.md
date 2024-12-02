# Running the code

`tasks.py` contains all the key data generating information. 

in `data/prompts` you'll find all the prompts. 
of importance:
- `{lang}.yaml` contains all the leafs for generating
- `system_prompt_intent.yaml` contains the system prompt for generating the intents
- `system_prompt_user.yaml` contains the system prompt for the user.

`intents.py` and `intents_fast.py` contain information that's meant to fill in the intent part of tasks.
