import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient
from litellm import acompletion
from dotenv import load_dotenv
from prompts.persona_prompt import get_persona_prompt
from pydantic import BaseModel
from autogen_agentchat.messages import TextMessage, ToolCallMessage, ToolCallResultMessage
from autogen_core.base import CancellationToken
import litellm
import os
from src.tasks import Persona, Geo, Task
from dalle_tool import DalleTool
from typing import List, Dict
from openai import OpenAI
import uuid
from openai import AsyncOpenAI

import os





load_dotenv()

# Add these constants at the top of the file after imports
DEFAULT_MODEL = "gpt-4o-mini"

# Create a class to hold the current configuration
class AgentConfig:
    def __init__(self):
        self.user_prompt = None
        self.model = DEFAULT_MODEL

# Create a global instance
config = AgentConfig()


async def ask_user(question: str) -> str:
    if config.user_prompt is None:
        raise ValueError("Persona not set. Please call set_persona() before using ask_user")


    response = await acompletion(
        model=config.model,
        messages=[
            {"role": "system", "content": config.user_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


async def generate_image(prompt: str) -> str:

    try:
        client = AsyncOpenAI()
        response = await client.images.generate(
            prompt=prompt,
            model="dall-e-3",
            n=1,
            quality="standard",
            response_format="b64_json",
            size="1024x1024",
            style="vivid",
        )
        
        # write b64 image under unique id and return id
        image_id = str(uuid.uuid4())
        
        # create output directory if it doesn't exist
        os.makedirs("output/dall-e-3", exist_ok=True)
        
        image_path = f"output/dall-e-3/IMAGE_{image_id}.png"
        with open(image_path, "wb") as f:
            # Decode base64 string to bytes before writing
            import base64
            image_bytes = base64.b64decode(response.data[0].b64_json)
            f.write(image_bytes)
        return f"IMAGE_{image_id}.png"
        
    except Exception as e:
        return f"Error: {e}"
    


def set_persona(prompt: str):    
    config.user_prompt = prompt

async def run_agent_evaluation(task: Task) -> List[Dict]:
    """Run agent evaluation with a given task."""
    # Set the persona for the current evaluation
    
    assistant_agent = AssistantAgent(
        name="assistant",
        # llm_config={
        #     "config_list": [
        #         {
        #             "model": config.model,
        #             "api_key": os.getenv("OPENAI_API_KEY") if "gpt" in config.model else os.getenv("ANTHROPIC_API_KEY"),
        #             "api_type": "openai" if "gpt" in config.model else "anthropic"
        #         }
        #     ]
        # } if "gpt" in config.model else None,
        model_client=OpenAIChatCompletionClient(
            model=config.model
        ) if "gpt" in config.model else None,
        tools=[ask_user, generate_image],
        # system_message="You are a helpful general-purpose assistant that can answer questions and generate images. You can also ask the user for more information if you deem it necessary if you are unsure about the cultural nuances of the task. Do not ask whether the user needs any further assistance etc. when you are done. DO NOT ASK MORE THAN 3 QUESTIONS TO THE USER OVERALL AND ONLY ASK 1 QUESTION AT A TIME.",
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")
    
    set_persona(task.prompt_external())

    # Define a team
    agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    
    prompt = f"""
    Do never ask more than a total of 3 follow up questions to the user.
    
    Question from the user:
    {task.intent}
    """
    
    stream = agent_team.run_stream(task=task.intent)

    task_result = await Console(stream)
    
    # Convert messages to litellm format
    messages = []
    print(task_result.messages)
    for m in task_result.messages:
        if isinstance(m, TextMessage):
            messages.append({"role": m.source, "content": m.content})
        elif isinstance(m, ToolCallMessage):
            if m.content[0].name == "ask_user":
                messages.append({"role": "assistant", "content": f"{m.content[0].arguments}"})
            elif m.content[0].name == "generate_image":
                messages.append({"role": "assistant", "content": f"Generate Image with prompt: {m.content[0].arguments}"})
        elif isinstance(m, ToolCallResultMessage):
            if not "IMAGE_" in m.content[0].content:
                messages.append({"role": "user", "content": f"{m.content[0].content}"})
    

    return messages



if __name__ == "__main__":
    
    import asyncio
    from typing import Dict, Any
    import json
    
    # Example task
    example_task = {
        "modality": "text2text",
        "lang": "en", 
        "domain": "moral discipline and social values",
        "persona": {
            "age": 44,
            "sex": "female",
            "geo": {
                "country": "United States",
                "state": "California", 
                "city": "San Francisco"
            },
            "political_leaning": None,
            "lang": "en"
        },
        "intent": "What should I wear to the wedding next week?",
        "optimal_response": None,
        "cultural_nuances": None,
        "external_files": [],
        "task_id": 141
    }

    # Create Task object from dict
    from dataclasses import dataclass
    
    @dataclass
    class Task:
        intent: str
        persona: Dict[str, Any]
        external_system_prompt: str = "You are roleplaying as a {age} year old {sex} and lives in {location}.\nWhen answering questions, consider your background:\n- Your age and generational perspective\n- Your lifestyle living in {location}\n- Your gender identity and how it might influence your viewpoint\nRespond naturally as this persona would speak, incorporating relevant personal experiences and viewpoints that would be authentic to this character.\nRemember to:\n1. Stay consistent with the persona's background\n2. Use appropriate language and tone for your age and profession\n3. Reference relevant local context from {location} when applicable\n4. If you are asked questions you dont know the answer to, say so.\n"

        def prompt_external(self):
            location = f"{self.persona['geo']['city']}, {self.persona['geo']['state']}"
            return self.external_system_prompt.format(
                age=self.persona['age'],
                sex=self.persona['sex'],
                location=location
            )

    task = Task(
        intent=example_task["intent"],
        persona=example_task["persona"]
    )

    # Run agent
    async def main():
        messages = await run_agent_evaluation(task)
        print(json.dumps(messages, indent=2))

    asyncio.run(main())
    