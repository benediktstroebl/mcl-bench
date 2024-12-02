import json
import os
from typing import List, Dict, Optional
from dataclasses import asdict
import asyncio
from pathlib import Path

from tasks import Task, Persona, Geo
from agent import run_agent_evaluation
from litellm import completion

class EvaluationPipeline:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def evaluate_intent_only(self, task: Task) -> List[Dict]:
        """Setting 1: Evaluate with just the intent."""
        messages = [
            {"role": "user", "content": task.intent}
        ]
        
        response = self._get_model_response(messages)
        messages.append({"role": "assistant", "content": response})
        
        self._save_results("intent_only.jsonl", messages, task)
        return messages

    def evaluate_with_persona(self, task: Task, exclude_attributes: Optional[List[str]] = None) -> List[Dict]:
        """Setting 2: Evaluate with intent and persona context."""
        # Filter out excluded attributes from persona
        persona_dict = asdict(task.persona)
        if exclude_attributes:
            for attr in exclude_attributes:
                if attr in persona_dict:
                    del persona_dict[attr]

        context = f"Given the following persona:\n{task.persona.generate_prompt()}\n\nRespond to: {task.intent}"
        
        messages = [
            {"role": "user", "content": context}
        ]
        
        response = self._get_model_response(messages)
        messages.append({"role": "assistant", "content": response})
        
        self._save_results("with_persona.jsonl", messages, task)
        return messages

    async def evaluate_with_agent(self, task: Task) -> List[Dict]:
        """Setting 3: Evaluate using the agent with user interaction."""
        messages = await run_agent_evaluation(task)
        self._save_results("with_agent.jsonl", messages, task)
        return messages

    def _get_model_response(self, messages: List[Dict]) -> str:
        """Get response from the model."""
        response = completion(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    def _save_results(self, filename: str, messages: List[Dict], task: Task):
        """Save results to jsonl file."""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "a") as f:
            result = {
                "task": asdict(task),
                "messages": messages
            }
            f.write(json.dumps(result) + "\n")

async def main():
    # Create a synthetic task for testing
    geo = Geo(country="USA", state="NY", city="New York")
    persona = Persona(age=25, sex="female", geo=geo, political_leaning="liberal")
    task = Task(
        modality="text2text",
        lang="en",
        domain="fashion",
        persona=persona,
        intent="What should I wear to a summer wedding?",
        external_files=[],
        internal_system_prompt="You are a helpful fashion advisor.",
        external_system_prompt="Provide fashion advice considering the context: {question}"
    )

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline()

    # Run all three evaluation settings
    pipeline.evaluate_intent_only(task)
    pipeline.evaluate_with_persona(task, exclude_attributes=["political_leaning"])
    # await pipeline.evaluate_with_agent(task)

if __name__ == "__main__":
    asyncio.run(main()) 