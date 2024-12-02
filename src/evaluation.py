import json
import os
from typing import List, Dict, Optional
from dataclasses import asdict
import asyncio
from pathlib import Path
from datetime import datetime

from tasks import Task, Persona, Geo
from agent import run_agent_evaluation
from litellm import completion
from prompts.persona_context import get_persona_context_prompt

class EvaluationPipeline:
    def __init__(self, output_dir: str = "output", model: str = "gpt-4o-mini"):
        self.base_dir = output_dir
        self.model = model
        self._setup_directories()

    def _setup_directories(self):
        """Create directory structure for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.base_dir) / self.model / timestamp
        
        # Create directories for each evaluation setting
        self.intent_dir = self.output_dir / "intent_only"
        self.persona_dir = self.output_dir / "with_persona"
        self.agent_dir = self.output_dir / "with_agent"

        for dir_path in [self.intent_dir, self.persona_dir, self.agent_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, task: Task) -> str:
        """Generate a unique filename for the task result."""
        # Create a short identifier from task properties
        task_id = f"{task.domain}_{task.modality}_{task.lang}"
        return f"{task_id}.json"

    def evaluate_intent_only(self, task: Task) -> List[Dict]:
        """Setting 1: Evaluate with just the intent."""
        messages = [
            {"role": "user", "content": task.intent}
        ]
        
        response = self._get_model_response(messages)
        messages.append({"role": "assistant", "content": response})
        
        self._save_result(self.intent_dir, task, messages)
        return messages

    def evaluate_with_persona(self, task: Task, exclude_attributes: Optional[List[str]] = None) -> List[Dict]:
        """Setting 2: Evaluate with intent and persona context."""
        # Filter out excluded attributes from persona if specified
        if exclude_attributes:
            # Create a copy of the persona to avoid modifying the original
            persona_dict = asdict(task.persona)
            for attr in exclude_attributes:
                if attr in persona_dict:
                    del persona_dict[attr]
            # Recreate persona object with filtered attributes
            # First recreate the Geo object
            geo_data = persona_dict.get('geo')
            geo = Geo(**geo_data) if geo_data else task.persona.geo
            # Then create the Persona with the proper Geo object
            filtered_persona = Persona(
                age=persona_dict.get('age', task.persona.age),
                sex=persona_dict.get('sex', task.persona.sex),
                geo=geo,
                political_leaning=persona_dict.get('political_leaning', task.persona.political_leaning)
            )
        else:
            filtered_persona = task.persona

        # Get the formatted prompt from the prompt template
        context = get_persona_context_prompt(filtered_persona, task.intent)
        
        messages = [
            {"role": "user", "content": context}
        ]
        
        response = self._get_model_response(messages)
        messages.append({"role": "assistant", "content": response})
        
        self._save_result(self.persona_dir, task, messages)
        return messages

    async def evaluate_with_agent(self, task: Task) -> List[Dict]:
        """Setting 3: Evaluate using the agent with user interaction."""
        messages = await run_agent_evaluation(task)
        self._save_result(self.agent_dir, task, messages)
        return messages

    def _get_model_response(self, messages: List[Dict]) -> str:
        """Get response from the model."""
        response = completion(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def _save_result(self, directory: Path, task: Task, messages: List[Dict]):
        """Save result as a JSON file."""
        filename = self._generate_filename(task)
        output_path = directory / filename
        
        result = {
            "task": asdict(task),
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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
    pipeline = EvaluationPipeline(model="gpt-4o-mini")

    # Run all three evaluation settings
    pipeline.evaluate_intent_only(task)
    pipeline.evaluate_with_persona(task, exclude_attributes=["political_leaning"])
    await pipeline.evaluate_with_agent(task)

if __name__ == "__main__":
    asyncio.run(main()) 