import json
import os
import traceback
from typing import List, Dict, Optional
from dataclasses import asdict
import asyncio
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from tasks import Task, Persona, Geo
from agent import run_agent_evaluation
from litellm import completion
from image_gen_agent import get_image_gen_agent
from prompts.persona_context import get_persona_context_prompt

class EvaluationPipeline:
    def __init__(self, output_dir: str = "output", model: str = "gpt-4o-mini", max_concurrent: int = 5, resume_timestamp: str = None):
        self.base_dir = output_dir
        self.model = model
        self.max_concurrent = max_concurrent
        self.resume_timestamp = resume_timestamp
        self._setup_directories()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.completed_tasks = self._load_completed_tasks() if resume_timestamp else {
            'intent': set(),
            'persona': set(),
            'agent': set()
        }

    def _setup_directories(self):
        """Create directory structure for results."""
        if self.resume_timestamp:
            # Use existing timestamp directory
            timestamp = self.resume_timestamp
            print(f"Resuming from previous run: {timestamp}")
        else:
            # Create new timestamp directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.output_dir = Path(self.base_dir) / self.model / timestamp
        
        # Create directories for each evaluation setting
        self.intent_dir = self.output_dir / "intent_only"
        self.persona_dir = self.output_dir / "with_persona"
        self.agent_dir = self.output_dir / "with_agent"

        for dir_path in [self.intent_dir, self.persona_dir, self.agent_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_completed_tasks(self) -> Dict[str, set]:
        """Load set of completed task IDs from existing results for each evaluation type."""
        completed_tasks = {
            'intent': set(),
            'persona': set(),
            'agent': set()
        }
        
        # Map directories to their evaluation types
        dir_map = {
            self.intent_dir: 'intent',
            self.persona_dir: 'persona',
            self.agent_dir: 'agent'
        }
        
        # Check each directory for completed tasks
        for directory, eval_type in dir_map.items():
            if directory.exists():
                for file in directory.glob("*.json"):
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            if 'task_id' in data:
                                completed_tasks[eval_type].add(data['task_id'])
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        
        # Print completion status for each type
        for eval_type, completed in completed_tasks.items():
            print(f"Found {len(completed)} completed tasks for {eval_type} evaluation")
        
        return completed_tasks

    def _generate_filename(self, task: Task) -> str:
        """Generate a unique filename for the task result."""
        # Create a short identifier from task properties
        return f"{task.task_id}.json"

    def evaluate_intent_only(self, task: Task) -> List[Dict]:
        """Setting 1: Evaluate with just the intent."""
        messages = []
        
        if task.modality in ['text2image', 'image2image']:
            # Use image generation agent
            messages = get_image_gen_agent(task.intent).get_chat_history()
            messages = json.loads(messages)
            response = messages[-1]['content']
        else:
            # Use regular text completion
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
            persona_dict = asdict(task.persona)
            for attr in exclude_attributes:
                if attr in persona_dict:
                    del persona_dict[attr]
            geo_data = persona_dict.get('geo')
            geo = Geo(**geo_data) if geo_data else task.persona.geo
            filtered_persona = Persona(
                age=persona_dict.get('age', task.persona.age),
                sex=persona_dict.get('sex', task.persona.sex),
                geo=geo,
                political_leaning=persona_dict.get('political_leaning', task.persona.political_leaning)
            )
        else:
            filtered_persona = task.persona

        context = get_persona_context_prompt(filtered_persona, task.intent, task.lang)
        messages = []
        
        if task.modality in ['text2image', 'image2image']:
            # Use image generation agent with persona context
            messages = get_image_gen_agent(context).get_chat_history()
            messages = json.loads(messages)
            response = messages[-1]['content']
        else:
            # Use regular text completion
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
        
        # Determine evaluation type from directory
        if directory == self.intent_dir:
            eval_type = 'intent'
        elif directory == self.persona_dir:
            eval_type = 'persona'
        else:
            eval_type = 'agent'
        
        # Extract image filenames for image generation tasks
        image_files = []
        if task.modality in ['text2image', 'image2image']:
            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"]
                    # Look for IMAGE_ pattern in the response
                    words = content.split()
                    for word in words:
                        if word.startswith("IMAGE_") and word.endswith(".png"):
                            image_files.append(word)
        
        result = {
            "task_id": task.task_id,
            "evaluation_type": eval_type,
            "task": asdict(task),
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "image_files": image_files if image_files else None
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    async def _evaluate_task(self, task: Task, pbar: tqdm) -> None:
        """Evaluate a single task with all three methods."""
        try:
            async with self.semaphore:  # Limit concurrent executions
                pbar.write(f"\nProcessing task {task.task_id}")
                
                # Check and run intent-only evaluation
                if task.task_id not in self.completed_tasks['intent']:
                    pbar.set_description(f"Task {task.task_id} - Intent only")
                    self.evaluate_intent_only(task)
                    self.completed_tasks['intent'].add(task.task_id)
                else:
                    pbar.write(f"Skipping intent evaluation for task {task.task_id} - already completed")
                pbar.update(1)
                
                # Check and run persona-context evaluation
                if task.task_id not in self.completed_tasks['persona']:
                    pbar.set_description(f"Task {task.task_id} - Persona context")
                    self.evaluate_with_persona(task)
                    self.completed_tasks['persona'].add(task.task_id)
                else:
                    pbar.write(f"Skipping persona evaluation for task {task.task_id} - already completed")
                pbar.update(1)
                
                # Check and run agent-based evaluation
                if task.task_id not in self.completed_tasks['agent']:
                    pbar.set_description(f"Task {task.task_id} - Agent based")
                    await self.evaluate_with_agent(task)
                    self.completed_tasks['agent'].add(task.task_id)
                else:
                    pbar.write(f"Skipping agent evaluation for task {task.task_id} - already completed")
                pbar.update(1)
                
                if (task.task_id in self.completed_tasks['intent'] and 
                    task.task_id in self.completed_tasks['persona'] and 
                    task.task_id in self.completed_tasks['agent']):
                    pbar.write(f"Completed all evaluations for task {task.task_id}")
                
        except Exception as e:
            traceback.print_exc()
            pbar.write(f"Error processing task {task.task_id}: {str(e)}")
            # Save error information
            error_file = self.output_dir / f"task_{task.task_id}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task": asdict(task),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            # Update progress bar for any remaining evaluations for this task
            remaining_steps = 3 - (pbar.n % 3)
            if remaining_steps > 0:
                pbar.update(remaining_steps)

    async def evaluate_tasks_from_file(self, input_file: str) -> None:
        """Evaluate all tasks from a JSON file using all three evaluation methods."""
        print(f"Loading tasks from {input_file}")
        
        # Load tasks from JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
        
        print(f"Found {len(tasks_data)} tasks")
        print(f"Running with max {self.max_concurrent} concurrent tasks")
        
        # Create progress bar
        pbar = tqdm(total=len(tasks_data) * 3, desc="Evaluating tasks")
        
        # Create Task objects
        tasks = []
        for task_data in tasks_data:
            try:
                geo = Geo(**task_data['persona']['geo'])
                persona = Persona(
                    age=task_data['persona']['age'],
                    sex=task_data['persona']['sex'],
                    geo=geo,
                    political_leaning=task_data['persona'].get('political_leaning'),
                    lang=task_data['persona'].get('lang')
                )
                
                task = Task(
                    modality=task_data['modality'],
                    lang=task_data['lang'],
                    domain=task_data['domain'],
                    persona=persona,
                    intent=task_data['intent'],
                    external_files=task_data['external_files'],
                    internal_system_prompt=task_data['internal_system_prompt'],
                    external_system_prompt=task_data['external_system_prompt'],
                    task_id=task_data['task_id']
                )
                
                if task.intent:  # Only add tasks with intent
                    tasks.append(task)
                else:
                    pbar.write(f"Skipping task {task.task_id} - no intent specified")
                    pbar.update(3)
                    
            except Exception as e:
                pbar.write(f"Error creating task object: {str(e)}")
                pbar.update(3)
        
        # Create tasks for concurrent execution
        evaluation_tasks = [self._evaluate_task(task, pbar) for task in tasks]
        
        # Run tasks concurrently
        await asyncio.gather(*evaluation_tasks)
        
        # Close progress bar
        pbar.close()

async def main():
    # Example usage with resume functionality
    pipeline = EvaluationPipeline(
        model="gpt-4o-mini", 
        max_concurrent=3,
        resume_timestamp=None
    )
    await pipeline.evaluate_tasks_from_file("data/tasks_with_intents/gpt-4o/en_tasks.json")

if __name__ == "__main__":
    asyncio.run(main()) 