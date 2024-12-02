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
from src.tasks import Persona, Geo
from typing import List, Dict
from src.tasks import Task

load_dotenv()
        
        
persona = Persona(age=25, sex="male", geo=Geo(city="New York", state="NY", country="USA"), political_leaning="liberal")

# Define a tool
async def ask_user(question: str) -> str:
    system_prompt = get_persona_prompt(persona)
    response = await acompletion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

async def run_agent_evaluation(task: Task) -> List[Dict]:
    """Run agent evaluation with a given task."""
    # Define an agent
    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
        ),
        tools=[ask_user],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task=task.intent)

    task_result = await Console(stream)
    
    # Convert messages to litellm format
    messages = []
    for m in task_result.messages:
        if isinstance(m, TextMessage):
            messages.append({"role": m.source, "content": m.content})
        elif isinstance(m, ToolCallMessage):
            if m.content[0].name == "ask_user":
                m_call_id = m.content[0].id
                tool_result_message = None
                for m_2 in task_result.messages:
                    if isinstance(m_2, ToolCallResultMessage) and m_2.content[0].call_id == m_call_id:
                        tool_result_message = m_2
                        break
                    
                if tool_result_message:
                    messages.append({"role": "user", "content": tool_result_message.content[0].content})
    
    return messages

async def main() -> None:
    # Define an agent
    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
        ),
        tools=[ask_user],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="what should I wear to a wedding?")

    task_result = await Console(stream)
    
    return task_result


if __name__ == "__main__":

    task_result = asyncio.run(main())


    messages = []
    for m in task_result.messages:
        if isinstance(m, TextMessage):
            messages.append({"role": m.source, "content": m.content})
        elif isinstance(m, ToolCallMessage):
            if m.content[0].name == "ask_user":
                m_call_id = m.content[0].id
                tool_result_message = None
                for m_2 in task_result.messages:
                    if isinstance(m_2, ToolCallResultMessage) and m_2.content[0].call_id == m_call_id:
                        tool_result_message = m_2
                        break
                    
                
                messages.append({"role": "user", "content": tool_result_message.content[0].content})

    # Add follow-up question about location
    messages.append({"role": "user", "content": "Based on the above conversation. Where do you think the user is from? Please make you best guess and do NOT avoid to answer this question."})

    response = litellm.completion(
        model="gpt-4o",
        messages=messages
    )
    print(response.choices[0].message.content)
