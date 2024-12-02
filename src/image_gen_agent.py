from phi.agent import Agent
from phi.model.openai import OpenAIChat

from dalle_tool import DalleTool



def get_image_gen_agent(prompt: str) -> Agent:
    agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[DalleTool()],
    debug_mode=True,
    instructions=[
        "You are an AI agent that can generate images using DALL-E.",
        "DALL-E will return an image path.",
        "Return ONLY the path to the image in your response and nothing else.",
        ],
    )
    
    agent.print_response(prompt)
    return agent

if __name__ == "__main__":
    agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[DalleTool()],
    debug_mode=True,
    instructions=[
        "You are an AI agent that can generate images using DALL-E.",
        "DALL-E will return an image path.",
        "Return ONLY the path to the image in your response and nothing else.",
        ],
    )
    image = agent.print_response("Generate an image of a white siamese cat")
    
    print(agent.get_chat_history())