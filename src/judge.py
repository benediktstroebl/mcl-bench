from litellm import completion 
from tqdm import tqdm
from dotenv import load_dotenv
from dataclasses import dataclass 

load_dotenv()

from tasks import PromptProtocol

@dataclass 
class JudgeResponses: 

    