from os import getenv
from typing import Optional, Literal

from phi.tools import Toolkit
from phi.utils.log import logger
import uuid

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("`openai` not installed. Please install using `pip install openai`")


class DalleTool(Toolkit):
    def __init__(
        self,
        model: str = "dall-e-3",
        n: int = 1,
        size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        response_format: Literal["url", "base64"] = "b64_json",
        api_key: Optional[str] = None,
    ):
        super().__init__(name="dalle")

        self.model = model
        self.n = n
        self.size = size
        self.quality = quality
        self.style = style
        self.response_format = response_format
        self.api_key = api_key or getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable.")

        self.register(self.generate_image)

    def generate_image(self, prompt: str) -> str:
        """Use this function to generate an image given a prompt.

        Args:
            prompt (str): A text description of the desired image.

        Returns:
            str: The URL of the generated image, or an error message.
        """
        if not self.api_key:
            return "Please set the OPENAI_API_KEY"

        try:
            client = OpenAI(api_key=self.api_key)
            logger.info(f"Generating image for prompt: {prompt}")
            response = client.images.generate(
                prompt=prompt,
                model=self.model,
                n=self.n,
                quality=self.quality,
                response_format=self.response_format,
                size=self.size,
                style=self.style,
            )
            
            # write b64 image under unique id and return id
            image_id = str(uuid.uuid4())
            image_path = f"IMAGE_{image_id}.png"
            with open(image_path, "wb") as f:
                # Decode base64 string to bytes before writing
                import base64
                image_bytes = base64.b64decode(response.data[0].b64_json)
                f.write(image_bytes)
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return f"Error: {e}"
