from openai import OpenAI
from pydantic import BaseModel
import os

class ModelGPTWrapper:
    def __init__(self, model_name):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Assuming OpenAI API client is initialized this way
        self.model_name = model_name

    def invoke(self, messages, temperature):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # temperature=0  # Assuming deterministic output
            temperature=temperature
        )
        return SimpleResponse(content=response.choices[0].message.content)

    def with_structured_output(self, return_format: BaseModel):
        return StructuredOutputModel(self, return_format)

class SimpleResponse:
    def __init__(self, content):
        self.content = content

    def dict(self):
        return {"content": self.content}

class StructuredOutputModel:
    def __init__(self, model_wrapper, return_format: BaseModel):
        self.model_wrapper = model_wrapper
        self.return_format = return_format

    def invoke(self, messages, temperature):
        # Here, we assume OpenAI function calling or JSON mode is used for structured output.
        functions = [{
            "name": "structured_response",
            "parameters": self.return_format.schema()
        }]
        response = self.model_wrapper.client.chat.completions.create(
            model=self.model_wrapper.model_name,
            messages=messages,
            functions=functions,
            function_call={"name": "structured_response"},
            # temperature=0
            temperature=temperature
        )
        structured_data = response.choices[0].message.function_call.arguments
        # Convert to dict following the pydantic model
        return self.return_format.parse_raw(structured_data)

# Instantiate the specific model wrappers
modelGPT4o = ModelGPTWrapper("gpt-4o")
modelGPTo1 = ModelGPTWrapper("gpt-o1")
