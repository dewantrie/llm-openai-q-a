import openai
import json

from prompt.reader import PromptReader
from typing import Tuple, Any

class UserAgent:
    def __init__(self, organization: str, api_key: str, model: str) -> None:
        self.organization = organization
        self.api_key = api_key
        self.model = model
        
        openai.api_key = self.api_key
        self.messages = []

    def add_to_question(self, question: str) -> None:
        read = PromptReader.read_agent_prompt(__file__, 'hypothetical.txt')
        prompt = PromptReader.clean_prompt(read).replace('{question}', question)

        self.messages.append(
            {"role": "system", "content": "Output only valid JSON"})
        self.messages.append({"role": "user", "content": prompt})

    def ask_question(self, question: str) -> Tuple[str, Any]:
        self.add_to_question(question)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.5,
        )

        text = completion.choices[0].message.content
        return question, json.loads(text)['hypothetical']
