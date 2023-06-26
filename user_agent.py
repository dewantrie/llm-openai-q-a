import openai
import json

class UserAgent:
    def __init__(self, organization, api_key, model):
        self.organization = organization
        self.api_key = api_key
        self.model = model
        
        openai.api_key = self.api_key
        self.messages = []

    def add_to_question(self, question):
        HA_INPUT = f"""
        Generate a hypothetical answer to the user's question with bahasa. This answer will be used to rank search results. 
        Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
        like MODEL VARIANT did something, or MODEL VARIANT said something at PLACE. 

        User question: {question}

        Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
        """

        self.messages.append(
            {"role": "system", "content": "Output only valid JSON"})
        self.messages.append({"role": "user", "content": HA_INPUT})

    def ask_question(self, question):
        self.add_to_question(question)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.5,
        )

        text = completion.choices[0].message.content
        return question, json.loads(text)['hypotheticalAnswer']
