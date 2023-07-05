import openai
import os

from typing import List, Any
from numpy import dot
from prompt.reader import PromptReader
from embedding import Embedding

_WEBSITE = os.environ['THIS_WEBSITE']
_ENDPOINT = os.environ['THIS_ENDPOINT']

class HelpdeskAgent:
    def __init__(self, organization: str, api_key: str, model: str) -> None:
        self.organization = organization
        self.api_key = api_key
        self.model = model

        openai.api_key = self.api_key
        self.messages = []

    def rerank(self, user_ask_question: str, cars: List[Any]) -> List[Any]:
        hypothetical_answer_embedding = Embedding.embed(user_ask_question)[0]
        car_embeddings = Embedding.embed(
            [
                f"{row['description']} {row['price']} {row['slug']} "
                for row in cars
            ]
        )

        cosine_similarities = []
        for car_embedding in car_embeddings:
            cosine_similarities.append(dot(hypothetical_answer_embedding, car_embedding))

        scored_cars = zip(cars, cosine_similarities)
        sorted_cars = sorted(scored_cars, key=lambda x: x[1], reverse=True)

        formatted_top_results = [
            {
                "description": car["description"],
                "url": f'{_ENDPOINT}{car["slug"]}',
                "price": car["price"],
            }
            for car, _score in sorted_cars[0:5]
        ]

        return formatted_top_results

    def ask_answer(self, question: str, answer: str, cars: List[Any]):
        read = PromptReader.read_agent_prompt(__file__, 'instruct.txt')
        prompt = PromptReader.clean_prompt(read)
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{_WEBSITE}', _WEBSITE)
        prompt = prompt.replace('{_ENDPOINT}', _ENDPOINT)
        prompt = prompt.replace('{hypothetical}', str(self.rerank(answer, cars)))

        self.messages.append({"role": "user", "content": prompt})
        for resp in openai.ChatCompletion.create(model=self.model,
                                                 messages=self.messages,
                                                 temperature=0.5,
                                                 stream=True):
            chunk = resp.choices[0].delta.get("content", "")
            if chunk:
                yield '%s' % chunk
                
