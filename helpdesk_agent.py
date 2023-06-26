import openai
import os
from numpy import dot

_WEBSITE = os.environ['THIS_WEBSITE']
_ENDPOINT = os.environ['THIS_ENDPOINT']

class HelpdeskAgent:
    def __init__(self, organization, api_key, model):
        self.organization = organization
        self.api_key = api_key
        self.model = model

        openai.api_key = self.api_key
        self.messages = []

    def embeddings(self, user_ask_question):
        response = openai.Embedding.create(
            model="text-embedding-ada-002", input=user_ask_question)
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def similarity(self, user_ask_question, cars):
        hypothetical_answer_embedding = self.embeddings(user_ask_question)[0]
        car_embeddings = self.embeddings(
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

    def ask_answer(self, question, answer, cars):
        ANSWER_INPUT = f"""
        Generate an answer to the user's question based on the given search results. 
        TOP_RESULTS: {self.similarity(answer, cars)}
        USER_QUESTION: {question}

        Include as much information from {_WEBSITE} as possible in the answer with bahasa. Reference the relevant search result URL as markdown links and individual results. 
        """

        self.messages.append({"role": "user", "content": ANSWER_INPUT})
        for resp in openai.ChatCompletion.create(model=self.model,
                                                 messages=self.messages,
                                                 temperature=0.5,
                                                 stream=True):
            chunk = resp.choices[0].delta.get("content", "")
            if chunk:
                yield '%s' % chunk
                