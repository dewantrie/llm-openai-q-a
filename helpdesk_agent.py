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

    def rerank(self, user_ask_question, cars):
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
        Instructions:
        Generate an answer to the user's question based on the given search results.
        The result should be in English.

        Context:
        Include as much information {_WEBSITE} as possible in the answer.
        Reference the relevant search result.

        Input Data:
        TOP_RESULTS: {self.rerank(answer, cars)}
        USER_QUESTION: {question}

        Output Indicator:
        1. The output should include a detailed cars specifying.
        2. The output should be numbering.
        3. The output should be split by Price, Description, and Link.
        4. The link output must be combined with a slug like the following {_ENDPOINT}slug.
        5. The output should be as markdown.
        """

        self.messages.append({"role": "user", "content": ANSWER_INPUT})
        for resp in openai.ChatCompletion.create(model=self.model,
                                                 messages=self.messages,
                                                 temperature=0.5,
                                                 stream=True):
            chunk = resp.choices[0].delta.get("content", "")
            if chunk:
                yield '%s' % chunk
                
