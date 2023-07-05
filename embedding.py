import openai
from typing import List, Any

class Embedding:
    
    @staticmethod
    def embed(text: Any) -> List[float]:
        response = openai.Embedding.create(
            model='text-embedding-ada-002', input=text)
        embeddings = [data.embedding for data in response.data]
        return embeddings