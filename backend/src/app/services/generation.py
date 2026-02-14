from typing import List, Optional
from fastapi import Request
from litellm import completion
from src.app.v1.schema import ChatResponse

class GenerationService:
    def __init__(self, request: Request):
        self.config = request.app.state.config
        self.api_key = self.config.llm_api_key
        self.model = self.config.generation_model

    def generate_answer(self, question: str, context: List[str]) -> str:
        """
        Generates an answer using the configured LLM based on the provided context.
        """
        context_str = "\n".join(context)
        prompt = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context_str}

Question: {question}
Helpful Answer:"""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key
        )

        return response.choices[0].message.content
