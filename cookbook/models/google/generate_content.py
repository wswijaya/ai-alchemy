import os
import asyncio
from dotenv import load_dotenv

from typing import List

from google import genai
from google.genai import types

load_dotenv(dotenv_path="../../../.env")

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def query_gemini(query="Hello..."):
    """
    Queries using Gemini
    Args:
        query (str): The natural language question about the code.

    Returns:
        str: The answer from the LLM, or None on error.
    """
    try:
        messages = [
                f"Question: {query}",
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction='You are a super helpful assistant that can answer any questions.',
                temperature=0.5,
            ),
        )

        return response.text
    except Exception as e:
        print(f"Error querying the code with Google API for Gemini: {e}")
        return None


async def main():
    """Main function to load code, process it, analyze it, and query it."""
    # Example queries
    queries = [
        "What is the weather in Singapore today?",
        "Write haiku about the weather",
        "List 5 eating places near Orchard",
    ]
    for query in queries:
        answer = query_gemini(query)
        if answer:
            print(f"\nQuery: {query}")
            print(f"LLM Answer: {answer}")
        else:
            print(f"\nQuery: {query}")
            print("LLM could not answer the question.")
    
if __name__ == "__main__":
    asyncio.run(main())
