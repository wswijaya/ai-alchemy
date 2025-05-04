import os
import asyncio
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv(dotenv_path="../../../.env")

client = OpenAI()

def query_openai(query="Hello..."):
    """
    Queries using OpenAI
    Args:
        query (str): The natural language question about the code.

    Returns:
        str: The answer from the LLM, or None on error.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a super helpful assistant that can answer any questions.",
            },
            {
                "role": "user",
                "content": f"Question: {query}",
            },
        ]

        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL'),  # Replace with your Azure deployment name for chat models
            messages=messages,
            temperature=0.5,  # Adjust for desired randomness
        )

        return response.output_text
    except Exception as e:
        print(f"Error querying the code with Azure OpenAI: {e}")
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
        answer = query_openai(query)
        if answer:
            print(f"\nQuery: {query}")
            print(f"LLM Answer: {answer}")
        else:
            print(f"\nQuery: {query}")
            print("LLM could not answer the question.")
                
if __name__ == "__main__":
    asyncio.run(main())