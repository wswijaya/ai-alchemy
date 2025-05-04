import os
import asyncio
from dotenv import load_dotenv

from openai import AsyncAzureOpenAI

load_dotenv(dotenv_path="../../../.env")

az_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZ_OPENAI_API_ENDPOINT'),
    api_version=os.getenv('AZ_OPENAI_API_VERSION'),
    api_key=os.getenv('AZ_OPENAI_API_KEY'),
)

async def query_az_openai(query="Hello..."):
    """
    Queries using Azure AI Service (OpenAI)
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

        response = await az_client.chat.completions.create(
            model=os.getenv('AZ_MODEL_DEPLOYMENT_NAME'),  # Replace with your Azure deployment name for chat models
            messages=messages,
            max_tokens=4096,  # Adjust as needed
            temperature=0.5,  # Adjust for desired randomness
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying the code with Azure OpenAI: {e}")
        return None
    
async def main():
    """Main function."""
    # Example queries
    queries = [
        "What is the weather in Singapore today?",
        "Write haiku about the weather",
        "List 5 eating places near Orchard",
    ]
    for query in queries:
        answer = await query_az_openai(query)
        if answer:
            print(f"\nQuery: {query}")
            print(f"LLM Answer: {answer}")
        else:
            print(f"\nQuery: {query}")
            print("LLM could not answer the question.")
                
if __name__ == "__main__":
    asyncio.run(main())