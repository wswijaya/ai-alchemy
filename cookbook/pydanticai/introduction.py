import os
import asyncio
from dotenv import load_dotenv

import requests

from openai import AsyncAzureOpenAI

from pydantic import BaseModel
from pydantic_ai import Agent 
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(dotenv_path="../../.env")

az_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZ_OPENAI_API_ENDPOINT'),
    api_version=os.getenv('AZ_OPENAI_API_VERSION'),
    api_key=os.getenv('AZ_OPENAI_API_KEY'),
)

model = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=az_client),
)

#systemPrompt = "Be as verbose as possible, reply with a giant text"
systemPrompt = "# Be as concise as possible, reply with one sentence"

############################################################## WEATHER AGENT ##############################################################

class WeatherResultType(BaseModel):
    temperature: float
    weather_code: int
    fun_fact_aboout_the_location: str
    
    
weather_agent = Agent (model, system_prompt=systemPrompt, result_type=WeatherResultType)

@weather_agent.tool_plain
def get_weather_info(latitude: float, longitude: float) -> str:
    print(f"Getting weather info for {latitude}, {longitude}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,weather_code"
    response = requests.get(url)
    return response.json()

############################################################## FOODIE AGENT ##############################################################

foodie_agent = Agent (model, system_prompt='You\'re a helpful assistant who knows good food places in the locaation user specified.')

############################################################## MAIN AGENT ##############################################################
main_agent = Agent (model, system_prompt='You\'re a helpful assistant that can answer questions and delegate to other agents if needed.')

@main_agent.tool_plain
async def delegate_to_weather_agent(location: str) -> str:
    print(f"Delegating to weather agent for location: {location}")
    result = await weather_agent.run(f"What is the weather in {location}?")
    return result.output

@main_agent.tool_plain
async def delegate_to_foodie_agent(location: str) -> str:
    print(f"Delegating to foodie agent for location: {location}")
    result = await foodie_agent.run(f"Recommend a few popular restaurants or coffee shop in {location}?")
    return result.output

#async def main():
message_history = []
while True:
    current_message = input('You: ')
    if current_message.lower() == 'exit':
        break
    result = main_agent.run_sync (current_message, message_history=message_history)
    message_history = result.all_messages()
    print(result.output)

    
#if __name__ == "__main__":
#    asyncio.run(main())