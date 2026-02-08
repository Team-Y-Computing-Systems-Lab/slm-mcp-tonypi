import numpy as np

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


class AIResponse(BaseModel):
    """Your response to the user as an AI agent."""
    response: str = Field(description="response to the user")
    actions: list = Field(description="list of actions (tool calls)")


@tool
def get_object_coordinates(object_name_attributes: str):
    """Fetch approximate object coordinates from the image for one object at a time."""
    print("tool: get_object_coordinates called with:", object_name_attributes)
    coords = [
        np.random.randint(0, 10),
        np.random.randint(0, 10),
        np.random.randint(0, 10),
        np.random.randint(0, 10),
    ]
    print("coords:", coords)
    return coords


@tool
def get_weather(city: str):
    """Fetches the weather of the city; returns a simple dictionary."""
    return {city: "sunny and humid"}


if __name__ == "__main__":
    model = ChatOllama(
        model="llama3.1:8b",  # make sure this model supports tools
        temperature=0.8,
        num_predict=256,
        response_format=AIResponse,  # only if your Ollama build supports this
    )

    tools = [get_object_coordinates, get_weather]

    agent = create_agent(
        model,
        tools,
        system_prompt="You are a helpful AI assistant.",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "get me object red coordinates first and then blue and also what's the weather like in atlanta"
        }]
    })

    # Last AI message:
    last_msg = result["messages"][-1]
    print("=== LAST MESSAGE ===")
    print(last_msg)
    print("=== CONTENT ===")
    print(last_msg.content)
