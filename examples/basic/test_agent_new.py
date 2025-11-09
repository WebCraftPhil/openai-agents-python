from agents import Agent
from agents._config import Config, Instructions

# Create configuration for the agent
config = Config(
    instructions=Instructions(
        system="You are a helpful assistant that responds with short, concise answers.",
        task="When the user asks a question, provide a brief response."
    )
)

# Create and run the agent
def main():
    # Initialize the agent with our configuration
    agent = Agent(config=config)
    
    # Test the agent with a simple question
    response = agent.run("What's the capital of France?")
    print("Agent response:", response)

    # Test with another question to ensure continued interaction
    response = agent.run("What's the population of Paris?")
    print("Agent response:", response)

if __name__ == "__main__":
    main()