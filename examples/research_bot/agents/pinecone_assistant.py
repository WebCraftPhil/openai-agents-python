    from agents import Agent

    pinecone_assistant = Agent(
        name="PineconeAssistant",
        instructions=(
            "You answer with concise, hype-ready copy using only the retrieved context. "
            "If results feel stale, ask to refresh the index."
        ),
        model="gpt-5-mini",
        tools=[pinecone_tool],
    )