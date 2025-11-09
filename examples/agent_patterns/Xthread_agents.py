import asyncio

from agents import Agent, Runner, RunContextWrapper

async def main():
    context = RunContextWrapper(None)
    result = await Runner.run(Agent("Assistant"),"What do you want to create a thread about?", context=context.context)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

    "What do you want to create a thread about?"
    # This workflow defines agents to:
    # 1. Research the topic on the internet.
    # 2. Evaluate viral potential.
    # 3. List the top 5 findings suitable for viral X (Twitter) content.

    from agents import function_tool, Agent, Runner
    from pydantic import BaseModel
    from typing import List

    @function_tool
    async def web_search(topic: str) -> List[str]:
        """
        Search the internet for the given topic and return a list of relevant findings.
        """
        # For demonstration, we mock this result.
        # Replace this with a real web search call in your real app.
        return [
            f"{topic} - Recent breakthrough",
            f"{topic} - Top controversy",
            f"{topic} - Influencer opinions",
            f"{topic} - Unexpected fact",
            f"{topic} - Community trend",
            f"{topic} - Viral meme spin",
            f"{topic} - Popular misconception",
        ]

    class ViralCandidates(BaseModel):
        viral_ideas: List[str]

    viral_evaluator = Agent(
        name="viral_evaluator",
        instructions=(
            "You are a social media consultant. Given a list of findings about a topic, "
            "select and list the top 5 that have the highest potential for viral X (Twitter) content. "
            "Explain your reasoning for each choice."
        ),
        output_type=ViralCandidates,
    )

    research_agent = Agent(
        name="internet_researcher",
        instructions=(
            "You research the internet on any user-provided topic. "
            "Return a variety of angles, stories, controversies, opinions, and facts about the topic, optimized for social virality."
        ),
        tools=[web_search],
    )

    async def viral_workflow(topic: str):
        # Step 1: Research the topic
        research = await Runner.run(
            research_agent,
            f"Research the topic '{topic}' and find potentially viral information.",
        )
        findings = research.final_output if isinstance(research.final_output, list) else research.raw_output

        # Step 2: Evaluate viral potential and pick top 5
        eval_prompt = (
            f"Given these research findings about '{topic}':\n"
            + "\n".join(f"- {item}" for item in findings)
            + "\nList the top 5 with highest potential for virality on X. "
              "Return only the list of ideas/titles."
        )
        viral_result = await Runner.run(viral_evaluator, eval_prompt)
        print("Top 5 Viral X Ideas for", topic)
        for idea in viral_result.final_output.viral_ideas:
            print("-", idea)

    # Interactive demonstration
    user_topic = input("What do you want to create a thread about? ")
    asyncio.run(viral_workflow(user_topic))
    # INSERT_YOUR_CODE

    # --- Additional required imports for the new features (assume at the top) ---
    # import pinecone
    # from uuid import uuid4

    from agents import function_tool
    from typing import Dict

    # User supplies Pinecone API key/index name out of band (or fetch from env/config)
    PINECONE_API_KEY = "your-pinecone-api-key"
    PINECONE_ENV = "us-west1-gcp"  # or your environment
    PINECONE_INDEX_NAME = "viral-thread-ideas"

    # Initialize Pinecone index (idempotent)
    import pinecone

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536)  # update as per embedding size
    index = pinecone.Index(PINECONE_INDEX_NAME)

    # Tool: gets embeddings from OpenAI or compatible model
    @function_tool
    async def embed_text(text: str) -> list[float]:
        # For brevity, stub. Replace with call to embedding API.
        # from openai import AsyncOpenAI
        # resp = await AsyncOpenAI().embeddings.create(input=[text], model="text-embedding-ada-002")
        # return resp.data[0].embedding
        # FAKE embedding (should use real model):
        return [float((ord(c) % 7) / 6) for c in text.ljust(1536)[:1536]]

    # Store best ideas in Pinecone with embeddings and metadata
    async def store_in_pinecone(ideas: list[str], topic: str) -> list[str]:
        ids = []
        for idea in ideas:
            idea_id = str(uuid4())
            embedding = await embed_text(idea)
            meta = {"topic": topic, "idea": idea}
            index.upsert([(idea_id, embedding, meta)])
            ids.append(idea_id)
        return ids

    # Query Pinecone for top N ideas for this topic (semantic search)
    async def query_pinecone(topic: str, query: str, top_k=5) -> list[Dict]:
        query_embedding = await embed_text(query)
        res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True,
                          filter={"topic": {"$eq": topic}})
        return res.matches

    # The viral thread creator agent
    class ViralThread(BaseModel):
        thread: list[str]

    viral_thread_creator = Agent(
        name="viral_thread_creator",
        instructions=(
            "Given a selected viral idea, create a Twitter thread that could go viral. "
            "The thread should have short, punchy tweets one after another, and each message should fit in a single tweet (under 240 characters)."
        ),
        output_type=ViralThread,
    )
    
    # The thread editor agent
    class EnhancedThread(BaseModel):
        tweets: list[str]
        image_prompts: list[str]

    thread_editor = Agent(
        name="thread_editor",
        instructions=(
            "Given a draft Twitter thread (list of tweets), edit each tweet to maximize X (Twitter) engagement. "
            "For each tweet, make sure it is under 240 characters. "
            "For each tweet, also generate a short, compelling AI image prompt (for an image that would stop a user from scrolling and attract attention to the tweet). "
            "Return two lists: tweets (each edited for engagement), and image_prompts."
        ),
        output_type=EnhancedThread,
    )

    async def viral_workflow_with_pinecone(topic: str):
        # Step 1: Research the topic
        research = await Runner.run(
            research_agent,
            f"Research the topic '{topic}' and find potentially viral information.",
        )
        findings = research.final_output if isinstance(research.final_output, list) else research.raw_output

        # Step 2: Evaluate viral potential and pick top 5
        eval_prompt = (
            f"Given these research findings about '{topic}':\n"
            + "\n".join(f"- {item}" for item in findings)
            + "\nList the top 5 with highest potential for virality on X. "
              "Return only the list of ideas/titles."
        )
        viral_result = await Runner.run(viral_evaluator, eval_prompt)
        top_ideas = viral_result.final_output.viral_ideas

        # Step 3: Store top ideas in Pinecone
        await store_in_pinecone(top_ideas, topic)

        # Step 4: Present top 5 to user and ask for selection
        print("\nTop 5 Viral X Ideas for", topic)
        for idx, idea in enumerate(top_ideas, 1):
            print(f"{idx}. {idea}")
        chosen_idx = None
        while chosen_idx is None:
            try:
                chosen_idx = int(input("Select the number of the idea you'd like a thread for: ")) - 1
                if not (0 <= chosen_idx < len(top_ideas)):
                    raise ValueError
            except ValueError:
                print("Please enter a valid number 1-", len(top_ideas))
                chosen_idx = None
        chosen_idea = top_ideas[chosen_idx]

        # Step 5: Let viral_thread_creator generate the thread
        thread_result = await Runner.run(
            viral_thread_creator,
            f"Write a viral Twitter thread about this idea: '{chosen_idea}'. Number/list each tweet.",
        )
        thread = thread_result.final_output.thread

        # Step 6: Editor agent optimizes thread & adds image prompts
        editor_result = await Runner.run(
            thread_editor,
            {
                "tweets": thread
            }
        )
        enhanced = editor_result.final_output
        print("\n--- Final Enhanced Viral Thread ---")
        for i, (tweet, img_prompt) in enumerate(zip(enhanced.tweets, enhanced.image_prompts), 1):
            print(f"\nTweet {i}:", tweet)
            print("AI Image Prompt:", img_prompt)

    # Interactive demonstration
    user_topic = input("What do you want to create a thread about? ")
    asyncio.run(viral_workflow_with_pinecone(user_topic))
