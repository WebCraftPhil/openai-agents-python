    from agents import function_tool, ToolOutputText

    def pinecone_search(query: str, top_k: int = 5) -> str:
        results = index.query(
            vector=embed(query), top_k=top_k, include_metadata=True
        )
        return ToolOutputText(
            text="\n".join(
                f"{r['metadata']['title']} — {r['metadata']['summary']}"
                for r in results["matches"]
            )
        )

    pinecone_tool = function_tool(
        pinecone_search,
        description="Search the curated tech/AI knowledge base.",
    )