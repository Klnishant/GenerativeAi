import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
import streamlit as st
from typing import List, Dict
from pydantic import BaseModel
import asyncio
from duckduckgo_search import DDGS
from praisonaiagents import Agent, Task, PraisonAIAgents, TaskOutput

_ = load_dotenv(find_dotenv())
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq()

class SearchResult(BaseModel):
    query: str
    results: List[Dict[str, str]]
    total_results: int

async def async_search_tool(query: str) -> Dict:
    """Perform asynchronus search and return structured result"""
    await asyncio.sleep(1) #simulate network delay

    try:
        results = []
        ddgs = DDGS()

        for result in ddgs.text(keywords=query, max_results=5):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        print(f"Error during async search: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0
        }
    
# create Agent

async_agent = Agent(
    name="async agent",
    role="Search specialist",
    goal="Perform fast parallel searches with structured result",
    backstory="Efficient in data retrival and parallel search operations",
    tools=[async_search_tool],
    self_reflect=False,
    verbose=True,
    markdown=True,
    llm=llm,
)

summary_agent = Agent(
    name="summary agent",
    role="Reaserch Synthesizer",
    goal="Create concise summary for multiple search result",
    backstory="Expert in analyzing and synthesizing information from multiple sources",
    self_reflect=True,
    verbose=True,
    markdown=True,
    llm=llm,
)

# Create Task

async_task = Task(
    name="async search",
    description="Search for 'Async programming' and return results in JSON format with query, results array, and total_results count.",
    expected_output="SearchResult model with structured data",
    agent= async_agent,
    async_execution=True,
    output_json=SearchResult
)

async def run_parallel_tasks():
    """Run multiple asyncs task in parallel"""
    print("\n Running parallel async task")

    search_topics = [
        "Latest AI Developments 2024",
        "Machine Learning Best Practices",
        "Neural Networks Architecture"
    ]

    parallel_tasks = [
        Task(
            name=f"search task {i}",
            description=f"Search for '{topic}' and return structured results with query details and findings.",
            expected_output="SearchResult model with search data",
            agent=async_agent,
            async_execution=True,
            output_json=SearchResult
        ) for i,topic in enumerate(search_topics)
    ]

    # create summary task

    summary_task = Task(
        name="summary task",
        description="Analyze all search results and create a concise summary highlighting key findings, patterns, and implications.",
        expected_output="Structured summary with key findings and insights",
        agent=summary_agent,
        async_execution=False,
        context=parallel_tasks
    )

    # start agent

    agents = PraisonAIAgents(
        agents=[async_agent, summary_agent],
        tasks=parallel_tasks + [summary_task],
        verbose=1,
        process='sequential'
    )

    # run all tasks
    results = await agents.astart()

    print(parallel_tasks)
    print(summary_task)

    print(results)

    # Return results in serializable format
    return {
        "serach results":{
            "task_status": {k: v for k , v in results["task_status"].items() if k!=summary_task.id},
            "task_results": [str(results["task_results"][i]) if results["task_results"][i] else None
                                 for i in range(len(parallel_tasks))]
        },
        "summary": str(results["task_results"][summary_task.id]) if results["task_results"].get(summary_task.id) else None,
        "topics": search_topics,
    }

async def main():
    """Main execution function"""
    print("Starting Async AI Agents Examples...")

    try:
        print(async_task)
        print(await async_search_tool("Latest ai development 2024"))
        results = await run_parallel_tasks()

        # display results in streamlit
        st.title("Search Results")
        st.header("Search Topics")
        st.write(results["topics"])

        st.header("Search Result")
        for i, result in enumerate(results["search_results"]["task_results"]):
            st.subheader(f"Topic {i+1}")
            st.json(result)
        
        st.header("Summary")
        st.write(results["summary"] or " No summary generated")

    except Exception as e:
        print(f"Error in main execution: {e}")
        st.error(f"Error in main execution: {e}")

# Run streamlit

if __name__ == "__main__":
    asyncio.run(main())