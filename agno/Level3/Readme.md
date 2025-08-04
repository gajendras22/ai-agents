**web-teams:**
--
web-teams.py is an example of orchestrating multiple AI agents as a collaborative team to perform advanced web search and synthesis tasks using the agno framework.

🔑Key Features:
1. It defines three specialized agents:
    - Query Reformulator Agent: Improves user queries for more effective searching.
    - Web Searcher Agent: Uses [DuckDuckGo](https://duckduckgo.com/) to search the web based on the reformulated query.
    - Answer Synthesis Agent: Synthesizes information from multiple articles into a coherent answer.
2. Coordinates these agents as a team using the Team class, so each agent performs its task in sequence, passing results to the next.
3. Enables advanced features such as tool usage visibility, markdown formatting, debug mode, and sharing of context/history among agents.

⚙️ Setup

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ Run the script

```python web-teams.py```

This file demonstrates how to build modular, collaborative AI workflows for complex information retrieval and summarization tasks.

OUTPUT:
![image](../assets/448034075-8c766f3e-cd50-4e10-9492-cb93df02732e.png)

CLI:
![image](../assets/448034888-b74a8f84-25de-49f3-8f1f-29452edcbaa7.png)
![image](../assets/448035094-288367b6-7393-4af4-9db5-d8b6e41514a4.png)

**websearch:**
--
websearch.py is a sample script demonstrating how to build an AI-powered web search agent using the agno framework.

Creates a single AI agent (WebSearchAgent) that:
- Uses the Gemini language model for understanding and generating responses.
- Integrates DuckDuckGo search tools to fetch real-time web results.
- Streams responses and intermediate reasoning steps for transparency.
- Supports markdown formatting and monitoring for debugging.

⚙️ Setup

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ Run the script

```python websearch.py```

This file is a practical example of combining LLMs with web search tools for real-time, interactive information retrieval.

OUTPUT:
![image](../assets/448037856-29b94b21-5a13-4ba5-a077-78f21d78f175.png)
