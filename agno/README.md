Creation of AI Agents using AGNO
-----------------------------------------
AGNO:
  
  [Agno](https://docs.agno.com/introduction) is a lightweight library for building Agents with memory, knowledge, tools and reasoning.
  
  Here are some key features:
  - Model Agnostic: Agno Agents can connect to 23+ model providers, no lock-in.
  - Lightning Fast: Agents instantiate in ~3μs and use ~5Kib memory on average (see performance for more details).
  - Reasoning is a first class citizen: Make your Agents “think” and “analyze” using Reasoning Models, ReasoningTools or our custom chain-of-thought approach.
  - Natively Multi-Modal: Agno Agents are natively multi-modal, they can take in text, image, audio and video and generate text, image, audio and video as output.
  - Advanced Multi-Agent Architecture: Agno provides an industry leading multi-agent architecture (Agent Teams) with 3 different modes: route, collaborate and coordinate.
  - Agentic Search built-in: Give your Agents the ability to search for information at runtime using one of 20+ vector databases. Get access to state-of-the-art Agentic RAG that uses hybrid search with re-ranking. Fully async and highly performant.
  - Long-term Memory & Session Storage: Agno provides plug-n-play Storage & Memory drivers that give your Agents long-term memory and session storage.
  - Structured Outputs: Agno Agents can return fully-typed responses using model provided structured outputs or json_mode.
  - Pre-built FastAPI Routes: Agno provides pre-built FastAPI routes to serve your Agents, Teams and Workflows.
  - Monitoring: Monitor agent sessions and performance in real-time on agno.com.

![image](https://github.com/user-attachments/assets/974898e8-fb0c-42b1-860a-f35777f1386a)


**Level 1: Basic**

Build a basic “Hello World” agent
Develop a single agent (no tools) powered by an LLM

Sample interaction:

```
User: What are the principles of OOPS?
AI: (whatever answer we get from LLM)
```

**Level 2: Conversational Memory**

Solve and learn conversation state through agents

Sample interaction:
```
User: My name is Sakhi, and I am an engineering student interested in compiler design and ML. What are some good career options?
AI: (some answer from LLM)
User: What are some relevant textbooks that align with my interests?
AI: (based on prior info.... )
```

**Level 3: Tools**

Integrate web search (e.g., Google Search) with agentic flow  (like Perplexity)

Sample interaction:
```
User: What is the latest on US tariffs?
AI: (performs the search and returns the results relevant to the search - but not as a list of search results but relevant answer to the question)
``` 
**Level 4: Vector Store**

Create a vector store (e.g., Compiler Design book) and push content into it
Vectorize a resume and extract structured info
Design sequential or parallel agent workflows
Enable agent-to-tool calling
Enable OCR (traditional or GenAI) to read scans as well

Sample interaction:
```
User: What is my AIR? What is my CGPA?
AI: (based on the resume - the ans should be provided)
```
**Level 5: Notebook LM mimic**

Implement a multi-agent, multi-tool system (PDFs, YouTube urls, Web Page urls, copy & paste text)
Create a [Google NotebookLM](https://notebooklm.google.com/) like feature to generate a mind map from large data dumps within this program
create a NotebookLM style audio podcast 

🎥 **Demo**

Check out a demo video of the project [here](https://drive.google.com/file/d/18Wg2raXwxN59V_wJUibrPRMmE3xWuyvK/view?usp=sharing).

