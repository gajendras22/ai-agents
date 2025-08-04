# Goal
Define a complex problem that needs to be solved by an Agentic solution using each framework that we add to this repository. 
The idea is to have an iterative way to approach the solution so that coming up with the solution does not seem overwhelming. 
This approach has proved useful to train individuals with no knowledge of GenAI to gain familiarity and confidence in a brand new Agentic framework.

**Level 1: Basic**

Build a basic “Hello World” agent. Develop a single agent (with no tools) powered by an LLM.

Sample interaction:

```
User: What are the principles of OOPS?
AI: (whatever answer we get from LLM)
```    

**Level 2: Conversational Memory**

LLMs are stateless by default. Let's learn to add memory and manage conversation state for agents.

Sample interaction:
```
User: My name is Sakhi, and I am an engineering student interested in compiler design and ML. What are some good career options?
AI: (some answer from LLM)
User: What are some relevant textbooks that align with my interests?
AI: (based on prior info.... )
```
<br>

**Level 3: Tools**

LLMs by default cannot interact or observe the real world. They need tools for that. 
So let's learn to integrate some tools. 
Add web search (e.g., Google Search) with agentic flow. In short make our simple system behave like a light-weight clone of Perplexity.

Sample interaction:
```
User: What is the latest on US tariffs?
AI: (performs the search, understands the content and then answers the question based on the content)
```  
 
**Level 4: Vector Store**

Public search engines will still NOT have access to proprietary info. So let's learn to implement Retrieval Augmented Generation aka RAG.
Let's learn to create a vector store and push content into it.
E.g. Ingest a resume and then enable your system to answer questions based on that info.

Sample interaction:
```
User: What is my AIR? What is my CGPA?
AI: (based on the resume - the ans should be provided)
```  

**Level 5: Notebook LM mimic**

Implement a multi-agent, multi-tool system to mimic [Google NotebookLM](https://notebooklm.google.com/).  
The system should take input from the user and be able to ingest files, YouTube videos or webpages.  
The system should also be able to answer "simple questions" from the user based on the ingested information.  
The system should also be able to generate "mind map" for the user's question/topic based on the ingested information.  
The system should also be able to create a "2-person audio podcast" for the user's question/topic based on the ingested information.   
