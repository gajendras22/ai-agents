Document Q&A WorkFlow
--
l4-w.py is an advanced example script that demonstrates how to build an end-to-end document question answering (Q&A) workflow using the Agno framework, the Gemini language model, and ChromaDB for vector storage.

🔑 Key Features
- Downloads and processes PDFs from a URL
- Uses Gemini embeddings + ChromaDB for semantic search
- Answers user questions based on document content using a Gemini agent
- Caches repeated queries for faster responses
- Maintains workflow state with SQLite
- Includes retry logic for Google API quota errors
- Launches an interactive Agno Playground web UI for real-time Q&A

⚙️ **Setup**

Create a .env file in the project root:

```
GOOGLE_API_KEY=your-api-key-here
AGNO_API_KEY=your-agno-api-key
```

▶️ **Run the script**

```python l4-w.py```


🎨**Customization:**

- Change the PDF URL in the script to use your own document.
- Adjust the workflow or agent instructions to fit your use case.


This script is ideal for building interactive, document-aware AI assistants that can answer questions based on the content of uploaded or linked PDFs.

OUTPUT:
--
Link for the PDF used here: https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf

![image](https://github.com/user-attachments/assets/f2e8fb04-2517-42cc-8755-76932f6ebe2f)

