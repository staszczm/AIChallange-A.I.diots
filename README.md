# AI Challenge 2023
Python application that uses _LangChain_ along with _OpenAI API_ to analyse the given repository and answer the 
user's questions related to its content.

[![Showcase of the application](https://img.youtube.com/vi/B__fuZhpt9U/0.jpg)](https://www.youtube.com/watch?v=B__fuZhpt9U)

### Step-by-step program's behaviour
1. Clones the repository
2. Splits Python files into smaller chunks
3. Creates an embedding database of them
4. Initialises LangChain with OpenAI model
5. Done! You can ask about the repository now!

### Required Python libraries:
* langchain
* git
* os

**Note:**  This project requires you to provide your own **OpenAI API key** in `API` environmental variable!