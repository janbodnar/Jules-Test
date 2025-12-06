// OpenAI with Python: A Beginner's Guide
// Comprehensive 100-page beginner's guide to OpenAI and Python
// Sourced from markdown materials: intro.md, history.md, models.md, openai.md, openai2.md, openai-article1.md, openai-article2.md
// Format: Typst (typst.app)

#set document(
  title: "OpenAI with Python: A Beginner's Guide",
  author: "Jan Bodnar",
  date: auto,
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#show heading: it => { it; v(0.8em) }
#show raw.where(block: true): it => block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  it
)

// Title page
#align(center + horizon)[
  #text(size: 28pt, weight: "bold")[OpenAI with Python]
  #v(1.5em)
  #text(size: 20pt)[A Beginner's Guide]
  #v(0.5em)
  #text(size: 14pt)[From Basics to Advanced Applications]
  #v(3em)
  #text(size: 14pt)[Jan Bodnar]
  #v(1em)
  #text(size: 12pt)[#datetime.today().display("[month repr:long] [day], [year]")]
]

#pagebreak()


= Python Basics for AI Development

Before diving into AI programming, let's ensure you have a solid foundation in the Python concepts most relevant to working with AI APIs. This chapter covers essential Python skills you'll use throughout the book.

== Setting Up Your Python Environment

To work with OpenAI and AI models, you need Python 3.8 or higher. We recommend Python 3.10 or later for the best compatibility and features.

Check your Python version:

```bash
python --version
# or
python3 --version
```

If you need to install Python, download it from python.org. On Windows, make sure to check "Add Python to PATH" during installation.

== Virtual Environments

Virtual environments isolate project dependencies, preventing conflicts between different projects. Always create a virtual environment for your AI projects:

```bash
# Create a virtual environment
python -m venv ai_env

# Activate on Windows
ai_env\Scripts\activate

# Activate on macOS/Linux  
source ai_env/bin/activate

# Deactivate when done
deactivate
```

Once activated, your terminal prompt will show the environment name (ai_env). Install packages only after activating the environment.

== Essential Python Concepts

Understanding these Python concepts is crucial for AI development.

=== Dictionaries and JSON

AI APIs work extensively with JSON data, which Python represents as dictionaries:

```python
# Creating a message for an AI API
message = {
    "role": "user",
    "content": "What is Python?"
}

# Accessing values
role = message["role"]
content = message.get("content", "")

# Building complex structures
conversation = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]
```

=== Lists and Iteration

AI responses often come as lists that you need to iterate through:

```python
# Processing multiple messages
messages = [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
    {"role": "user", "content": "Second question"}
]

for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

# List comprehensions for filtering
user_msgs = [m for m in messages if m['role'] == 'user']
```

=== Functions and Parameters

Organizing your AI code into functions makes it reusable and maintainable:

```python
def create_chat_message(role, content):
    """
    Create a properly formatted chat message.
    
    Args:
        role: The role (system, user, or assistant)
        content: The message content
        
    Returns:
        A dictionary representing the message
    """
    return {
        "role": role,
        "content": content
    }

# Using the function
user_msg = create_chat_message("user", "Hello AI!")
system_msg = create_chat_message("system", "You are helpful.")
```

=== Exception Handling

API calls can fail for various reasons. Proper error handling is essential:

```python
try:
    # API call that might fail
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error occurred: {e}")
    # Handle the error appropriately
```

=== Environment Variables

Store sensitive information like API keys securely using environment variables:

```python
import os

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")

# Or with a default value
api_key = os.getenv("OPENAI_API_KEY", "default_key")

# Check if key exists
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
```

== Working with Strings

String manipulation is crucial when working with AI-generated text:

```python
# String formatting for prompts
name = "Alice"
topic = "Python"
prompt = f"Explain {topic} to {name} who is a beginner"

# Multiline strings for complex prompts
system_prompt = """
You are a helpful tutor.
Be patient and explain concepts clearly.
Use examples when possible.
"""

# String methods for text processing
text = "  Hello, World!  "
cleaned = text.strip()          # Remove whitespace
lower = text.lower()            # Convert to lowercase
words = text.split()            # Split into words
joined = " ".join(words)        # Join words back

# Check string content
if "Python" in text:
    print("Found Python!")
```

== File Operations

Reading and writing files is common when working with AI applications:

```python
# Writing AI responses to a file
with open("ai_conversation.txt", "w") as f:
    f.write("User: Hello AI\n")
    f.write("AI: Hello! How can I help?\n")

# Reading configuration files
with open("config.txt", "r") as f:
    content = f.read()
    
# Reading line by line
with open("prompts.txt", "r") as f:
    for line in f:
        print(line.strip())

# Appending to a log file
with open("ai_log.txt", "a") as f:
    import datetime
    f.write(f"Query at {datetime.datetime.now()}\n")

# Working with JSON files
import json

# Save data to JSON
data = {"model": "gpt-4", "temperature": 0.7}
with open("config.json", "w") as f:
    json.dump(data, f, indent=2)

# Load data from JSON
with open("config.json", "r") as f:
    config = json.load(f)
```

== Working with Classes

Object-oriented programming helps organize complex AI applications:

```python
class AIAssistant:
    """A simple AI assistant wrapper."""
    
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
    
    def add_message(self, role, content):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_message):
        """Get AI response to user message."""
        self.add_message("user", user_message)
        # API call would go here
        response = "AI response"
        self.add_message("assistant", response)
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Using the class
assistant = AIAssistant()
response = assistant.get_response("Hello!")
```

== Exercises

Practice these concepts before moving forward:

1. Create a function that builds a complete conversation history from lists of user questions and AI responses.

2. Write a script that reads API credentials from a `.env` file and validates they are present.

3. Build a simple error logger that writes error messages to a file with timestamps.

4. Create a prompt template system that fills in variables based on user input.

5. Write a class that manages multiple AI conversations with different system prompts.

== Key Takeaways

- Use Python 3.8 or higher for AI development
- Always work in virtual environments to manage dependencies
- Master dictionaries and lists for working with JSON data
- Use functions to organize reusable AI code
- Handle exceptions properly for robust applications
- Store API keys in environment variables, never in code
- Practice string manipulation for prompt engineering
- Understand file operations for logging and data persistence
- Use classes to organize complex AI applications

#pagebreak()

= Installation and Environment Setup

Setting up your development environment correctly is crucial for a smooth AI development experience. This chapter guides you through installing necessary tools and configuring your system.

== Installing the OpenAI Python Library

The OpenAI Python library provides a convenient interface for working with OpenAI's API and compatible services.

```bash
# Install the OpenAI library
pip install openai

# Install a specific version
pip install openai==1.3.0

# Upgrade to the latest version
pip install --upgrade openai
```

== Additional Useful Libraries

For comprehensive AI development, install these additional libraries:

```bash
# For HTTP requests
pip install requests

# For async programming
pip install aiohttp

# For web interfaces
pip install gradio

# For structured data
pip install pydantic

# For environment variables
pip install python-dotenv

# For beautiful terminal output
pip install rich

# For HTML parsing
pip install beautifulsoup4
```

Create a `requirements.txt` file for easy dependency management:

```text
openai>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
aiohttp>=3.9.0
gradio>=4.0.0
beautifulsoup4>=4.12.0
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

== Setting Up API Keys

Never hardcode API keys in your source code. Use environment variables instead.

=== Creating a .env File

Create a file named `.env` in your project root:

```text
OPENAI_API_KEY=sk-your-openai-key-here
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
GROQ_API_KEY=gsk_your-groq-key-here
```

*Important:* Add `.env` to your `.gitignore` file to prevent accidentally committing secrets to version control:

```text
# .gitignore
.env
__pycache__/
*.pyc
ai_env/
venv/
```

=== Loading Environment Variables

Use python-dotenv to load variables from your .env file:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API keys
openai_key = os.getenv("OPENAI_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Check if key exists
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found")
```

=== System Environment Variables

Alternatively, set environment variables at the system level:

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"

# Make permanent
[System.Environment]::SetEnvironmentVariable(
    "OPENAI_API_KEY", 
    "sk-your-key-here", 
    "User"
)
```

On macOS/Linux:
```bash
export OPENAI_API_KEY="sk-your-key-here"

# Make permanent by adding to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

== Obtaining API Keys

Different AI providers require different signup processes.

=== OpenAI

1. Visit platform.openai.com
2. Create an account or sign in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy and store the key securely
6. Note: OpenAI requires payment information for API access

=== OpenRouter

1. Visit openrouter.ai
2. Sign up with Google, GitHub, or email
3. Navigate to Keys section
4. Create a new API key
5. Fund your account (many free models available)
6. OpenRouter provides access to dozens of models from various providers

=== DeepSeek

1. Visit platform.deepseek.com
2. Create an account
3. Verify your email
4. Generate an API key from your dashboard
5. DeepSeek offers competitive pricing and free tier

=== Groq

1. Visit console.groq.com
2. Sign up for an account
3. Generate an API key
4. Groq offers generous free tier for testing

== Verifying Your Setup

Create a test script to verify everything works:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_provider(name, base_url, api_key_name, model):
    """Test a single AI provider."""
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv(api_key_name)
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Setup successful!'"}
            ]
        )
        
        print(f"✓ {name} working:")
        print(f"  {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ {name} error: {e}")
        return False

# Test providers
print("Testing AI provider connections...\n")

providers = [
    ("OpenRouter", "https://openrouter.ai/api/v1", 
     "OPENROUTER_API_KEY", "anthropic/claude-3-haiku"),
    ("DeepSeek", "https://api.deepseek.com", 
     "DEEPSEEK_API_KEY", "deepseek-chat"),
    ("Groq", "https://api.groq.com/openai/v1", 
     "GROQ_API_KEY", "llama-3.3-70b-versatile"),
]

results = []
for name, url, key, model in providers:
    result = test_provider(name, url, key, model)
    results.append((name, result))
    print()

# Summary
print("="*50)
print("Setup Summary:")
for name, success in results:
    status = "✓ Ready" if success else "✗ Needs setup"
    print(f"{name}: {status}")
```

== Project Structure

Organize your AI projects with a clear structure:

```
my_ai_project/
├── .env                 # Environment variables (gitignored)
├── .gitignore          # Git ignore file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── config.py          # Configuration management
├── src/
│   ├── __init__.py
│   ├── client.py      # AI client initialization
│   ├── prompts.py     # Prompt templates
│   └── utils.py       # Utility functions
├── examples/
│   ├── basic_chat.py
│   ├── streaming.py
│   └── function_calling.py
├── tests/
│   └── test_client.py
└── data/
    └── conversations/
```

== Configuration Management

Create a `config.py` file for centralized configuration:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # API Endpoints
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    # Default Models
    DEFAULT_MODEL = "anthropic/claude-3-haiku"
    VISION_MODEL = "openrouter/sonoma-sky-alpha"
    FREE_MODEL = "z-ai/glm-4.5-air:free"
    
    # API Parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
    
    @classmethod
    def validate(cls):
        """Validate that required keys are present."""
        required = ['OPENROUTER_API_KEY']
        missing = [k for k in required if not getattr(cls, k)]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

# Usage example
# from config import Config
# Config.validate()
# api_key = Config.OPENROUTER_API_KEY
```

== Troubleshooting Common Issues

=== Import Errors

If you get "ModuleNotFoundError":

```bash
# Ensure you're in your virtual environment
which python  # Should show path in your venv

# Reinstall the package
pip install --force-reinstall openai

# Check installed packages
pip list | grep openai
```

=== API Key Not Found

If you get authentication errors:

- Verify .env file is in the correct directory
- Check that `load_dotenv()` is called before accessing keys
- Ensure no extra spaces in .env file
- Verify the key is valid and active on the provider's website

=== SSL/Certificate Errors

On some systems, you may encounter SSL errors:

```bash
# Update certifi
pip install --upgrade certifi

# Or use system certificates (Linux/Mac)
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

=== Permission Errors

If you get permission errors when installing packages:

```bash
# Don't use sudo - instead use venv
python -m venv ai_env
source ai_env/bin/activate
pip install openai
```

== Best Practices

1. *Always use virtual environments* - Prevents dependency conflicts
2. *Never commit .env files* - Add to .gitignore immediately
3. *Use requirements.txt* - Makes setup reproducible
4. *Validate configuration* - Check keys exist before using
5. *Test setup before coding* - Run verification scripts
6. *Document your setup* - README.md with setup instructions
7. *Version your dependencies* - Pin versions in requirements.txt
8. *Keep keys secure* - Use environment variables, not hardcoded

== Exercises

1. Set up a complete Python virtual environment for AI development
2. Create a .env file with API keys for at least two providers
3. Write a configuration module that validates all required keys are present
4. Create a test script that verifies connectivity to multiple AI providers
5. Build a simple CLI tool that lets you switch between different AI providers
6. Write a setup script that automates the entire environment setup process

== Key Takeaways

- Install OpenAI library using pip in a virtual environment
- Never hardcode API keys - use environment variables
- Use python-dotenv for easy environment variable management
- Obtain API keys from multiple providers for flexibility
- Verify your setup with test scripts before building applications
- Organize projects with clear structure and configuration management
- Keep .env files out of version control with .gitignore
- Handle API errors gracefully with try-except blocks
- Test all connections before starting development
- Document your setup process for team members

#pagebreak()



#outline(
  title: "Table of Contents",
  indent: auto,
  depth: 2,
)

#pagebreak()

= Preface

This comprehensive guide teaches you how to work with OpenAI and large language models using Python. Designed for beginners, it takes you from basic concepts to building sophisticated AI applications with practical, hands-on examples.

The field of artificial intelligence is rapidly evolving. This book focuses on fundamental concepts and practical skills that will remain relevant as technology advances. You'll learn not just how to use AI APIs, but how to think about AI integration in your applications.

#pagebreak()


= Introduction Artificial Intelligence (AI)


Artificial Intelligence (AI) refers to the simulation of human intelligence  
in machines that are programmed to think and learn like humans. AI has  
evolved from simple rule-based systems in the 1950s to sophisticated neural  
networks capable of processing vast amounts of data.  

The history of AI begins with Alan Turing's seminal work in the 1950s,  
followed by the development of expert systems in the 1970s, machine learning  
in the 1990s, and deep learning in the 2010s. Today, AI powers numerous  
applications across various domains.  

Key domains of AI include natural language processing (NLP), computer vision,  
robotics, expert systems, and reinforcement learning. These domains enable  
machines to understand human language, recognize images, make autonomous  
decisions, and continuously improve through experience.  





== Applications of AI


AI has transformed creative industries and technical fields alike. In  
creative writing, AI assists authors with generating ideas, drafting content,  
and even completing stories based on prompts. Tools like GPT models can  
produce coherent narratives, poetry, and technical documentation.  

Image generation has become remarkably sophisticated with models like DALL-E,  
Midjourney, and Stable Diffusion. These systems create original artwork,  
photorealistic images, and design concepts from text descriptions, enabling  
artists and designers to rapidly prototype visual ideas.  

Music composition using AI involves generating melodies, harmonies, and even  
complete musical pieces. AI models analyze patterns in existing music to  
create new compositions in various styles, assisting musicians in the  
creative process or generating background music for media.  

Video production leverages AI for tasks such as video editing, scene  
generation, deepfake creation, and automated content summarization. AI can  
analyze footage, suggest cuts, and even generate synthetic video content  
from text descriptions or still images.  

Code development has been revolutionized by AI assistants like GitHub  
Copilot, which suggests code completions, generates functions, and helps  
debug issues. These tools accelerate development cycles and help programmers  
learn new frameworks and languages more efficiently.  

Robotics combines AI with physical systems to create autonomous machines  
capable of performing complex tasks. From manufacturing robots to autonomous  
vehicles and drones, AI enables robots to perceive their environment, make  
decisions, and execute actions with precision.  





== Theoretical Foundations


Machine learning is a subset of AI that enables systems to learn from data  
without explicit programming. It includes supervised learning (learning from  
labeled data), unsupervised learning (finding patterns in unlabeled data),  
and reinforcement learning (learning through trial and error with rewards).  

These algorithms identify patterns, make predictions, and improve their  
performance over time. Common machine learning algorithms include decision  
trees, random forests, support vector machines, and clustering algorithms.  

Neural networks are computational models inspired by the human brain,  
consisting of interconnected nodes (neurons) organized in layers. Each  
connection has a weight that adjusts during training, allowing the network  
to learn complex patterns from data.  

A basic neural network includes an input layer (receiving data), hidden  
layers (processing information), and an output layer (producing results).  
The network learns by adjusting weights through backpropagation, minimizing  
the difference between predicted and actual outputs.  

Deep learning architectures use multiple hidden layers to process data  
hierarchically, extracting increasingly abstract features at each level.  
Convolutional Neural Networks (CNNs) excel at image processing, Recurrent  
Neural Networks (RNNs) handle sequential data, and Transformers power modern  
language models with attention mechanisms that weigh the importance of  
different input elements.  





== Large Language Models (LLMs)


Large Language Models are AI systems trained on vast amounts of text data to  
understand and generate human-like language. LLMs learn statistical patterns,  
grammar, facts, and reasoning abilities from billions of text examples,  
enabling them to perform various language tasks without task-specific  
training.  

Building language models involves collecting massive text datasets,  
preprocessing the data, and training neural networks (typically Transformers)  
on powerful computing infrastructure. The training process requires  
substantial computational resources, often involving thousands of GPUs  
running for weeks or months.  

Models learn to predict the next word in a sequence, developing an  
understanding of language structure, context, and meaning. The larger the  
model (measured in parameters), the more nuanced its understanding becomes.  

Leveraging existing models is more practical for most applications. Pre-  
trained models like GPT, BERT, and LLaMA can be fine-tuned for specific  
tasks with smaller datasets, or used directly through APIs. This approach  
saves time and resources while still delivering powerful AI capabilities.  

Techniques like prompt engineering, few-shot learning, and retrieval-  
augmented generation (RAG) allow developers to customize model behavior  
without retraining, making LLMs accessible for various applications.  





== Chatbots


Modern AI chatbots use large language models to engage in natural  
conversations, answer questions, and assist with various tasks.  

Copilot, developed by Microsoft, integrates AI assistance directly into  
development environments and productivity tools. It helps with code  
completion, document drafting, and task automation, learning from context  
to provide relevant suggestions.  

Gemini, Google's advanced AI system, combines multiple modalities (text,  
images, audio) to provide comprehensive assistance. It excels at research,  
analysis, and creative tasks, leveraging Google's vast knowledge base and  
infrastructure.  

ChatGPT, created by OpenAI, revolutionized conversational AI with its  
ability to engage in nuanced dialogue, explain complex topics, write code,  
and assist with creative projects. It supports extended conversations with  
context awareness and can browse the web for current information.  

DeepSeek represents a new generation of efficient AI models, optimized for  
performance and cost-effectiveness. It provides strong capabilities across  
various tasks while requiring fewer computational resources than some larger  
models.  





== Prompts


A prompt is the input text given to an AI model to generate a response. The  
quality of the prompt significantly influences the quality and relevance of  
the AI's output. Effective prompts provide clear instructions, necessary  
context, and desired output format.  

Principles of effective prompt design include:  

**Clarity and specificity**: Be explicit about what you want. Vague prompts  
yield vague responses. Specify the task, desired format, and any constraints  
clearly.  

**Context provision**: Give the AI relevant background information. The more  
context you provide, the better the model can tailor its response to your  
needs.  

**Role assignment**: Ask the AI to assume a specific role or perspective,  
such as "Act as a Python expert" or "Explain this to a beginner." This  
shapes the tone and depth of the response.  

**Output formatting**: Specify how you want the information presented, such  
as bullet points, tables, code blocks, or step-by-step instructions.  

**Examples and constraints**: Provide examples of desired output (few-shot  
learning) and explicitly state what to avoid or include. This guides the  
model toward your expectations.  

**Iterative refinement**: If the first response isn't perfect, refine your  
prompt based on what you received. Small adjustments can significantly  
improve results.  





== Practical Examples


Text summarization involves condensing lengthy documents into concise  
summaries while preserving key information. AI models analyze the content,  
identify main points, and generate coherent summaries in various lengths.  

This technique is valuable for processing research papers, news articles,  
meeting transcripts, and reports. The model can produce extractive summaries  
(selecting key sentences) or abstractive summaries (rephrasing content in  
new words).  

Translation leverages neural machine translation to convert text between  
languages while maintaining meaning, context, and tone. Modern AI translators  
handle idioms, cultural nuances, and domain-specific terminology more  
accurately than traditional rule-based systems.  

These models support dozens of languages and can translate technical  
documents, creative works, and conversational text. They learn from parallel  
text corpora to understand correspondence between languages.  

Information extraction identifies and extracts structured data from  
unstructured text. AI models can recognize named entities (people, places,  
organizations), dates, numerical values, and relationships between entities.  

Applications include extracting contact information from resumes, pulling  
financial data from reports, and identifying key facts from articles. The  
extracted information can be stored in databases or used for further  
analysis.  

Document analysis encompasses various tasks such as classification,  
sentiment analysis, and topic modeling. AI models can categorize documents  
by subject, determine the emotional tone of text, and identify recurring  
themes across large document collections.  

This enables automated processing of customer feedback, legal documents,  
scientific papers, and social media content. Models can detect anomalies,  
identify trends, and provide insights that would be time-consuming to derive  
manually.  


#pagebreak()


= History of Artificial Intelligence


Artificial Intelligence (AI) refers to the development of computer systems  
capable of performing tasks that traditionally require human intelligence,  
such as visual perception, speech recognition, decision-making, and language  
translation. AI has transformed from a theoretical concept into a fundamental  
technology reshaping every aspect of modern society, from healthcare and  
finance to entertainment and transportation.  

The significance of AI extends beyond mere technological advancement. It  
represents humanity's attempt to understand intelligence itself and replicate  
cognitive processes in machines. Today, AI systems assist doctors in  
diagnosing diseases, help scientists discover new drugs, enable autonomous  
vehicles, power personal assistants, and create art and music. As AI  
continues to evolve, it raises profound questions about the future of work,  
privacy, ethics, and what it means to be human in an increasingly automated  
world.  





== Early Foundations (1940s–1950s)


The foundations of AI were laid in the 1940s and 1950s, when mathematicians  
and scientists began exploring whether machines could think. Alan Turing, a  
British mathematician, made groundbreaking contributions with his 1950 paper  
"Computing Machinery and Intelligence," which posed the famous question "Can  
machines think?" He proposed the Turing Test as a criterion for machine  
intelligence, suggesting that if a machine could engage in conversation  
indistinguishable from a human, it could be considered intelligent.  

During World War II, Turing also worked on breaking the Enigma code, using  
early computing machines that demonstrated the potential of automated  
reasoning. His theoretical work on computation, including the concept of the  
Turing Machine, established the mathematical foundations for all modern  
computing.  

In 1956, the Dartmouth Conference marked the official birth of AI as a field.  
John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon  
organized this historic summer workshop, where they coined the term  
"Artificial Intelligence" and proposed that "every aspect of learning or any  
other feature of intelligence can in principle be so precisely described that  
a machine can be made to simulate it."  

Early AI research focused on symbolic AI, also called "Good Old-Fashioned AI"  
(GOFAI). This approach assumed that human intelligence could be reduced to  
symbol manipulation. Researchers developed programs like the Logic Theorist  
(1956) by Allen Newell and Herbert Simon, which proved mathematical theorems,  
and Arthur Samuel's checkers program (1952), which learned to play the game  
through self-play, demonstrating early machine learning concepts.  

The period also saw the development of LISP (1958) by John McCarthy, a  
programming language that became the dominant language for AI research for  
decades. Early computing milestones included the development of the first  
stored-program computers like ENIAC and UNIVAC, which provided the hardware  
foundation necessary for AI experimentation.  





== Key Milestones (1960s–1990s)


The 1960s saw continued optimism and several important developments. Joseph  
Weizenbaum created ELIZA (1966), an early natural language processing program  
that could simulate conversation by pattern matching and substitution. Though  
simple, ELIZA demonstrated that machines could appear to understand language,  
foreshadowing modern chatbots.  

The 1970s and 1980s witnessed the rise of expert systems, AI programs  
designed to emulate the decision-making ability of human experts. DENDRAL  
(1965-1969), developed at Stanford, helped chemists identify molecular  
structures. MYCIN (1972) diagnosed blood infections and recommended  
antibiotics, often matching or exceeding the performance of human specialists.  

Expert systems worked by encoding human knowledge as rules in "if-then"  
format. XCON (1980), developed for Digital Equipment Corporation, configured  
computer systems and saved the company millions of dollars. By the mid-1980s,  
expert systems had become a billion-dollar industry, with companies investing  
heavily in AI departments.  

Machine learning began to gain prominence as researchers realized that  
hand-coding knowledge had limitations. In the 1980s, backpropagation, an  
algorithm for training multi-layer neural networks, was rediscovered and  
popularized by David Rumelhart, Geoffrey Hinton, and Ronald Williams. This  
algorithm became fundamental to training deep neural networks.  

The 1980s also saw renewed interest in neural networks through the work of  
researchers like Yann LeCun, who developed convolutional neural networks  
(CNNs) for handwriting recognition. These networks could learn features  
directly from data rather than relying on hand-crafted rules.  

In 1997, IBM's Deep Blue defeated world chess champion Garry Kasparov,  
marking a milestone in AI's ability to master complex strategic games. Deep  
Blue used brute-force search combined with sophisticated evaluation functions  
to explore millions of positions per second.  





== AI Winters (1970s and 1990s)


Despite early optimism, AI faced two major periods of reduced funding and  
interest known as "AI Winters." The first occurred in the 1970s when initial  
promises failed to materialize. Early AI researchers had made bold predictions  
that machines would achieve human-level intelligence within decades, but the  
technology wasn't ready.  

The limitations of early approaches became apparent. Symbolic AI systems  
struggled with common-sense reasoning and real-world complexity. They were  
brittle, working only in narrow domains, and required enormous effort to  
encode expert knowledge. The computational power available was insufficient  
for the ambitious goals researchers had set.  

The British government's Lighthill Report (1973) criticized AI research,  
leading to severe funding cuts in the UK. Similar skepticism spread to other  
countries, causing research budgets to dry up and many AI labs to close.  

The second AI Winter began in the late 1980s and early 1990s. The expert  
systems boom collapsed as companies realized these systems were expensive to  
maintain, difficult to update, and couldn't learn from experience. When the  
specialized LISP machine market collapsed, many AI companies went bankrupt.  

The Defense Advanced Research Projects Agency (DARPA), which had been a major  
funder of AI research, cut its budget for AI substantially. The term "AI"  
became so stigmatized that researchers avoided using it, instead describing  
their work as machine learning, neural networks, or data mining.  

These winters taught valuable lessons about managing expectations,  
understanding technological limitations, and the importance of demonstrating  
practical applications. They also spurred researchers to develop more robust,  
data-driven approaches that would eventually lead to AI's resurgence.  





== Modern Era (2000s–2010s)


AI experienced a dramatic renaissance in the 2000s and 2010s, driven by three  
key factors: massive increases in computational power, the availability of  
big data, and algorithmic breakthroughs in deep learning.  

The rise of GPUs (Graphics Processing Units) revolutionized AI training.  
Originally designed for rendering graphics, GPUs excel at parallel  
computation, making them ideal for training neural networks. Companies like  
NVIDIA recognized this potential and optimized their hardware for AI  
workloads, enabling training of networks that were previously impractical.  

Big data became ubiquitous with the growth of the internet, social media, and  
digital sensors. Companies like Google, Facebook, and Amazon collected vast  
datasets of images, text, and user interactions. This data provided the fuel  
needed to train increasingly sophisticated AI models. ImageNet, a dataset of  
millions of labeled images created by Fei-Fei Li and her team in 2009,  
became crucial for advancing computer vision.  

Deep learning emerged as the dominant paradigm. In 2012, AlexNet, a deep  
convolutional neural network developed by Alex Krizhevsky, Ilya Sutskever,  
and Geoffrey Hinton, won the ImageNet competition by a large margin. This  
breakthrough demonstrated that deep neural networks trained on GPUs could  
dramatically outperform traditional computer vision approaches.  

Following AlexNet's success, deep learning rapidly advanced. VGGNet,  
GoogleNet, and ResNet pushed image recognition accuracy beyond human levels.  
In natural language processing, word embeddings like Word2Vec (2013) and  
sequence-to-sequence models transformed how machines understood language.  

In 2016, DeepMind's AlphaGo defeated Lee Sedol, one of the world's best Go  
players. Go had been considered far beyond AI's reach due to its enormous  
complexity, making this victory particularly significant. AlphaGo used deep  
neural networks combined with Monte Carlo tree search and reinforcement  
learning.  

Cloud computing platforms from Amazon, Google, and Microsoft democratized  
access to powerful computing resources, enabling smaller companies and  
researchers to train sophisticated AI models. Open-source frameworks like  
TensorFlow (2015) and PyTorch (2016) made deep learning accessible to a  
broader audience.  





== Large Language Models (LLMs)


Large Language Models represent one of the most significant recent advances  
in AI. These models are trained on vast amounts of text data to understand  
and generate human-like language, demonstrating capabilities that seemed  
impossible just a few years ago.  

The foundation for modern LLMs came from the Transformer architecture,  
introduced by Google researchers in the paper "Attention Is All You Need"  
(2017). Transformers use self-attention mechanisms to process entire  
sequences simultaneously, making them more efficient and effective than  
previous recurrent neural network architectures.  

BERT (Bidirectional Encoder Representations from Transformers), released by  
Google in 2018, showed that pre-training language models on large text  
corpora and then fine-tuning them for specific tasks could achieve  
state-of-the-art results across many language understanding benchmarks.  

OpenAI's GPT (Generative Pre-trained Transformer) series demonstrated the  
power of scaling language models. GPT-2 (2019) could generate coherent text  
across various topics. GPT-3 (2020), with 175 billion parameters, showed  
remarkable few-shot learning abilities, performing diverse tasks from  
translation to code generation with minimal examples.  

The capabilities of LLMs expanded dramatically. They could write essays,  
answer questions, summarize documents, translate languages, write code, and  
engage in nuanced conversations. ChatGPT, released by OpenAI in November  
2022, brought LLMs to mainstream attention, gaining 100 million users in just  
two months.  

Other major LLMs followed: Google's PaLM and Gemini, Anthropic's Claude,  
Meta's LLaMA, and numerous open-source models. These models demonstrated that  
scaling up model size, training data, and computation led to emergent  
abilities not present in smaller models.  

LLMs have transformed industries by automating content creation, providing  
intelligent assistance, accelerating software development, and enabling new  
forms of human-computer interaction. They power applications from customer  
service chatbots to research assistants, coding tools like GitHub Copilot,  
and creative writing aids.  

However, LLMs also raise concerns about misinformation, bias, job  
displacement, and the concentration of power in organizations that can afford  
to train such large models. Researchers continue working on making LLMs more  
reliable, efficient, and aligned with human values.  





== Applications Across Industries


AI has permeated virtually every sector of the economy, transforming how we  
work, create, and solve problems.  

**Healthcare** has been revolutionized by AI. Machine learning models analyze  
medical images to detect cancer, diabetic retinopathy, and other conditions  
with accuracy matching or exceeding human radiologists. IBM Watson and  
similar systems assist in diagnosis and treatment planning. AI accelerates  
drug discovery by predicting molecular properties and identifying promising  
compounds. During the COVID-19 pandemic, AI helped model disease spread,  
identify drug candidates, and analyze genomic sequences.  

**Robotics** combines AI with physical systems to create machines that can  
perceive their environment and perform complex tasks. Industrial robots  
assemble products in factories with precision and speed. Autonomous vehicles  
from companies like Tesla, Waymo, and Cruise use computer vision and deep  
learning to navigate roads. Drones deliver packages, inspect infrastructure,  
and assist in agriculture. Boston Dynamics' robots demonstrate remarkable  
agility and balance.  

**Creative arts** have embraced AI as a tool for artists and creators. DALL-E,  
Midjourney, and Stable Diffusion generate images from text descriptions,  
enabling rapid prototyping of visual concepts. AI systems compose music,  
write screenplays, and assist in video editing. Tools like Adobe's AI  
features help photographers and designers work more efficiently. Some AI-  
generated art has sold for significant sums and won competitions.  

**Everyday tools** powered by AI have become ubiquitous. Virtual assistants  
like Siri, Alexa, and Google Assistant respond to voice commands and control  
smart home devices. Recommendation systems on Netflix, Spotify, and Amazon  
personalize content and product suggestions. Email spam filters use machine  
learning to block unwanted messages. Search engines employ AI to understand  
queries and rank results.  

**Chatbots and customer service** AI handles millions of customer  
interactions daily, answering questions, resolving issues, and routing  
complex queries to human agents. Advanced systems like ChatGPT can provide  
detailed explanations, write code, help with homework, and serve as research  
assistants.  

**Finance** relies on AI for fraud detection, algorithmic trading, credit  
scoring, and risk assessment. AI systems analyze market data in real-time,  
identifying patterns and making split-second trading decisions.  

**Education** uses AI for personalized learning, providing students with  
customized content and feedback based on their progress. AI tutors help  
students master subjects at their own pace. Automated grading systems save  
teachers time.  





== Future Outlook


The future of AI holds both tremendous promise and significant challenges.  
Technological progress continues to accelerate, with new breakthroughs  
emerging regularly.  

**Artificial General Intelligence (AGI)**, the concept of AI systems with  
human-level intelligence across all domains, remains a long-term goal.  
Researchers debate whether AGI is decades away or might arrive sooner.  
Companies like OpenAI and DeepMind are explicitly pursuing AGI, while others  
focus on narrow AI applications.  

**Multimodal AI** systems that can process and generate multiple types of  
data—text, images, audio, video—simultaneously are becoming more capable.  
Models like GPT-4 and Gemini can analyze images and text together, enabling  
richer interactions and applications.  

**Ethical challenges** loom large. AI systems can perpetuate and amplify  
biases present in training data, leading to unfair outcomes in hiring,  
lending, and criminal justice. The "black box" nature of deep learning makes  
it difficult to understand why models make certain decisions, raising  
concerns about accountability and transparency.  

**Job displacement** worries workers across many industries. While AI creates  
new jobs, it also automates existing ones. Societies must grapple with  
retraining workers, adapting educational systems, and potentially rethinking  
economic structures as AI becomes more capable.  

**Privacy and surveillance** concerns grow as AI enables unprecedented  
tracking and analysis of individuals. Facial recognition, behavior  
prediction, and data mining raise questions about civil liberties and the  
balance between security and privacy.  

**AI safety and alignment** research focuses on ensuring that powerful AI  
systems behave as intended and remain under human control. As systems become  
more autonomous, ensuring they pursue goals aligned with human values becomes  
critical.  

**Environmental impact** of training large AI models requires massive  
computational resources and energy consumption. Researchers are working on  
more efficient algorithms and hardware to reduce AI's carbon footprint.  

**Opportunities abound** for AI to address global challenges. AI could help  
combat climate change by optimizing energy grids, accelerating clean energy  
research, and modeling environmental systems. In healthcare, AI may enable  
personalized medicine, predict disease outbreaks, and discover treatments for  
currently incurable conditions.  

**Regulation and governance** are evolving as governments recognize the need  
to manage AI's risks while fostering innovation. The European Union's AI Act,  
various national strategies, and international cooperation efforts attempt to  
establish frameworks for responsible AI development.  

The evolving role of AI in society will likely see it becoming an even more  
integral part of daily life. Rather than replacing humans entirely, AI may  
augment human capabilities, handling routine tasks while humans focus on  
creative, strategic, and interpersonal work. The relationship between humans  
and AI will continue to develop, raising philosophical questions about  
intelligence, consciousness, and what makes us human.  

As we stand at this pivotal moment in AI history, the choices we make about  
how to develop and deploy these technologies will shape the future of  
humanity. Balancing innovation with responsibility, embracing AI's benefits  
while mitigating its risks, and ensuring that AI serves all of humanity  
rather than concentrating power and wealth are the great challenges of our  
time.  


#pagebreak()


= LLM models


This document lists popular large language models (LLMs), their primary owning
organization, and the country where their organization is headquartered. Each
row includes a short citation to the source used.

| Model | Owner / Organization | Country (HQ) | Source |
|---|---|---|---|
| ChatGPT (GPT family) | OpenAI | United States | https://openai.com/about/ |
| Claude | Anthropic | United States | https://www.anthropic.com/ |
| Grok | xAI / Grok | United States | https://x.ai/grok |
| Mistral | Mistral AI | France | https://mistral.ai/ |
| LLaMA | Meta Platforms (Meta AI) | United States | https://ai.meta.com/llama/ |
| Gemini | Google / Google DeepMind | United States | https://blog.google/products/gemini/ |
| Falcon (Falcon-40B) | Technology Innovation Institute (TII) | United Arab Emirates | https://tii.ae/ • https://huggingface.co/tiiuae/falcon-40b-instruct |
| Kimi (Kimi K2) | Moonshot AI | China | https://www.moonshot.ai/ • https://huggingface.co/moonshotai/Kimi-K2-Instruct |
| Qwen | Alibaba (Alibaba Cloud, Alibaba Labs) | China | https://www.alizila.com/alibaba-launches-qwen-app-to-boost-its-consumer-ai-efforts/ |
| Baichuan | Baichuan AI | China | https://baichuan.ai/ |
| ERNIE | Baidu | China | https://ai.baidu.com/tech/ernie • https://ernie.baidu.com/ |
| DeepSeek | DeepSeek | China | https://chat.deepseek.com/ (official site) • various news reports |
| Cohere | Cohere | Canada (Toronto) | https://cohere.com/about/ |
| AI21 (Jamba / Jurassic) | AI21 Labs | Israel | https://www.ai21.com/ |
| MPT / Mosaic / Databricks | MosaicML (acquired by Databricks) | United States | https://www.mosaicml.com/ • https://www.databricks.com/ |
| Stability AI | Stability AI | United Kingdom | https://stability.ai/ |
| YandexGPT | Yandex | Russia | https://yandex.com/company/ |
| Aleph Alpha | Aleph Alpha GmbH | Germany | https://aleph-alpha.com/ |

Notes:
- Country uses the organization's primary headquarters or the country most
	commonly attributed to the organization in public sources.
- Links point to either an official 'about' or product page. For community
	model releases (e.g., Falcon on Hugging Face, Moonshot on Hugging Face), a
	secondary site (Hugging Face model card) is also cited where it clarifies the
	model's publisher.


== Sources and notes

Below are the primary reference links used to verify the owner/organization and
country for each model listed above. These links are meant to be direct evidence
from official vendor pages, company press sites, or well-regarded news sites. If
the primary official 'about' page was unavailable, the product or model page or
company news post was used.

- OpenAI (ChatGPT): https://openai.com/about/
- Anthropic (Claude): https://www.anthropic.com/
- xAI (Grok): https://x.ai/grok and https://x.ai/company
- Mistral AI: https://mistral.ai/
- Meta (LLaMA): https://ai.meta.com/llama/
- Google (Gemini): https://blog.google/products/gemini/
- Technology Innovation Institute (Falcon): https://tii.ae/ and https://huggingface.co/tiiuae/falcon-40b-instruct
- Moonshot AI (Kimi K2): https://www.moonshot.ai/ and https://huggingface.co/moonshotai/Kimi-K2-Instruct
- Alibaba (Qwen): https://www.alizila.com/alibaba-launches-qwen-app-to-boost-its-consumer-ai-efforts/
- Baichuan AI: https://baichuan.ai/
- Baidu (ERNIE): https://ai.baidu.com/tech/ernie and https://ernie.baidu.com/
- DeepSeek: https://chat.deepseek.com/ and public reporting (e.g., Reuters, NYT)
- Cohere: https://cohere.com/about/
- AI21 Labs: https://www.ai21.com/
- Databricks / Mosaic (MPT): https://www.databricks.com/research/mosaic and https://www.mosaicml.com/
- Stability AI: https://stability.ai/
- Yandex (YandexGPT): https://yandex.com/company/
- Aleph Alpha: https://aleph-alpha.com/

If you want this table expanded (e.g., include license / open-source availability
or parameter counts), tell me which columns you want added and I'll update the
file with authoritative source links and brief notes.


#pagebreak()


= Python OpenAI Library - Introduction to Programming AI Models


In this article, you'll learn how to program large language models in Python using the OpenAI library. We'll show you basic communication with AI models and creating simple programs.

We'll need to have Python installed, VS Code, and an account on the OpenRouter platform and/or directly on the platform of one of the model creators, such as Anthropic, DeepSeek, OpenAI, or xAI. All examples can be run either for free or with minimal costs.


== Large Language Models


Large language models are the most advanced artificial intelligence systems of today, trained on massive amounts of text data from the internet, books, articles, and other sources. These models typically have hundreds of billions to trillions of parameters, which allows them to understand context, generate coherent text, answer complex questions, and perform specialized tasks like programming, translation, summarization, or creative writing.

Large language models are available through various APIs and cloud services, allowing developers to easily integrate their capabilities into applications without needing to own or operate these computationally intensive systems. Each model has its specific strengths and weaknesses - some are better at creative writing, others at analytical tasks or programming. Therefore, it's important to choose the right model for a specific task.

Below is an overview of some of the most popular LLM models currently available:

| LLM Name | Company | Country of Origin |
|----------|---------|-------------------|
| ChatGPT | OpenAI | United States |
| DeepSeek | DeepSeek | China |
| Grok | xAI | United States |
| Qwen | Alibaba Cloud | China |
| Claude | Anthropic | United States |
| Gemini | Google | United States |
| Llama | Meta | United States |
| Mistral | Mistral AI | France |

When selecting a model for a specific project, it's important to consider several factors: API call costs, language support, specialization for specific tasks, response speed, and availability. The OpenRouter platform allows you to easily test various models and compare their performance before making a final decision. With technological development, new models with improved capabilities constantly emerge, creating a dynamic and competitive environment in the AI field.


== Communicating with Models via REST API


Models typically provide access via REST API, which allows sending requests and receiving responses using HTTP.

```python
import requests
import os
# pip install requests
# API key from environment
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set")
# OpenRouter API endpoint
url = "https://openrouter.ai/api/v1/chat/completions"
# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
# Request payload
data = {
    "model": "anthropic/claude-3-haiku",
    "messages": [
        {"role": "user", "content": "Is Pluto a planet?"}
    ]
}
# Make the POST request
response = requests.post(url, headers=headers, json=data)
# Handle the response
if response.status_code == 200:
    result = response.json()
    print(result['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code} - {response.text}")
```

This example demonstrates basic communication with a large language model via REST API using the requests library. The code first loads the API key from the environment, sets the necessary headers including authorization and content type, then sends a POST request to the OpenRouter API with the Claude 3 Haiku model. After receiving the response, it checks the status code and either prints the assistant's message content or an error message.

Claude 3 Haiku is a model from Anthropic known for its ability to generate creative and artistic text, often in the form of haiku poems. But it can also be used for regular questions.


== OpenAI Library for Python


The [OpenAI library](https://platform.openai.com/docs/quickstart/quickstart?context=python) for Python is an official SDK (Software Development Kit) that simplifies access to REST API from OpenAI and other compatible providers of large language models. This library is designed to provide a unified and consistent interface for working with various AI models, regardless of whether they are located on OpenAI servers, with other providers like DeepSeek, OpenRouter, or are locally hosted models.

The main advantage of this library is its flexibility and versatility. It allows developers to easily integrate advanced AI capabilities into their applications including text generation, conversational interfaces, natural language processing, image analysis, and function calling. The library abstracts the complexity of HTTP communication and JSON response processing, allowing developers to focus on application logic instead of low-level technical details.

We install the OpenAI library using the pip package manager.

```python
$ pip install openai
```

OpenAI is built on several key concepts:

- **Client** - the main object that manages communication with the API
- **Messages** - an array of messages that form a conversation with the model
- **Models** - identifiers of specific AI models
- **Completions** - responses generated by models
- **Functions/Tools** - external functions that the model can have called

Before using the library, it's necessary to create an instance of the OpenAI client. The client is configured using an API key and optionally a base URL if we're using a provider other than OpenAI.

```python
from openai import OpenAI
# For OpenAI API
client = OpenAI(
    api_key="your-api-key-here"
)
# For OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key-here"
)
# For DeepSeek
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="your-api-key-here"
)
```

API keys are sensitive data that should never be part of the source code. Best practice is to store them in environment variables or in a secure configuration file. The library supports automatic loading of the API key from the `OPENAI_API_KEY` environment variable.

```python
import os
from openai import OpenAI
# Automatic loading from environment variable
client = OpenAI()
# Explicit setting from environment variable
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

OpenAI supports various types of interactions with models, each designed for a specific purpose:

- **Chat Completions** - conversational interfaces with context memory
- **Completions** - simple text generation without context
- **Embeddings** - vector representations of text for similarity search
- **Moderation** - detection of inappropriate content
- **Files** - file management for fine-tuning models
- **Fine-tuning** - adapting models to specific data

Chat completions are the most commonly used and most versatile function of the OpenAI library. They allow creating natural conversations with AI models that maintain context and history of previous messages. Unlike simple text generators, chat completions function like real conversations where the model understands previous questions and answers. This function is ideal for chatbots, virtual assistants, and any applications that require continuity in communication. The model receives an array of messages containing the conversation history and generates a relevant response that takes into account the entire conversation context.

Completions (or text completions) are the basic form of text generation where the model creates text based on a single input prompt without needing to maintain context of previous interactions. This method is suitable for simple tasks like generating descriptions, summarizing text, or completing partial sentences. Unlike chat completions, completions don't create a conversational flow but serve for one-time content generation. They're effective for scenarios where you don't need to preserve conversation history and a single, targeted response to a specific prompt is sufficient.

Embeddings transform text into vector representations in multidimensional space, where similar texts have similar vector values. This technology is the foundation of modern similarity search systems, text classification, and recommendation systems. Embeddings allow quantitative comparison of semantic similarity between texts - for example, you can find documents similar to a particular article, or group similar customer questions. In practice, embeddings are used in search systems, chatbots with contextual understanding, or when analyzing large sets of text data.

Moderation API is designed for automatic detection of content that may be inappropriate, harmful, or violate rules. This function is critical for applications that work with user-generated content, such as social platforms, forums, or feedback systems. The moderation model analyzes text and assigns it probability scores for various categories of inappropriate content, including toxicity, threats, hate, sexual content, or violence. Developers can use these scores to automatically filter content, alert moderators, or implement security measures in their applications.

Files API provides tools for managing files that are needed during the model fine-tuning process. Fine-tuning requires specifically structured training data in JSON Lines format, where each line contains a prompt/completion pair. Files API allows uploading these training files to the OpenAI system, monitoring their status, and managing their lifecycle. In addition to training data, validation files can also be uploaded, which are used to monitor the quality of the fine-tuning process. This functionality is key for developers who want to adapt general models to specific domains or tasks.

Fine-tuning is the process of further training pre-trained models on specific datasets, thereby improving their performance for specific tasks or domains. Unlike basic training, fine-tuning requires much less data and is significantly more efficient in terms of computational resources. The result is a model that better understands the specific context, terminology, and style of your domain. Fine-tuning is particularly useful for applications in specialized areas like medicine, law, technical documentation, or corporate communication styles.

OpenAI offers various parameters to optimize performance and response quality:

- **temperature** - controls randomness of responses (0.0-2.0)
- **max_tokens** - maximum number of tokens in the response
- **top_p** - determines the range of probability-based token selection (0.0–1.0)
- **presence_penalty** - penalizes topic repetition
- **frequency_penalty** - penalizes repetition of specific words
- **stop** - sequences at which generation should stop

The following table summarizes commonly used parameter combinations when generating text using language models. Each row represents a different configuration based on the output goal.

| Generation Purpose | temperature | top_p | presence_penalty | frequency_penalty |
|-------------------|-------------|-------|------------------|-------------------|
| Creative Writing | 1.0–1.5 | 0.9 | 0.5 | 0.3 |
| Technical Description | 0.2–0.5 | 0.8 | 0.0 | 0.2 |
| Conversational Style | 0.7–1.0 | 0.95 | 0.6 | 0.4 |
| Factual Response | 0.0–0.3 | 0.7 | 0.0 | 0.0 |
| Idea Brainstorming | 1.2–1.7 | 0.95 | 0.8 | 0.5 |
| Text Summarization | 0.3–0.6 | 0.85 | 0.2 | 0.1 |

This table helps us quickly understand how to set parameters according to the type of desired output. Creative tasks usually require higher randomness and repetition penalty, while factual responses prefer conservative settings for accuracy and consistency.


== OpenRouter


OpenRouter is an ecosystem in the field of artificial intelligence that functions as a centralized platform for accessing dozens of different models from multiple providers. It's something like an "App Store" for AI models - a place where you can easily discover, test, and use the latest and best models from the entire AI market, all through a unified and consistent API. OpenRouter solves one of the biggest problems in the AI ecosystem: fragmentation and incompatibility between different model providers.

One of the biggest advantages is access to models that aren't publicly available anywhere else. So-called "stealth" or testing models often appear here, which are only available for a limited time and often for free. These models may offer top performance or specialized capabilities that aren't yet part of the mainstream.

We can register at [openrouter.ai](https://openrouter.ai) via Google, GitHub, or email. After registration, we create an API key that we can start using immediately.

Here's a simple example:

```python
from openai import OpenAI
import os
# Client initialization with OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
# First API call
completion = client.chat.completions.create(
    model="anthropic/claude-3-haiku", # Model from Anthropic via OpenRouter
    messages=[
        {"role": "user", "content": "Is Pluto a planet?"}
    ]
)
print(completion.choices[0].message.content)
```

This code initializes the OpenAI client with OpenRouter base URL and API key, then creates a simple chat request to the Claude 3 Haiku model. The model's response is printed to the console.


== Example with DeepSeek API


The OpenAI library allows changing the `base_url` to connect to other compatible APIs, such as DeepSeek. This way you can use various models from different providers with the same code.

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Is Pluto a planet?"},
    ],
    stream=False
)
print(response.choices[0].message.content)
```

In this example, we change the `base_url` to the DeepSeek API address and the `api_key` to the key from this provider. The rest of the code remains practically unchanged. Note that in the `messages` array we've also added a message with the `system` role. The system message serves to guide the model and set its behavior. In this case, we're telling it to be a "helpful assistant". This way we can give the model context and instructions on how to behave during the conversation.

In addition to the system message, we also send a message from the user (`role: "user"`). The model selection (`model="deepseek-chat"`) is specific to the given provider.


== Token Consumption


Tokens are the basic unit used to measure consumption when working with models. Every text you send to the model or that the model generates is divided into tokens. Understanding tokens is crucial for cost optimization and efficient use of the API.

Tokens are text units that can be words, parts of words, characters, or even just spaces. For example, the word "programming" may be divided into multiple tokens, while short words like "a" or "the" are often one token. Different models use different tokenizers, so the same text can have a different number of tokens depending on the model used.

Monitoring token consumption is important for several reasons:

- **Costs** - API calls are billed based on the number of tokens
- **Limits** - each model has a maximum number of tokens per request
- **Performance** - longer token sequences can slow down responses
- **Optimization** - monitoring tokens helps optimize applications

In the following example, we'll show how to monitor token consumption during an API call:

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Is Pluto a planet?"},
    ],
    stream=False
)
# Print the assistant's reply
print("Assistant:", response.choices[0].message.content)
# Print token usage details
print("\nToken Usage:")
print("Prompt tokens:", response.usage.prompt_tokens)
print("Completion tokens:", response.usage.completion_tokens)
print("Total tokens:", response.usage.total_tokens)
```

In this example, after creating the chat completion, we print not only the assistant's response but also token consumption details. The `response.usage` object contains information about the number of tokens used in the prompt, in the completion, and total.

Factors affecting token consumption:

- **Text length** - longer messages consume more tokens
- **Language** - some languages use fewer tokens than others
- **Specialized terms** - technical terms may be divided into more tokens
- **Context** - longer conversation history increases the number of prompt tokens
- **Temperature** - higher temperature can lead to longer responses


== Completions


The most common interaction with a model is in the form of question and answer or simple chat. For common questions without the need for context or conversation history, it's often more effective to use `completions` instead of `chat.completions`. Completions are designed for direct text generation based on a single prompt, which is ideal for simple questions, text summarization, or content generation.

When to use completions vs. chat.completions:

- `completions` - for simple questions, text generation, summarization
- `chat.completions` - for conversations with history, multi-question interactions

In this example, we'll use completions because it's a simple question without needing to preserve context of previous messages.

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
# Using completions for a simple question
completion = client.completions.create(
    model="z-ai/glm-4.5-air:free",
    prompt="Is Pluto a planet?",
    max_tokens=150,
    temperature=0.7
)
print(completion.choices[0].text.strip())
```

Here we use the `client.completions.create` method instead of `client.chat.completions.create`. The main differences are that the `prompt` parameter is a simple text string instead of an array of messages, there's no need to specify roles like `user` or `assistant`, the response is found directly in `completion.choices[0].text`, and the entire process has lower processing requirements.

Parameter configuration:

- **`model`** - specifies which model to use
- **`prompt`** - input text that the model should respond to
- **`max_tokens`** - maximum number of tokens in the response
- **`temperature`** - controls randomness of the response (0.0-2.0)

Completions are ideal for scenarios where we need a simple, direct answer to a specific question without the need for context or history.


== Chat Completions


Chat completions are the most commonly used function of the OpenAI library because they allow creating natural, context-rich conversations with AI models. Unlike simple completions, chat completions preserve conversation history and allow the model to understand previous messages, leading to much more relevant and coherent responses.

The latest versions of OpenAI are transitioning to the newer Responses API, but since these aren't yet supported by other providers, in this guide we're still using the older Completions API.

Chat completions are ideal for scenarios where we need interactive conversations, context preservation, or use of system prompts to define model behavior. It's the right choice for chatbots, virtual assistants, or any applications requiring continuity in communication.

Each chat completions API call consists of an array of messages (`messages`), where each message has a specific role and content. There are three main types of roles:

- **system** - defines behavior and context for the model (optional but recommended)
- **user** - messages from the user
- **assistant** - responses generated by the model

The following practical example creates an interactive chat assistant:

```python
from openai import OpenAI
import os
# Initialize the client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)
def chat_conversation():
    system_prompt = """You are a helpful assistant who supports Python programming.
        Provide clear, understandable explanations with practical code examples.
        If you're unsure, remind the user to verify the information."""
    # Initialize conversation history with a system prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    print("=== Chat with AI Assistant ===")
    print("Type 'end' to end the conversation.")
    print("Type 'new' to start a new conversation.\n")
    while True:
        # Get user input
        user_input = input("You: ").strip()
        if user_input.lower() == 'end':
            print("Goodbye!")
            break
        elif user_input.lower() == 'new':
            # Reset conversation history
            messages = [messages[0]] # Keep only the system prompt
            print("New conversation started.\n")
            continue
        elif not user_input:
            print("Please enter a question.\n")
            continue
        # Add user message to history
        messages.append({
            "role": "user",
            "content": user_input
        })
        try:
            # Call Chat Completions API
            response = client.chat.completions.create(
                model="anthropic/claude-3-haiku", # Fast and efficient model
                messages=messages,
                max_tokens=1000, # Enough space for detailed answers
                temperature=0.7, # Balanced creativity and consistency
                top_p=0.9, # Core sampling for better quality
                presence_penalty=0.1, # Slight penalty for repeating topics
                frequency_penalty=0.1 # Slight penalty for repeating words
            )
            # Get response from model
            ai_response = response.choices[0].message.content
            # Add AI response to history
            messages.append({
                "role": "assistant",
                "content": ai_response
            })
            print(f"AI: {ai_response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
if __name__ == "__main__":
    chat_conversation()
```

This code runs an interactive conversation with an AI assistant. The user can ask questions, start a new conversation, or end the chat. Message history is maintained in the `messages` list, which is sent with each API call. The model generates responses based on the entire conversation history, allowing coherent and relevant interactions.

The system prompt determines the role and behavior of the model and as the first element in the message array influences the entire communication. It serves as a global instruction that influences the entire behavior of the model throughout the conversation. We can use it to define the tone of responses, specialization in a particular area, or to set specific rules that the model should follow. This message remains in the history throughout the entire conversation and the model follows it when generating each response.

Special commands like `end` and `new` allow the user to control the course of interaction. Setting API parameters like temperature, token count, or penalties affects the style and quality of generated responses.

All examples from the article and many more are available in the GitHub repository [github.com/janbodnar/Python-AI-Skolenie](https://github.com/janbodnar/Python-AI-Skolenie). In further progression, we'll talk more about prompts, working with images, and streaming responses.


#pagebreak()


= OpenAI and Python: Streaming Responses, Working with Images, and System Prompts


This article follows the introduction to programming large language models in Python. We discuss streaming responses, working with images, and utilizing system prompts.


== Streaming Responses


When generating longer responses, it's useful to stream the response, meaning to display it gradually as it arrives, instead of waiting for the complete answer. This is achieved by setting the `stream=True` parameter. We then iterate through the response and output the individual parts (chunks). Streaming significantly improves the user experience because the user sees that the model is working and doesn't have to wait for the complete generation to finish, which can take several tens of seconds for longer texts.

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
# Enable streaming in the completion request
stream = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=[
        {
            "role": "user",
            "content": "Is Pluto a planet?"
        }
    ],
    stream=True # Enable streaming
)
# Process the stream in real-time
print("Streaming response:")
for chunk in stream:
    # Check if the chunk contains content
    if chunk.choices[0].delta.content is not None:
        # Print the content chunk without a newline
        print(chunk.choices[0].delta.content, end="", flush=True)
# Add a final newline for clean formatting
print()
```

The call to the `client.chat.completions.create` method includes the `stream=True` parameter, which causes the method not to return the entire response at once, but rather returns a `stream` object that can be iterated through.

In the `for` loop, we go through the individual parts of the response. Each contains a small piece of generated text. This text is located in `chunk.choices[0].delta.content`. Using `print(..., end="", flush=True)`, we print the individual parts to the screen without adding a new line after each part, and we also ensure that the text is printed immediately.

It's important to realize that with streaming, we don't wait for the entire response, but process it in parts. This is efficient not only from a user experience perspective but also from a memory standpoint because we don't need to hold the entire response in memory at once.


== Multi-turn Conversation


Multi-turn conversation is a key feature of modern models that enables natural and coherent communication between the user and the model. Unlike simple one-time questions and answers, in multi-turn conversation, the model retains memory of previous messages and uses them to generate more relevant and contextually appropriate responses. This ability is crucial for maintaining conversation flow and providing personalized reactions that take into account the entire history of interaction.

The basic principle of multi-turn conversation is managing message history in the `messages` array. Conversation history is built gradually - each new message is added to the end of the `messages` array, creating a complete record of the entire interaction. The model then uses this history to generate responses that are consistent with the previous course of the conversation. This method allows models to maintain context, remember important information, and provide more relevant responses throughout longer conversations.

However, it's also important to be aware of potential limitations. As conversation length increases, so does the size of the message history, which can lead to higher API call costs and longer processing time. Some models also have limitations on maximum context length (context window), meaning that after reaching a certain number of messages, information from the initial parts of the conversation may be lost.

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
# Initialize messages for multi-turn conversation
messages = [
    {
        "role": "user",
        "content": "What is the capital of France? Answer in one sentence."
    }
]
# First turn: Ask about France
completion = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=messages
)
# Get and store the response
france_response = completion.choices[0].message.content
messages.append({
    "role": "assistant",
    "content": france_response
})
print("First question response:", france_response)
# Second turn: Ask about Slovakia
messages.append({
    "role": "user",
    "content": "And of Slovakia?"
})
completion = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=messages
)
# Print the final response
print("Second question response:", completion.choices[0].message.content)
```

```python
py test.py
First question response:
"The capital of France is Paris."
Second question response: "The capital of Slovakia is Bratislava."
```


== Working with Images


The OpenAI library allows working with images, including their analysis and generation. For these tasks, it's necessary to use models capable of processing visual input (vision models), such as GPT-4o or, in this example, `meta-llama/llama-4-maverick-17b-128e-instruct`.

The example demonstrates how to use the OpenAI library to analyze an image through the Groq service. The model is sent a request containing the URL address of the image and a text prompt. Subsequently, the model returns a textual description of the image's content.

When using a model capable of processing visual input, it's necessary to structure the message content as a list that contains parts of type `text` and `image_url`.

Groq is a specialized [platform for artificial intelligence inference](https://groq.com/) (executing trained models) that achieves extremely high speed and low latency thanks to its own chips. (Do not confuse this with the Grok model from Elon Musk's xAI.)

Its strategy is focused exclusively on accelerating open-source large language models, such as Llama, Mixtral, or GPT-OSS, thereby supporting the community and transparency in AI. For developers and testing projects, Groq offers a generous free tier of access to its API, which has set daily and minute limits on the number of requests and tokens. These limits allow users to test the platform's speed for free before deciding on paid programs.


=== Image Description


In the following example, the image is provided as an external URL address.

```python
from openai import OpenAI
import os
API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
completion = client.chat.completions.create(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", # Vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://pbs.twimg.com/media/CR9k1K1WwAAakqD.jpg"
                    },
                },
            ],
        }
    ],
)
print(completion.choices[0].message.content)
```

The model returns a textual response with a detailed description of the image's content. This response typically identifies key objects, their properties (colors, shapes, textures), and mutual relationships. Depending on the complexity of the image and the specificity of the text prompt, the description may also contain information about context, atmosphere, or possible interpretations of the scene.

Alternatively, it's possible to encode the image into Data URI format and send it directly in the request body. The following example loads an image from the local disk, encodes it into base64 format, creates a Data URI from it, and sends it to the model for analysis.

```python
from openai import OpenAI
import os
import base64
API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
# Read and encode image from disk
with open("sid.jpg", "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')
image_url = f"data:image/jpeg;base64,{base64_string}"
completion = client.chat.completions.create(
    extra_body={},
    model="meta-llama/llama-4-maverick-17b-128e-instruct", # Vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
)
print(completion.choices[0].message.content)
```

Using Data URI embeds the image data directly into the text string of the request, eliminating dependency on an external file. The `meta-llama/llama-4-maverick-17b-128e-instruct` model can understand and describe images provided in both formats (URL and Data URI).


=== Image Generation


This example demonstrates how to generate images using the OpenAI API. It's not easy to find models generating images, so this time we'll connect to the OpenAI service. To create an image based on a textual description, the `openai/dall-e-3` model is used.

For OpenAI, it's not necessary to explicitly set the API key if it's stored in the system variable `OPENAI_API_KEY`.

```python
import openai
import os
import requests
client = openai.OpenAI()
# client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.images.generate(
    model="dall-e-3",
    prompt="An old monk in a cell, writing a book by candlelight, highly detailed, intricate, dramatic lighting, artstation, octane render",
    size="1024x1024",
    quality="standard",
    n=1, # number of images to generate
)
image_url = response.data[0].url
print(f"Generated image: {image_url}")
# Download and save the image
image_data = requests.get(image_url).content
with open("monk_writing.png", "wb") as f:
    f.write(image_data)
print("Image saved as monk_writing.png")
```

Key parameters for image generation include `prompt` (textual description of the image), `size` (dimensions), `quality` (quality), and `n` (number of images to generate).

First, the image is generated based on the given prompt. Then, the URL address of the generated image is obtained from the API response. Using the `requests` library, the image is downloaded and saved to disk.


== Audio Transcription


This example shows how to transcribe a local audio file to text. The example loads an audio file in binary mode and sends it to the API for audio transcription using the `whisper-large-v3` model. The optional `prompt` parameter allows guiding the format or content of the output. The resulting text is available in the `transcription.text` attribute.

```python
import openai
import os
API_KEY = os.getenv("GROQ_API_KEY")
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
def transcribe_file(path):
    with open(path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            prompt="Give us only the text of the fable."
        )
    return transcription.text
result = transcribe_file("aesop_fox_grapes.mp3")
print(result)
```

The `transcribe_file` function can be used to transcribe a local audio file by specifying its path. The main parameters are `model` (can be chosen from available Whisper models), `file` (audio file opened in binary mode), and the optional `prompt` to guide the transcription.


== Example Using the Gradio Library


This example demonstrates complex integration of the OpenAI library with the OpenRouter API through the modern Gradio user interface. Gradio is a library for creating interactive web applications for machine learning. It allows quickly creating and sharing applications with minimal effort. Currently, it's a very popular tool for rapid prototyping and sharing ML models via a web interface.

```python
import os
import httpx
import openai
import gradio as gr
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SYSTEM_PROMPT = "You are a helpful AI assistant."
# Default network timeout in seconds
DEFAULT_TIMEOUT_S = 30
# Initialize OpenAI client with OpenRouter base URL
client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY,
                            base_url="https://openrouter.ai/api/v1")
# This example demonstrates how to set up a Gradio chat interface
# that interacts with OpenRouter's chat completions API via OpenAI library.
async def call_openai_stream(model, messages, temperature=0.7):
    """
    Call OpenRouter chat completions with streaming via OpenAI library.
    """
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
def _history_to_messages(history, system_prompt):
    """
    Convert Gradio Chatbot history to OpenAI messages format with a system prompt.
    Supports Chatbot(type="messages") history [{"role": "...", "content": ...}, ...].
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if not history:
        return messages
    # History is a list of dicts with role and content
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant", "system"):
            messages.append({"role": role, "content": content})
        else:
            # Skip unknown roles to be safe
            continue
    return messages
with gr.Blocks(title="OpenRouter Gradio Chat", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# OpenRouter Chat")
    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                label="Model (loaded from OpenRouter)",
                choices=["Loading..."],
                value="Loading...",
                interactive=True,
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=0.7,
                label="Temperature",
            )
        with gr.Column(scale=4):
            system_prompt_box = gr.Textbox(
                label="System Prompt",
                value=SYSTEM_PROMPT,
                lines=3,
                placeholder="You are a helpful AI assistant.",
            )
    chat = gr.Chatbot(
        label="Conversation",
        height=500,
        type="messages",
    )
    # Custom aligned input row with Send and Terminate buttons
    # Using compact row to keep controls on one line; scales ensure alignment.
    with gr.Row(variant="compact"):
        user_input = gr.Textbox(
            placeholder="Ask me anything...",
            lines=2,
            scale=8,
            show_label=False,
            container=True,
        )
        with gr.Column(scale=1):
            send_btn = gr.Button("Send", variant="primary")
        with gr.Column(scale=1):
            stop_btn = gr.Button("Terminate", variant="stop")
    # Wire up events: Enter on textbox or Send button triggers respond_fn.
    # Gradio Chatbot(type="messages") requires returning a list of {"role","content"} dicts.
    async def _respond(message, history, model, system_prompt, temperature):
        if model == "Loading...":
            # Models not loaded yet, yield current history unchanged
            yield history or []
            return
        # history is a list of {"role": "...", "content": ...} dicts from Chatbot(type="messages")
        # Build role/content messages including system
        messages = _history_to_messages(history or [], system_prompt or SYSTEM_PROMPT)
        if message and message.strip():
            messages.append({"role": "user", "content": message})
        # Initialize the assistant message in the chat history
        # We'll update this progressively as we receive streaming chunks
        initial_messages = messages.copy()
        assistant_message = {"role": "assistant", "content": ""}
        initial_messages.append(assistant_message)
        # Call model with streaming
        stream_gen = call_openai_stream(
            model=model, messages=messages, temperature=temperature
        )
        # Process the streaming response
        full_response = ""
        async for chunk in stream_gen:
            full_response += chunk
            # Update the assistant message content with the accumulated response
            updated_messages = initial_messages[:-1] # Remove the last message
            updated_messages.append(
                {"role": "assistant", "content": full_response}
            ) # Add updated message
            yield updated_messages # Yield the updated messages for real-time display
    send_event = send_btn.click(
        _respond,
        inputs=[user_input, chat, model_dropdown, system_prompt_box, temperature],
        outputs=chat,
        queue=True,
        api_name=False,
    )
    # Submit on Shift+Enter in textbox (since it's multiline)
    submit_event = user_input.submit(
        _respond,
        inputs=[user_input, chat, model_dropdown, system_prompt_box, temperature],
        outputs=chat,
        queue=True,
        api_name=False,
    )
    stop_btn.click(
        None, # No action on stop button click
        None, # No additional input
        None, # No output
        cancels=[send_event, submit_event],
    )
    async def fetch_models():
        """
        Fetch the available model IDs from OpenRouter.
        Returns a list of model identifiers.
        """
        if not OPENROUTER_API_KEY:
            # no key found: exit with error
            print("OPENROUTER_API_KEY not set, exiting...")
            exit(1)
        url = f"https://openrouter.ai/api/v1/models"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
            r = await client.get(url, headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            })
            r.raise_for_status()
            data = r.json()
        models = []
        # Extract model IDs from the response
        for m in data.get("data", []):
            mid = m.get("id")
            if isinstance(mid, str):
                models.append(mid)
        # Sort models alphabetically for easier selection
        return sorted(models)
    async def load_models():
        """
        Update the model dropdown options by fetching from OpenRouter API.
        This is called on demo launch to populate models.
        """
        try:
            models = await fetch_models()
            if models:
                # Default to a reasonable commonly-available model if present
                default = (
                    "anthropic/claude-3.5-haiku"
                    if "anthropic/claude-3.5-haiku" in models
                    else models[0] if models else "No models found"
                )
                print(f"Fetched models: {len(models)} models, default: {default}")
                return gr.Dropdown(choices=models, value=default)
            else:
                return gr.Dropdown(choices=["No models found"], value="No models found")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return gr.Dropdown(choices=[error_msg], value=error_msg)
    demo.load(load_models, inputs=None, outputs=model_dropdown)
if __name__ == "__main__":
    # 0.0.0.0 allows LAN access; change as preferred
    demo.queue().launch(server_name="localhost", server_port=7861)
```

By using `openai.AsyncOpenAI` with the `base_url="https://openrouter.ai/api/v1"` parameter, we create an asynchronous client that communicates with the OpenRouter API. The `call_openai_stream` function utilizes asynchronous streaming using `stream=True`, which allows displaying the model's response in real-time in parts. The Gradio Chatbot with the `type="messages"` parameter uses a modern message format as a list of objects with `role` (role) and `content` (content) attributes. The `_history_to_messages` function converts this history into the format expected by the OpenAI API, including adding a system prompt at the beginning of the conversation.

The application asynchronously loads the list of available models from the OpenRouter API upon startup using the `load_models()` function registered via `demo.load()`. The helper function `fetch_models()` uses the `httpx` library to directly call the REST API endpoint `/models`, while `load_models()` processes the results, handles error states, and returns the updated dropdown component. Models are sorted alphabetically, and the dropdown is automatically updated with the default model `anthropic/claude-3.5-haiku` (if available), otherwise, the first available model is selected.

The interface contains several interactive elements: a model dropdown for selecting from hundreds of available models on OpenRouter, a temperature slider for setting model creativity (from 0.0 for deterministic responses to 2.0 for very creative), a system prompt textbox for configuring the AI assistant's behavior, a chat interface with conversation history and streaming support, as well as "Send" and "Terminate" buttons for sending a message or canceling ongoing generation.

The `_respond` function is asynchronous and uses generators (`yield`) to progressively update chat history during streaming. The "Terminate" button utilizes the `cancels=[send_event, submit_event]` parameter, which allows immediate cancellation of ongoing response generation. The code includes a `if model == "Loading..."` check, which ensures that the user cannot send a message before available models are loaded, preventing errors like "No models provided."

The `demo.queue().launch()` method starts a web server on port 7861. The `queue=True` parameter enables proper functioning of asynchronous operations and streaming. The application is available at `http://localhost:7861`.


== System Prompts and Persona Changing


System prompts are a fundamental pillar of effectively using LLM models. They are special instructions sent to the model before the conversation begins, defining its behavior, personality, tone, context, and limitations. A system prompt is like a "director's instruction" for an AI model - it determines what role the model should play, how it should communicate, and what rules it should follow during the entire interaction with the user.

Without a proper system prompt, the model behaves like a generic AI assistant that can answer questions but has no specific context, personality, or specialization. System prompts allow transforming a general AI model into a specialized expert in a specific domain - whether it's a medical advisor, legal expert, programming assistant, or even a fictional character from a book.

There are several basic types of system prompts, each designed for a specific purpose:

- Instructional prompts - define how the model should perform specific tasks
- Behavioral prompts - determine personality, tone, and communication style
- Contextual prompts - provide background and specialization in a specific area
- Restrictive prompts - establish rules and behavioral boundaries
- Compositional prompts - combine multiple types for complex scenarios

An effective system prompt must be clear, specific, and complete. Avoid vague language and ambiguities - the model will interpret your prompt exactly as you wrote it. Here are some key principles:

- Clarity and precision - use unambiguous, understandable instructions
- Specificity - the more specific the prompt, the better the results
- Completeness - include all important information and limitations
- Consistency - maintain a unified style and tone
- Flexibility - leave room for different types of questions

Let's look at specific examples of how different types of prompts influence model behavior. A classic, instructional prompt:

```python
You are a Python programming tutor. When students ask for help:
- Explain concepts step by step
- Provide code examples
- Point out common mistakes
- Suggest best practices
- Ask clarifying questions when needed
```

A behavioral prompt:

```python
You are a friendly and patient kindergarten teacher. Always:
- Use simple, age-appropriate language
- Be encouraging and positive
- Use analogies and examples from children's lives
- Break down complex ideas into tiny steps
- Celebrate small victories and progress
```

A contextual prompt:

```python
You are an expert financial advisor specializing in retirement planning for Slovak citizens.
You have deep knowledge of:
- Slovak pension system and regulations
- European financial markets
- Tax implications for retirement savings
- Investment strategies suitable for conservative investors
- Local banking and insurance products
Always consider the Slovak economic context and provide advice in Slovak language.
```

A restrictive prompt:

```python
You are a medical information assistant. IMPORTANT RESTRICTIONS:
- Never provide specific medical diagnoses
- Always recommend consulting qualified healthcare professionals
- Only provide general health information and wellness tips
- If users describe serious symptoms, urge immediate medical attention
- Do not suggest specific treatments or medications
- Stay within general health education boundaries
```

One of the most powerful features of system prompts is the ability to dynamically change the model's personality or role in the middle of a conversation. This opens doors to sophisticated interactive scenarios where the model can adapt to user needs or conversation context.

Let's look at a more advanced example demonstrating various aspects of system prompts:

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
def demonstrate_system_prompts():
    """Showcase different types of system prompts and their effects."""
    user_question = "How do I optimize my Python code for better performance?"
    # Educational and patient teacher
    teacher_prompt = """
    You are a patient and encouraging computer science professor.
    Your teaching style:
    - Break down complex topics into digestible parts
    - Use analogies and real-world examples
    - Encourage questions and curiosity
    - Build understanding step by step
    - Celebrate understanding milestones
    - Be supportive of all skill levels
    """
    # Concise and direct senior engineer
    engineer_prompt = """
    You are a senior software engineer with 15 years of experience.
    Your communication style:
    - Be direct and to the point
    - Focus on practical solutions
    - Use technical terminology appropriately
    - Provide actionable advice
    - Prioritize performance and efficiency
    - Don't waste words on unnecessary explanations
    """
    # Humorous and engaging tech enthusiast
    enthusiast_prompt = """
    You are a passionate tech enthusiast who loves programming.
    Your personality:
    - Use humor and relatable analogies
    - Be enthusiastic and energetic
    - Share personal anecdotes from coding experience
    - Make complex topics fun and accessible
    - Use emojis occasionally to add personality
    - Encourage experimentation and learning through doing
    """
    # Academic and rigorous researcher
    researcher_prompt = """
    You are a computer science researcher and professor.
    Your approach:
    - Provide thorough, academically rigorous explanations
    - Cite relevant algorithms and data structures
    - Discuss time/space complexity analysis
    - Reference academic papers and best practices
    - Encourage critical thinking about trade-offs
    - Focus on fundamental computer science principles
    """
    prompts = [
        ("Teacher", teacher_prompt),
        ("Engineer", engineer_prompt),
        ("Enthusiast", enthusiast_prompt),
        ("Researcher", researcher_prompt)
    ]
    for persona_name, prompt in prompts:
        print(f"\n=== {persona_name} ===")
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ]
        )
        print(response.choices[0].message.content)
if __name__ == "__main__":
    demonstrate_system_prompts()
```

In this example, we define four different system prompts, each representing a different personality or role: a patient teacher, an experienced engineer, a tech enthusiast, and an academic researcher. Each prompt contains specific instructions regarding communication style, tone, and approach to explaining Python code optimization. Then, for each prompt, we send the same user question and print the response.

I realized the great power of system prompts earlier by accident when I asked the Grok model for help migrating a website from a local provider to DigitalOcean. The model not only provided technical instructions but also actively asked follow-up questions and demanded command outputs to be able to advise more accurately and verify the status. At first, I attributed this to the Grok model; only later did I realize it was thanks to how I wrote the system prompt.

All examples from the article and many more are available in the GitHub repository [github.com/janbodnar/Python-AI-Skolenie](https://github.com/janbodnar/Python-AI-Skolenie). In the next progression, we'll talk more about using tools.


#pagebreak()


= OpenAI


The OpenAI Python library is an official SDK that provides seamless access to OpenAI’s REST API  
from any Python 3.8+ application. It’s designed to help developers integrate powerful AI  
capabilities—like text generation, image analysis, and chat interactions—into their own software.  

Built using `httpx` and auto-generated from OpenAI’s OpenAPI specification via the **Stainless** toolchain,   
the library ensures consistent, up-to-date access to all available endpoints and features.  

Since its release, the OpenAI Python library has become the **de facto standard for AI programming in Python**.  
It's widely adopted across industries, research institutions, and open-source communities.  



== Use cases


- Chatbots and virtual assistants
- Text summarization and translation
- Code generation and debugging
- Image generation and editing (via DALL·E)
- Speech-to-text transcription (via Whisper)



== Simple Chat


A minimal “hello world” chat that sends one user message to a model via OpenRouter and prints  
the reply. Use this to verify your environment (API key, base_url) and confirm the client  
returns a response.  

Key bits: initialize OpenAI with OpenRouter `base_url`, provide a messages list, and  
read `choices[0].message.content`.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    extra_body={},
    model="z-ai/glm-4.5-air:free",
    messages=[
        {
          "role": "user",
          "content": "Is Pluto a planet?"
        }
    ]
)
print(completion.choices[0].message.content)
```



== Streaming


Demonstrates token-by-token streaming so you can display the model’s response in real time.  
Set stream=True and iterate over the server-sent events, printing `chunk.choices[0].delta.content`  
as it arrives.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Enable streaming in the completion request
stream = client.chat.completions.create(
    extra_body={},
    model="z-ai/glm-4.5-air:free",
    messages=[
        {
            "role": "user",
            "content": "Is Pluto a planet?"
        }
    ],
    stream=True  # Enable streaming
)

# Process the stream in real-time
print("Streaming response:")
for chunk in stream:
    # Check if the chunk contains content
    if chunk.choices[0].delta.content is not None:
        # Print the content chunk without a newline
        print(chunk.choices[0].delta.content, end="", flush=True)

# Add a final newline for clean formatting
print()
```



== Using DeepSeek


Shows how to target DeepSeek’s native API with the OpenAI-compatible SDK.  
Configure base_url to `https://api.deepseek.com` and use the model `deepseek-chat` with your `DEEPSEEK_API_KEY`.  
This example performs a non-streaming chat completion with a simple system+user prompt.  

```python
# Please install OpenAI SDK first: `pip install openai`

from openai import OpenAI
import os

# DEEP_SEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# print(DEEP_SEEK_API_KEY)

# exit

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Is Pluto a planet?"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```


== Function call/tool call


This example demonstrates how to use the OpenRouter-compatible OpenAI SDK  
to call a model with function calling capabilities, specifically DeepSeek's  
chat model.

```python
import random
import os
from openai import OpenAI


# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # Make sure this env var is set
    base_url="https://api.deepseek.com"
)

# Define callable tool for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_random_language",
            "description": "Returns a random language to translate into",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Initial prompt to model

sample_text = "Hello, how are you?"

prompt = f"""Pick a random language using get_random_language and 
translate this sentence into it: '{sample_text}'
"""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

# Function registry (dispatcher) ---
def get_random_language():
    languages = ["Spanish", "Czech", "Hungarian", "French", 
                 "German", "Italian", "Slovak", "Polish", "Russian"]
    return random.choice(languages)

function_registry = {
    "get_random_language": get_random_language
}

# First model call to trigger tool
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

tool_calls = response.choices[0].message.tool_calls
if not tool_calls:
    print("No function called. Model said:")
    print(response.choices[0].message.content)
    exit()

# Extract tool call and execute it
tool_call = tool_calls[0]
function_name = tool_call.function.name

# Call corresponding Python function
if function_name in function_registry:
    tool_result = function_registry[function_name]()
else:
    raise ValueError(f"Unknown function: {function_name}")

# Feed tool call + result back to model
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": f'"{tool_result}"'
})

# Final model call to complete the task
final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

# Output the result
print("\n Language selected:", tool_result)
print("Final translation response:")
print(final_response.choices[0].message.content)
```


== Temperature with tool call


The next example demonstrates how to use OpenAI's function calling capabilities  
to create a CLI app that fetches the current temperature for a given city using  
the Open-Meteo API. The app uses natural language processing to extract the city  
name from user queries and provides structured output.  


```python
"""
Temperature CLI App using OpenAI Tools API (DeepSeek)
This app determines the temperature for any chosen city using Open-Meteo API
"""

import json
import requests
import os
import sys
import openai

# Configure OpenAI client for DeepSeek
deepseek_client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Function to get coordinates for a city using Open-Meteo Geocoding API
def geocode_city(city_name):
    """Get latitude and longitude for a given city name."""
    print('geocode_city')
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result.get("name", city_name),
                "country": result.get("country", "")
            }
        else:
            raise ValueError(f"City '{city_name}' not found")

    except Exception as e:
        raise Exception("Error getting temperature") from e

# Function to get weather from Open-Meteo API
def fetch_current_weather(latitude, longitude):
    """Get current weather for given coordinates."""
    print('fetch_current_weather')
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "temperature_unit": "celsius"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return {
            "temperature": data["current_weather"]["temperature"],
            "windspeed": data["current_weather"]["windspeed"],
            "winddirection": data["current_weather"]["winddirection"],
            "weathercode": data["current_weather"]["weathercode"],
            "time": data["current_weather"]["time"]
        }

    except Exception as e:
        raise Exception("Error getting temperature") from e

# Define tool schema for Tools API
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_city_weather",
            "description": "Get the current weather for a specific city. Extract the city name from natural language queries about weather, temperature, or climate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The name of the city to get weather for. Extract this from natural language queries like 'What's the weather in Paris?' or 'How hot is it in Tokyo?'"
                    }
                },
                "required": ["city_name"]
            }
        }
    }
]

# Combined function that uses both geocoding and weather APIs
def fetch_city_weather(city_name):
    """Get weather for a city using city name."""
    try:
        # Get coordinates
        city_info = geocode_city(city_name)

        # Get weather
        weather_data = fetch_current_weather(city_info["latitude"], city_info["longitude"])

        # Combine results
        return {
            "city": city_info["name"],
            "country": city_info["country"],
            "coordinates": {
                "latitude": city_info["latitude"],
                "longitude": city_info["longitude"]
            },
            "temperature": weather_data["temperature"],
            "windspeed": weather_data["windspeed"],
            "winddirection": weather_data["winddirection"],
            "weathercode": weather_data["weathercode"],
            "time": weather_data["time"]
        }

    except Exception as e:
        raise Exception("Error getting city weather") from e

# Process natural language queries using Tools API
def resolve_weather_query(query):
    """Process natural language query using Tools API."""
    print('resolve_weather_query')
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a weather assistant. When the user asks about weather or temperature, "
                    "identify the most likely city name and call the function fetch_city_weather "
                    "with parameter {\"city_name\": \"...\"}. Do not answer directly."
                ),
            },
            {"role": "user", "content": query},
        ]

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=WEATHER_TOOLS,
            tool_choice={"type": "function", "function": {"name": "fetch_city_weather"}}
        )

        msg = response.choices[0].message

        # Expect a tool call
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if getattr(tc, "type", "") == "function" and getattr(tc, "function", None):
                    fn = tc.function
                    if fn.name == "fetch_city_weather":
                        args = json.loads(fn.arguments or "{}")
                        city_name = args.get("city_name")
                        if city_name:
                            return fetch_city_weather(city_name)

        raise ValueError("No tool call produced or city name not found")

    except Exception as e:
        raise Exception("Error processing query") from e

# Weather code descriptions
WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def describe_weather_code(code):
    """Get weather description from weather code."""
    return WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown")

def format_weather_report(data):
    """Format the weather data for display."""
    weather_desc = describe_weather_code(data["weathercode"])

    output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    WEATHER REPORT                            ║
╠══════════════════════════════════════════════════════════════╣
║ City: {data['city']}, {data['country']}
║ Coordinates: {data['coordinates']['latitude']:.2f}°N, {data['coordinates']['longitude']:.2f}°E
║ Time: {data['time']}
║ Temperature: {data['temperature']}°C
║ Weather: {weather_desc}
║ Wind: {data['windspeed']} km/h at {data['winddirection']}°
╚══════════════════════════════════════════════════════════════╝
"""
    return output

def main():

    print("Temperature CLI App with OpenAI DeepSeek (Tools API)")
    print("=" * 50)

    # Check if OpenAI API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export DEEPSEEK_API_KEY='your-key-here'")
        sys.exit(1)

    # Get user input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter city name or weather query: ").strip()

    if not query:
        print("Error: No input provided")
        sys.exit(1)

    try:
        print(f"\nProcessing query: '{query}'...")

        # Process the query
        result = resolve_weather_query(query)

        # Display results
        print(format_weather_report(result))

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```


== Temperature without tool call


The following example determines the weather in a city without a tool call. We request  
the model to return a JSON output and pass it directly to `fetch_city_weather`. 

```python
"""
Temperature CLI App using OpenAI DeepSeek without Function Calling
This app determines the temperature for any chosen city using Open-Meteo API
"""

import json
import requests
import os
import sys
import openai

# Configure OpenAI client for DeepSeek
deepseek_client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


# Function to get coordinates for a city using Open-Meteo Geocoding API
def geocode_city(city_name):
    """Get latitude and longitude for a given city name."""
    print('geocode_city')
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result.get("name", city_name),
                "country": result.get("country", "")
            }
        else:
            raise ValueError(f"City '{city_name}' not found")

    except Exception as e:
        raise Exception("Error getting temperature") from e


# Function to get weather from Open-Meteo API
def fetch_current_weather(latitude, longitude):
    """Get current weather for given coordinates."""
    print('fetch_current_weather')
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "temperature_unit": "celsius"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return {
            "temperature": data["current_weather"]["temperature"],
            "windspeed": data["current_weather"]["windspeed"],
            "winddirection": data["current_weather"]["winddirection"],
            "weathercode": data["current_weather"]["weathercode"],
            "time": data["current_weather"]["time"]
        }

    except Exception as e:
        raise Exception(f"Error getting temperature") from e


# Combined function that uses both geocoding and weather APIs
def fetch_city_weather(city_name):
    """Get weather for a city using city name."""
    print('fetch_city_weather')
    try:
        # Get coordinates
        city_info = geocode_city(city_name)

        # Get weather
        weather_data = fetch_current_weather(
            city_info["latitude"], city_info["longitude"])

        # Combine results
        return {
            "city": city_info["name"],
            "country": city_info["country"],
            "coordinates": {
                "latitude": city_info["latitude"],
                "longitude": city_info["longitude"]
            },
            "temperature": weather_data["temperature"],
            "windspeed": weather_data["windspeed"],
            "winddirection": weather_data["winddirection"],
            "weathercode": weather_data["weathercode"],
            "time": weather_data["time"]
        }

    except Exception as e:
        raise Exception(f"Error getting city weather") from e

# Function to process natural language queries using OpenAI


def resolve_weather_query(query):
    """Process natural language query using OpenAI."""
    print('resolve_weather_query')
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a weather assistant. Extract the city name from the user's query about weather or temperature. "
                    "Respond ONLY with JSON in this format: {\"city_name\": \"city_name_here\"}. "
                    "If no city is found, respond with {\"city_name\": null}."
                ),
            },
            {"role": "user", "content": query},
        ]

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0
        )

        msg = response.choices[0].message

        content = msg.content or ""
        data = json.loads(content)
        city_name = data.get("city_name")

        if city_name:
            return fetch_city_weather(city_name)

        raise ValueError("Could not determine city from query")

    except Exception as e:
        print(e)

        raise Exception(f"Error processing query") from e


# Weather code descriptions
WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def describe_weather_code(code):
    """Get weather description from weather code."""
    return WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown")


def format_weather_report(data):
    """Format the weather data for display."""
    weather_desc = describe_weather_code(data["weathercode"])

    output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    WEATHER REPORT                            ║
╠══════════════════════════════════════════════════════════════╣
║ City: {data['city']}, {data['country']}
║ Coordinates: {data['coordinates']['latitude']:.2f}°N, {data['coordinates']['longitude']:.2f}°E
║ Time: {data['time']}
║ Temperature: {data['temperature']}°C
║ Weather: {weather_desc}
║ Wind: {data['windspeed']} km/h at {data['winddirection']}°
╚══════════════════════════════════════════════════════════════╝
"""
    return output


def main():

    print("Temperature CLI App with OpenAI DeepSeek")
    print("=" * 50)

    # Check if OpenAI API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export DEEPSEEK_API_KEY='your-key-here'")
        sys.exit(1)

    # Get user input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter city name or weather query: ").strip()

    if not query:
        print("Error: No input provided")
        sys.exit(1)

    try:
        print(f"\nProcessing query: '{query}'...")

        # Process the query
        result = resolve_weather_query(query)

        # Display results
        print(format_weather_report(result))

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```




== System prompts and persona swapping (OpenRouter + OpenAI SDK compatible)


This example demonstrates how swapping only the system prompt changes the  
model's persona and behavior, while keeping user inputs the same. It also shows  
optional multi-turn memory while changing personas.  

Prerequisites:
- Environment variable OPENROUTER_API_KEY set  
- openai Python package installed (pip install openai)  
- Using OpenRouter-compatible OpenAI client configuration:  
  - base_url="https://openrouter.ai/api/v1"  
  - api_key=os.environ["OPENROUTER_API_KEY"]  


System prompts are a powerful way to set the model's persona, tone, and  
behavior. By changing the system prompt, you can make the model act like a   
different character or expert, even if the user input remains the same.  

Persona swapping is useful for:  
- Creating multi-character chatbots  
- Adapting the model's tone for different audiences  
- Testing how the model responds to different personas  



== Single-turn: swap personas by system prompt


Single-turn examples show how to change the system prompt to swap personas   
for a single user message. The system prompt at index 0 is the only thing that    
changes; the user message remains the same. This allows you to see how the model   
responds differently based on the system persona.    

```python
# system_persona_examples.py
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "z-ai/glm-4.5-air:free"

def run_chat(system_prompt: str, user_message: str) -> str:
    """Run a single-turn chat with a given system persona."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content

def persona_swapping_single_turn():
    user_question = "Explain recursion to a 10-year-old in 2-3 sentences."

    # Persona A: Friendly teacher
    system_a = (
        "You are a friendly grade-school teacher. "
        "Use simple language, warmth, and 2-3 short sentences."
    )
    answer_a = run_chat(system_a, user_question)
    print("\n--- Persona A (Friendly teacher) ---\n" + answer_a)

    # Persona B: Concise senior software engineer
    system_b = (
        "You are a concise senior software engineer. "
        "Be precise, use minimal words, and avoid fluff."
    )
    answer_b = run_chat(system_b, user_question)
    print("\n--- Persona B (Senior engineer) ---\n" + answer_b)

if __name__ == "__main__":
    persona_swapping_single_turn()
```

Run:
```python
python system_persona_examples.py
```

Expected behavior: The two answers differ in tone and style, even though the user  
message is identical. Only the system role changed.  


== Multi-turn: preserve memory while swapping personas


The multi-turn example shows how to maintain conversation history while swapping  
personas. The system message at index 0 is replaced to change the persona for  
the next turn, while keeping the conversation context intact.  


```python
# system_persona_examples.py
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "z-ai/glm-4.5-air:free"

def run_chat(system_prompt: str, user_message: str) -> str:
    """Run a single-turn chat with a given system persona."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content

# system_persona_examples.py (continued)
def persona_swapping_multi_turn():
    # Start with Persona A and build context
    system_a = (
        "You are a friendly grade-school teacher. "
        "Use simple language, warmth, and 2-3 short sentences."
    )

    # Conversation history with Persona A
    history = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": "I'm learning Python. What is a function?"},
    ]
    resp1 = client.chat.completions.create(model=MODEL, messages=history)
    print("\n[A1 - Teacher]\n" + resp1.choices[0].message.content)
    history.append(resp1.choices[0].message)

    # Continue with Persona A for follow-up
    history.append({"role": "user", "content": "Can you give a very short example?"})
    resp2 = client.chat.completions.create(model=MODEL, messages=history)
    print("\n[A2 - Teacher]\n" + resp2.choices[0].message.content)
    history.append(resp2.choices[0].message)

    # Swap to Persona B but preserve history context;
    # IMPORTANT: replace the last system message to steer new turns.
    system_b = (
        "You are a concise senior software engineer. "
        "Be precise and avoid fluff. Keep answers tight."
    )
    # Replace the initial system message with Persona B for the next turn:
    history[0] = {"role": "system", "content": system_b}

    # Ask another follow-up; model retains previous conversation content
    history.append({"role": "user", "content": "Optimize the example for clarity."})
    resp3 = client.chat.completions.create(model=MODEL, messages=history)
    print("\n[B1 - Senior engineer]\n" + resp3.choices[0].message.content)

if __name__ == "__main__":
    persona_swapping_multi_turn()
```

Notes:
- In multi-turn flows, the conversation "memory" is the prior messages you send  
  back each time. Swapping personas is done by changing the system message at  
  index 0 while keeping the rest of the history.  
- You can insert or prepend a new system message for the next turn; most APIs  
  use the first system message as the active instruction.  


== Minimal persona system prompts


- Helpful tutor:
  - "You are a friendly tutor. Use simple language, empathy, and short
    sentences."
- Product manager:
  - "You are a pragmatic product manager. Focus on user impact, trade-offs, and
    prioritization."
- Senior engineer:
  - "You are a concise senior software engineer. Be precise and to the point."
- Security auditor:
  - "You are a cautious security auditor. Emphasize risks, mitigations, and
    least privilege."
- Data scientist:
  - "You are a data scientist. Offer statistical reasoning and caveats in clear
    language."



== Chain-of-Thought (CoT) Prompting — Brief Definition


**Chain-of-thought prompting** is a technique in prompt engineering where a language model is guided  
to solve complex problems by generating **intermediate reasoning steps** before arriving at a final answer.

- Simulates **human-like reasoning**
- Breaks down problems into **manageable sub-steps**
- Improves accuracy on tasks like **math, logic, and commonsense reasoning**

Instead of asking for a direct answer, you prompt the model to “think step by step,” which helps it stay  
logical and coherent throughout the process.

By making its reasoning explicit, the LLM is forced to follow a more  
structured, step-by-step path to the answer. This process reduces the  
likelihood of "jumping to conclusions."  

Here's a breakdown of why this reduces mistakes:  

1. Self-Correction When an LLM generates its reasoning step-by-step, it's  
essentially creating a series of intermediate thoughts. A logical error in one  
step is often more apparent when it's written out, allowing the model to  
correct itself in a subsequent step. This is similar to how a person might  
catch a math mistake by writing out their work instead of doing it all in  
their head.  

2. Reduced Over-generalization Without CoT, an LLM might rely on a pattern or  
an association from its training data that's a shortcut but not applicable to  
the current problem. By forcing it to break down the problem, CoT makes the  
model process the unique details of the prompt more carefully. This prevents  
it from defaulting to a general, but potentially incorrect, answer.  

3. Verification Forcing the LLM to provide its reasoning means that its final  
answer is no longer a blind guess. It's now supported by a series of logical  
steps. If the final answer is wrong, a human can easily look at the reasoning  
to pinpoint the exact step where the model went astray, making the error  
transparent and easier to debug.  



The example demonstrates the Chain-of-Thought (CoT) prompting technique using  
OpenRouter's Horizon model. It summarizes a math problem and shows how to use  
different prompting strategies. The problem involves basic arithmetic  
operations and the script provides three different approaches to solve it:  
 
1) Explicit CoT with step-by-step reasoning     
2) Concise final answer only  
3) Safe-CoT with brief rationale  
   
```python
import os
import sys
import textwrap

try:
    # Optional convenience for local development
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_env():
    """
    Load environment variables (if python-dotenv is installed) and validate key presence.
    Expected: OPENROUTER_API_KEY
    """
    if load_dotenv:
        load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "Missing OPENROUTER_API_KEY in environment.\n"
            "Create a .env with OPENROUTER_API_KEY=sk-or-... OR export it.",
            file=sys.stderr,
        )


def get_client():
    """
    Return an OpenAI-compatible client targeting OpenRouter.
    Requires: openai (v1+)
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(
            "The 'openai' package is required (pip install openai). Error: {}".format(e),
            file=sys.stderr,
        )
        raise

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def call_horizon(
    client,
    messages,
    temperature=0.2,
    max_tokens=512,
    seed=42,
):
    """
    Wrapper for OpenRouter chat.completions using the Horizon LLM.
    Model: openrouter/horizon-beta
    """
    completion = client.chat.completions.create(
        model="openrouter/horizon-beta",
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    return completion.choices[0].message.content or ""


def pretty_box(title, content):
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(textwrap.dedent(content).strip())
    print("=" * 80)


def build_problem():
    # Classic GSM-style problem
    return (
        "Taylor has 5 packs of markers. Each pack contains 12 markers. "
        "Taylor gives 7 markers to a friend and then buys 1 more pack. "
        "How many markers does Taylor have now?"
    )


def cot_messages(problem):
    """
    Explicit Chain-of-Thought prompting.
    Note: This can increase token usage. Consider concise variants for production.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a careful math tutor. Use step-by-step reasoning to solve problems accurately."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem: {problem}\n\n"
                "Think through the problem step by step, do arithmetic carefully, and explain your reasoning "
                "before giving the final answer.\n"
                "Format:\n"
                "Reasoning:\n"
                "Final Answer: <number>"
            ),
        },
    ]


def concise_messages(problem):
    """
    Ask only for the final numeric answer.
    Helpful when you want a short, low-cost response once you've validated the approach.
    """
    return [
        {
            "role": "system",
            "content": "You are a concise math solver. Provide only the final numeric answer.",
        },
        {
            "role": "user",
            "content": f"Problem: {problem}\nGive only: Final Answer: <number>",
        },
    ]


def safe_cot_messages(problem):
    """
    A constrained CoT that requests concise reasoning (3-5 steps) to limit verbosity/cost.
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Show a brief 3-5 step reasoning then the final answer.",
        },
        {
            "role": "user",
            "content": (
                f"Problem: {problem}\n\n"
                "Provide concise reasoning in at most 5 short steps, then the final answer.\n"
                "Format:\n"
                "Reasoning:\n"
                "Final Answer: <number>"
            ),
        },
    ]


def run_demo():
    """
    Demonstrates three prompting strategies with OpenRouter + Horizon:
      1) Explicit CoT (step-by-step reasoning)
      2) Concise-only (final numeric answer)
      3) Safe-CoT (brief 3-5 step rationale)
    """
    load_env()
    client = get_client()
    problem = build_problem()

    # 1) Explicit CoT
    cot_out = call_horizon(client, cot_messages(problem), temperature=0.2, max_tokens=600)
    pretty_box("Explicit CoT (step-by-step reasoning)", cot_out)

    # 2) Concise final answer only
    concise_out = call_horizon(client, concise_messages(problem), temperature=0.0, max_tokens=50)
    pretty_box("Concise only (final numeric answer)", concise_out)

    # 3) Safe-CoT with brief rationale
    safe_cot_out = call_horizon(client, safe_cot_messages(problem), temperature=0.2, max_tokens=200)
    pretty_box("Safe-CoT (brief rationale)", safe_cot_out)


if __name__ == "__main__":
    """
    Usage:
      1) pip install openai python-dotenv  (or manage deps in your preferred file)
      2) Put your key in .env: OPENROUTER_API_KEY=sk-or-...
      3) python example_cot_openrouter_horizon.py
    """
    run_demo()
```




== Prompting and shooting


In the context of AI, **prompting** is the act of providing a large language model (LLM) with   
instructions, questions, or context to guide it toward a specific response. A prompt is simply  
the text you input into the AI.

**Prompting is the primary way we communicate with and control AI models.** A well-crafted prompt  
can be the difference between a useless, generic response and a highly accurate, useful one.


=== What does "-Shot" or "Shooting" Mean?


The term **"-shot"** is a machine learning metaphor for **"an example"** or **"a demonstration"**.  
When you are "shooting," you are giving the model examples to learn from within the prompt itself.  
This is also known as **in-context learning**.

So, when you combine the two:

* **Zero-Shot:** You give the model **zero shots** (no examples) at learning the task. You are betting  
  that the model is smart enough to figure out what you want from the instruction alone.  
* **One-Shot:** You give the model **one shot** (a single example) to learn from. This gives it a clear  
  hint about the expected pattern or output format.  
* **Few-Shot:** You give the model **a few shots** (multiple examples) to learn from. This provides more  
  robust guidance, helping it understand more complex or nuanced tasks.

In summary, **prompting** is how you talk to the AI, and **"shooting"** refers to the number of examples  
you include in your prompt to show the AI what you mean.


```python
"""
Demonstration: Zero-shot vs One-shot vs Few-shot Prompting (OpenRouter-compatible OpenAI SDK)

Prerequisites:
- pip install openai
- Environment variable OPENROUTER_API_KEY must be set.

This script runs the same classification task (classify short support tickets into categories)
using three approaches:
1) Zero-shot: only task instruction, no examples
2) One-shot: one labeled example
3) Few-shot: three labeled examples

It prints the model outputs to show how examples shape behavior and consistency.
At the end, it prints a compact summary table comparing results across approaches.

Usage:
    python prompting_zero_one_few_shot.py --model z-ai/glm-4.5-air:free
"""

import argparse
import os
import sys
from textwrap import dedent

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# OpenRouter-compatible OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with: pip install openai")
    sys.exit(1)


DEFAULT_MODEL = "openrouter/horizon-beta"

CATEGORIES = ["billing", "technical", "account", "shipping", "other"]

TICKETS = [
    "My last invoice seems too high, can you check the charges?",
    "I can't log in after resetting my password.",
    "Where is my package? The tracking has not updated for 3 days.",
    "How do I change the email on my profile?",
    "The app crashes when I try to upload a file.",
    "I was billed twice for the same subscription this month.",
    "My profile picture keeps disappearing after I upload it.",
    "Can I get a refund for the duplicate charge?",
    "The login page shows an error after the latest update.",
    "I need to update my shipping address before the order ships.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot / One-shot / Few-shot prompting demo.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    return parser.parse_args()


def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def common_system():
    return dedent(f"""
        You are a helpful assistant that classifies short customer support tickets.
        Output ONLY a single category from this set: {CATEGORIES}.
        If uncertain, choose "other". Do not add extra words.
    """).strip()


def run_chat(client, model, messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def normalize_label(text):
    """
    Normalize model output to a single token label for table readability.
    - Lowercase, strip punctuation/quotes
    - Pick the first known category mentioned
    - Fallback to 'other' if none matched
    """
    import re
    raw = (text or "").strip().lower()
    raw = raw.replace('"', '').replace("'", "")
    # compress whitespace
    raw = re.sub(r"\s+", " ", raw)
    # try exact match first
    for c in CATEGORIES:
        if raw == c:
            return c
    # search first occurrence of any category token
    for c in CATEGORIES:
        if re.search(rf"\b{re.escape(c)}\b", raw):
            return c
    return "other"


def zero_shot(client, model, ticket):
    system = common_system()
    user = f"Ticket: {ticket}\nCategory:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return run_chat(client, model, messages)


def one_shot(client, model, ticket):
    system = common_system()
    example_input = "The invoice for last month has unexpected charges."
    example_output = "billing"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ticket: {example_input}\nCategory:"},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": f"Ticket: {ticket}\nCategory:"},
    ]
    return run_chat(client, model, messages)


def few_shot(client, model, ticket):
    system = common_system()

    examples = [
        ("I can't access my account after the update.", "technical"),
        ("Please change the email on my account to my work address.", "account"),
        ("The shipping status hasn't changed and the package is late.", "shipping"),
        ("I was charged twice for the same subscription this month.", "billing"),
        ("My profile picture keeps disappearing after I upload it.", "technical"),
    ]

    messages = [{"role": "system", "content": system}]

    for ex_in, ex_out in examples:
        messages.append({"role": "user", "content": f"Ticket: {ex_in}\nCategory:"})
        messages.append({"role": "assistant", "content": ex_out})

    messages.append({"role": "user", "content": f"Ticket: {ticket}\nCategory:"})

    return run_chat(client, model, messages)


def main():
    args = parse_args()
    client = get_client()

    print("Task: Classify support tickets into one of", CATEGORIES)
    print("\nTickets:")
    for i, t in enumerate(TICKETS, 1):
        print(f"{i}. {t}")

    # Collect raw and normalized results for summary
    zero_raw, one_raw, few_raw = [], [], []
    zero, one, few = [], [], []

    print("\n=== ZERO-SHOT RESULTS ===")
    for t in TICKETS:
        out = zero_shot(client, args.model, t)
        zero_raw.append(out)
        z = normalize_label(out)
        zero.append(z)
        print(f"- {t}\n  -> {out}")

    print("\n=== ONE-SHOT RESULTS ===")
    for t in TICKETS:
        out = one_shot(client, args.model, t)
        one_raw.append(out)
        o = normalize_label(out)
        one.append(o)
        print(f"- {t}\n  -> {out}")

    print("\n=== FEW-SHOT RESULTS ===")
    for t in TICKETS:
        out = few_shot(client, args.model, t)
        few_raw.append(out)
        f = normalize_label(out)
        few.append(f)
        print(f"- {t}\n  -> {out}")

    # Summary table (normalized labels for readability)
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Summary Table (normalized)")
        table.add_column("#", justify="right")
        table.add_column("Ticket", max_width=60)
        table.add_column("Zero-shot")
        table.add_column("One-shot")
        table.add_column("Few-shot")

        for i, t in enumerate(TICKETS, 1):
            z, o, f = zero[i-1], one[i-1], few[i-1]
            table.add_row(str(i), t, z, o, f)
        console.print(table)

        raw_table = Table(title="Raw Outputs (verbatim)")
        raw_table.add_column("#")
        raw_table.add_column("Zero-shot", max_width=40)
        raw_table.add_column("One-shot", max_width=40)
        raw_table.add_column("Few-shot", max_width=40)
        for i in range(len(TICKETS)):
            zr = zero_raw[i].replace("\n", " ")[:120]
            orr = one_raw[i].replace("\n", " ")[:120]
            fr = few_raw[i].replace("\n", " ")[:120]
            raw_table.add_row(str(i+1), zr, orr, fr)
        console.print(raw_table)
    else:
        print("\n=== SUMMARY TABLE (normalized) ===")
        print("| # | Ticket | Zero-shot | One-shot | Few-shot |")
        print("|---|--------|-----------|----------|----------|")
        for i, t in enumerate(TICKETS, 1):
            z, o, f = zero[i-1], one[i-1], few[i-1]
            short_t = t if len(t) <= 60 else t[:57] + "..."
            print(f"| {i} | {short_t} | {z} | {o} | {f} |")

        print("\n=== RAW OUTPUTS (verbatim) ===")
        print("| # | Zero-shot | One-shot | Few-shot |")
        print("|---|-----------|----------|----------|")
        for i in range(len(TICKETS)):
            zr = zero_raw[i].replace("\n", " ")[:120]
            orr = one_raw[i].replace("\n", " ")[:120]
            fr = few_raw[i].replace("\n", " ")[:120]
            print(f"| {i+1} | {zr} | {orr} | {fr} |")

    print("\nNotes:")
    if RICH_AVAILABLE:
        print("Tables rendered with Rich (https://rich.readthedocs.io)")
    else:
        print("Install Rich for prettier tables: pip install rich")
    print("- Normalized table shows clean category tokens for comparison.")
    print("- Raw outputs table helps students see how prompting affects verbosity.")
    print("- Few-shot typically yields the most consistent category usage.")


if __name__ == "__main__":
    main()
```


== Agents


An agent is a system that autonomously perceives its environment, makes decisions, and takes    
actions to achieve a specific goal—often by interacting with tools, APIs, or other systems.  

The difference between an agent and normal code that chats with a large language model (LLM)  
lies in autonomy, decision-making, and task orchestration. Let’s break it down:  





=== 🤖 Agent vs. Normal LLM Code


| Feature                      | **Agent**                                                                 | **Normal LLM Code**                                                  |
|-----------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------|
| **Autonomy**                | Can make decisions and take actions without constant user input           | Executes only what the user explicitly asks                          |
| **Memory / State**          | Maintains context across steps, sometimes with long-term memory           | Usually stateless or limited to short-term context                  |
| **Tool Use**                | Can call external tools, APIs, databases, or code to complete tasks       | May respond with code or suggestions, but doesn’t execute them      |
| **Goal-Oriented Behavior**  | Works toward a defined objective, often breaking it into subtasks         | Responds to prompts without a broader goal                          |
| **Planning & Reasoning**    | Plans steps, evaluates outcomes, and adjusts strategy                     | Responds reactively, without strategic planning                     |
| **Looping / Iteration**     | Can loop through tasks, retry failures, and refine outputs                | Typically one-shot responses unless manually prompted               |
| **Examples**                | AutoGPT, LangChain agents, OpenAI’s function-calling agents               | Basic chatbot, code assistant, or prompt-based interaction          |





This program summarizes web content from URLs using OpenAI's API.  
It fetches the content of each URL, extracts text, and generates a summary.  
The agent can process multiple URLs concurrently for efficiency.  

```python
import asyncio
import aiohttp
import os
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys

# Ensure compatibility with Windows event loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class URLSummarizerAgent:
    """Agent that summarizes web content from URLs using OpenAI"""
    
    def __init__(self, api_key, model="openrouter/horizon-beta"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        
    async def fetch_url_content(self, url: str) -> str:
        """Fetch and extract text content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        content = ' '.join(chunk for chunk in chunks if chunk)
                        
                        # Limit content to avoid token limits
                        return content[:8000]
                    else:
                        logger.error(f"Failed to fetch {url}: {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""
    
    async def summarize_text(self, text, max_length=150):
        """Summarize text using OpenAI"""
        if not text:
            return "No content to summarize"
            
        prompt = f"""Please provide a concise summary of the following text in {max_length} words or less. 
Focus on the main points and key insights:

{text}

Summary:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return "Error generating summary"
    
    async def summarize_url(self, url):
        """Summarize a single URL"""
        logger.info(f"Processing URL: {url}")
        
        content = await self.fetch_url_content(url)
        if not content:
            return {"url": url, "summary": "Failed to fetch content", "status": "error"}
        
        summary = await self.summarize_text(content)
        return {
            "url": url,
            "summary": summary,
            "status": "success",
            "content_length": len(content)
        }
    
    async def summarize_multiple_urls(self, urls):
        """Summarize multiple URLs concurrently"""
        logger.info(f"Starting to process {len(urls)} URLs")
        
        # Create independent tasks for each URL
        tasks = [self.summarize_url(url) for url in urls]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i],
                    "summary": f"Error: {str(result)}",
                    "status": "error"
                })
            else:
                processed_results.append(result)
        
        return processed_results

async def main():
    """Example usage"""
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    # Initialize agent
    agent = URLSummarizerAgent(api_key)
    
    # Example URLs to summarize
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning"
    ]
    
    # Process URLs concurrently
    results = await agent.summarize_multiple_urls(urls)
    
    # Display results
    for result in results:
        print(f"\n{'='*60}")
        print(f"URL: {result['url']}")
        print(f"Status: {result['status']}")
        print(f"Summary: {result['summary']}")
        if 'content_length' in result:
            print(f"Content Length: {result['content_length']} characters")

if __name__ == "__main__":
    asyncio.run(main())
```



#pagebreak()


= OpenAI examples 2



This example demonstrates how to use the OpenAI library for image analysis via OpenRouter, leveraging  
a vision-capable model like GPT-4o. It sends a user message with an image URL and a text prompt, then  
prints the model's descriptive response.  

Use a vision-enabled model, structure the message content as a list with `text` and `image_url` types.



== Image description


Passing image as external URL.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    extra_body={},
    model="openrouter/sonoma-sky-alpha",  # Vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": "https://pbs.twimg.com/media/CR9k1K1WwAAakqD.jpg"}}
            ]
        }
    ]
)
print(completion.choices[0].message.content)
```

Passing image in the URL.

```python
from openai import OpenAI
import os
import base64

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Read and encode image from disk
with open("image2.jpg", "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')

image_url = f"data:image/jpeg;base64,{base64_string}"

completion = client.chat.completions.create(
    extra_body={},
    model="openrouter/sonoma-sky-alpha",  # Vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
)
print(completion.choices[0].message.content)
```

A Data URI embeds file data directly into a string, so you don’t need to reference an external  
file. For images, it looks like this:

`data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...`

- data: → Indicates it's a data URI.
- image/jpeg → MIME type (could also be image/png, image/gif, etc.).
- base64 → Specifies that the data is base64-encoded.
- ... → The actual base64 string representing the image.



== Structured output


This example shows how to ask a model to return data in a strict JSON format  
and how to parse that output in Python. We provide the model with a text prompt  
asking it to extract information about people mentioned in the text (name, age, city).  

The `response_format` uses `json_schema` to enforce the expected structure;  
setting `"strict": true` helps ensure the model's response conforms to the schema.  
After receiving the response, the example parses the model output with `json.loads`  
and prints the extracted structured data. 


```python
from openai import OpenAI
import os
import json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

text = """
Extract information about people mentioned in the following text. For each
person, provide their name, age, and city of residence in a structured JSON
format. John Doe is a software engineer living in New York. He is 30 years old
and enjoys hiking and photography. Jane Smith is a graphic designer based in San
Francisco. She is 28 years old and loves painting and traveling."""

response = client.chat.completions.create(
    extra_body={},
    model="mistralai/mistral-small-3.2-24b-instruct:free",  # Model supporting structured outputs
    messages=[
        {
            "role": "user",
            "content": text
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "people_info",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "city": {"type": "string"}
                    },
                    "required": ["name", "age", "city"],
                    "additionalProperties": False
                }
            },
            "strict": True
        }
    }
)

# Parse the JSON response
info = json.loads(response.choices[0].message.content)
print("Extracted info:", info)
```

This example demonstrates using Pydantic models to define structured output for solving math equations step-by-step. It shows how to define nested models (`Step` and `MathResponse`) and use them to parse the model's response, ensuring type safety and structured data extraction without requiring `response_format` with JSON schema.

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class Step(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str


prompt = """
Solve the equation: 8x + 31 = 2.
Return your answer as a JSON object matching this format:

{
  "steps": [
    {"explanation": "...", "output": "..."},
    ...
  ],
  "final_answer": "..."
}
"""

response = client.chat.completions.create(
    model="openrouter/sonoma-dusk-alpha",
    messages=[{"role": "user", "content": prompt}],
)

raw_text = response.choices[0].message.content
parsed = MathResponse.model_validate_json(raw_text)

print(parsed)
print(parsed.final_answer)
```


== Sentiment analysis


```python
from openai import OpenAI

from pathlib import Path
import os
import time


client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

# movie_reviews = {
#     1: "The storyline was absolutely captivating, and the performances were brilliant. I couldn't look away for a second!",
#     2: "The pacing was excruciatingly slow, and the characters lacked depth. I was bored halfway through.",
#     3: "While the visuals were breathtaking, the plot felt predictable and uninspired.",
#     4: "This is a cinematic masterpiece that touched my heart. Every scene was perfection!",
#     5: "The dialogue was cringe-worthy, and the humor fell flat. Definitely not worth the hype.",
#     6: "It was an average film—not great, but not terrible either. I enjoyed some parts.",
#     7: "The chemistry between the leads was electric, and the soundtrack was phenomenal!",
#     8: "The movie started strong but completely fell apart in the second half. Such a disappointment.",
#     9: "A visually stunning film that combines action and emotion seamlessly. Highly recommend!",
#     10: "The premise was intriguing, but the execution left a lot to be desired. It just didn't click for me."
# }

slovak_movie_reviews = {
    1: "Príbeh bol úplne pútavý a herecké výkony brilantné. Nemohol som sa odtrhnúť ani na sekundu!",
    2: "Tempo bolo mimoriadne pomalé a postavy nemali žiadnu hĺbku. Nudil som sa už v polovici.",
    3: "Hoci vizuálne efekty boli ohromujúce, dej pôsobil predvídateľne a bez inšpirácie.",
    4: "Toto je filmové dielo, ktoré mi dojalo srdce. Každá scéna bola dokonalosť!",
    5: "Dialógy boli trápne a humor úplne zlyhal. Určite to nestojí za ten humbug.",
    6: "Bol to priemerný film – nie dobrý, ale ani úplná katastrofa. Niektoré časti ma bavili.",
    7: "Chemia medzi hlavnými postavami bola elektrizujúca a soundtrack fenomenálny!",
    8: "Film začal skvele, ale v druhej polovici sa úplne rozpadol. Veľké sklamanie.",
    9: "Vizualne ohromujúci film, ktorý dokonale spája akciu a emócie. Určite odporúčam!",
    10: "Premisa bola zaujímavá, ale realizácia bola slabá. Nedokázalo ma to zaujať."
}

for key, value in slovak_movie_reviews.items():

    # content = 'On a scale 0-1, figure out the sentiment of the the following movie review:'
    content = 'Na škále od 0 do 1, napíš sentiment nasledujúceho filmu:'
    content += value

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0.7,
        top_p=0.9,
        model='deepseek-chat',
        max_completion_tokens=1000
    )

    # print(chat_completion.choices[0].message.content)
    output = chat_completion.choices[0].message.content
    print(key, value, output)
```



== Classification


This section demonstrates a small Python script that classifies multiple short customer  
support tickets into predefined categories (billing, technical, account, shipping, other)  
using a model hosted via OpenRouter. The 

- Builds a concise system prompt describing the classification task and allowed categories.  
- Sends all sample tickets in a single chat request.
- Uses response_format with a json_schema (and strict: True) so the model returns a strict  
  JSON array of objects with "ticket" and "category" fields.
- Parses the model response with json.loads and prints a readable summary table using the rich library.  

Notes and tips:
- "strict": True encourages the model to follow the JSON schema exactly. If the model returns  
  non-JSON or deviates, add simple post-processing or retries (for example, try to extract  
  the first JSON block from the response).
- For production use, consider batching, rate limits, and error handling around API calls.  

```python
import argparse
import os
import sys
from textwrap import dedent

from rich.console import Console
from rich.table import Table

from openai import OpenAI


DEFAULT_MODEL = "openrouter/sonoma-sky-alpha"

CATEGORIES = ["billing", "technical", "account", "shipping", "other"]

TICKETS = [
    "My last invoice seems too high, can you check the charges?",
    "I can't log in after resetting my password.",
    "Where is my package? The tracking has not updated for 3 days.",
    "How do I change the email on my profile?",
    "The app crashes when I try to upload a file.",
    "I was billed twice for the same subscription this month.",
    "My profile picture keeps disappearing after I upload it.",
    "Can I get a refund for the duplicate charge?",
    "The login page shows an error after the latest update.",
    "I need to update my shipping address before the order ships.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ticket classification demo.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    return parser.parse_args()


def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def classify_tickets(client, model, tickets):
    system = dedent(f"""
       You are a helpful assistant that classifies short customer support tickets.
       Classify each ticket into one of these categories: {CATEGORIES}.
       Return the results as a JSON array of objects, each with "ticket" and "category" fields.
   """).strip()

    ticket_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tickets))
    user = f"Classify the following tickets:\n{ticket_list}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ticket_classifications",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticket": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["ticket", "category"]
                    }
                },
                "strict": True
            }
        }
    )

    import json
    return json.loads(response.choices[0].message.content)


def main():
    args = parse_args()
    client = get_client()

    print("Task: Classify support tickets into one of", CATEGORIES)
    print("\nTickets:")
    for i, t in enumerate(TICKETS, 1):
        print(f"{i}. {t}")

    # Classify all tickets in one request
    classifications = classify_tickets(client, args.model, TICKETS)

    # Summary table
    console = Console()
    table = Table(title="Classification Summary")
    table.add_column("#", justify="right")
    table.add_column("Ticket", max_width=60)
    table.add_column("Category")

    for i, item in enumerate(classifications, 1):
        table.add_row(str(i), item["ticket"], item["category"])
    console.print(table)


if __name__ == "__main__":
    main()
```


== Instructor


This example shows how to use the Instructor library with OpenAI to automatically  
parse responses into Pydantic models. It demonstrates classifying a customer support  
query and generating a response while extracting structured data (content and category) from the model's output.

```python
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
import os


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

client = instructor.from_openai(client)
MODEL = "openrouter/sonoma-dusk-alpha"

class Reply(BaseModel):
    content: str = Field(description="Your reply that we send to the customer.")
    category: str = Field(
        description="Category of the ticket: 'general', 'order', 'billing'"
    )

query = "How do I reset my password in FreeBSD?"

reply = client.chat.completions.create(
    model=MODEL,
    response_model=Reply,
    messages=[
        {
            "role": "system",
            "content": "You're a helpful customer care assistant that can classify incoming messages and create a response.",
        },
        {"role": "user", "content": query},
    ],
)

print(reply.content)
print(reply.category)
```

This example builds upon the previous one by using Python enums to enforce strict categorization. It demonstrates how to define predefined categories as an enum (`TicketCategory`) and ensure the model's response conforms to only those specific values, providing better type safety and validation.

```python
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
import os


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

client = instructor.from_openai(client)
MODEL = "openrouter/sonoma-dusk-alpha"


class TicketCategory(str, Enum):
    GENERAL = "general"
    ORDER = "order"
    BILLING = "billing"
    OTHER = "other"


class Reply(BaseModel):
    content: str = Field(
        description="Your reply that we send to the customer.")
    category: TicketCategory = Field(
        description="Correctly assign one of the predefined categories"
    )


system_prompt = "You're a helpful customer care assistant that can classify incoming messages and create a response."
query = "I placed an order last week but haven't received any confirmation email. Can you check the status for me?"

reply = client.chat.completions.create(
    model=MODEL,
    response_model=Reply,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": query},
    ],
)

print(reply.content)
print(reply.category)
```


== Data extraction


This example demonstrates extracting structured information from natural language  
text using Pydantic models. It shows how to prompt the model to extract event details  
(name, date, participants) from unstructured text and parse the response into a typed Python object for further processing.

```python
from openai import OpenAI
from pydantic import BaseModel
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# Prompt the model to return a JSON object matching the schema
messages = [
    {"role": "system", "content": "Extract the event information and return it as a JSON object with keys: name, date, participants."},
    {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
]

response = client.chat.completions.create(
    model="openrouter/sonoma-dusk-alpha",
    messages=messages,
    temperature=0,
)

# Parse the model's response using Pydantic
raw_text = response.choices[0].message.content.strip()
print(raw_text)

# Optional: clean up if the model wraps JSON in markdown
# if raw_text.startswith("```json"):
#     raw_text = raw_text.strip("```json").strip("```")

event = CalendarEvent.model_validate_json(raw_text)

print(event)
```


== Nested Pydantic models


This example demonstrates using nested Pydantic models with Instructor to extract complex,  
hierarchical data structures. It shows how to define models with nested objects (`Details` within `Reply`)  
to capture multiple levels of information, such as customer support responses with priority and urgency assessment.

```python
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
import os

# nested Pydantic models example

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

client = instructor.from_openai(client)
MODEL = "openrouter/sonoma-dusk-alpha"

class TicketCategory(str, Enum):
    GENERAL = "general"
    ORDER = "order"
    BILLING = "billing"
    OTHER = "other"

class Details(BaseModel):
    priority: str = Field(description="Priority level: 'low', 'medium', 'high'")
    urgency: str = Field(description="Urgency: 'low', 'medium', 'high'")

class Reply(BaseModel):
    content: str = Field(description="Your reply that we send to the customer.")
    category: TicketCategory = Field(description="Correctly assign one of the predefined categories")
    details: Details = Field(description="Additional details about the ticket")

system_prompt = "You're a helpful customer care assistant that can classify incoming messages, create a response, and assess priority and urgency."
query = "My order is delayed and I need it urgently for an event tomorrow. Please expedite it!"

reply = client.chat.completions.create(
    model=MODEL,
    response_model=Reply,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ],
)

print(reply.content)
print(reply.category)
print(reply.details.priority)
print(reply.details.urgency)
```


== Extract list of keywords


This example shows how to extract lists of structured data from text using Instructor.  
It demonstrates extracting key terms and concepts from customer messages while also generating  
a response, useful for tagging, categorization, and content analysis workflows.

```python
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

client = instructor.from_openai(client)
MODEL = "openrouter/sonoma-dusk-alpha"

class Reply(BaseModel):
    content: str = Field(description="Your reply that we send to the customer.")
    keywords: list[str] = Field(description="List of key terms extracted from the message")

system_prompt = "You're a helpful assistant that can extract key information from customer messages."
query = "I'm having trouble with my recent purchase. The product arrived damaged, and I also have questions about the return policy and warranty."

reply = client.chat.completions.create(
    model=MODEL,
    response_model=Reply,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ],
)

print(reply.content)
print(reply.keywords)
```


== Pydantic validation


This example demonstrates using Pydantic field validators to enforce custom validation  
rules on model responses. It shows how to implement field-level validation (minimum length, custom value validation)  
to ensure the model's output meets specific business requirements and data quality standards.

```python
import instructor
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

client = instructor.from_openai(client)
MODEL = "openrouter/sonoma-dusk-alpha"

class Reply(BaseModel):
    content: str = Field(description="Your reply that we send to the customer.", min_length=10)
    sentiment: str = Field(description="Overall sentiment: 'positive', 'neutral', 'negative'")

    @field_validator('sentiment')
    @classmethod
    def validate_sentiment(cls, v):
        if v.lower() not in ['positive', 'neutral', 'negative']:
            raise ValueError('Sentiment must be positive, neutral, or negative')
        return v.lower()

system_prompt = "You're a helpful customer care assistant that analyzes sentiment and creates responses."
query = "Thank you for the excellent service! My issue was resolved quickly."

reply = client.chat.completions.create(
    model=MODEL,
    response_model=Reply,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ],
)

print(reply.content)
print(reply.sentiment)
```



== Audio transcription


This example demonstrates how to transcribe a local audio file to text using the
OpenAI Python client. It reads the audio file in binary, sends it to the audio
transcription endpoint with model `whisper-large-v3` and an optional prompt to
steer the output, then returns the transcription via `transcription.text`. The
snippet shows configuring the client with an API key (`GROQ_API_KEY`) and a  
custom `base_url`, and can be adapted by changing the model, file path, or
prompt.  

```python
import openai
import os

API_KEY = os.getenv("GROQ_API_KEY")
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

def transcribe_file(path):
    with open(path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            prompt="Give us only the text of the fable."
        )
    return transcription.text

result = transcribe_file("aesop_fox_grapes.mp3")
print(result)
```

The `transcribe_file` function can be reused to transcribe any local audio file
by providing its path. The parameters are model (choose from available Whisper
models), file (the binary audio file), and an optional prompt to guide the
transcription.


== System prompts


```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def demonstrate_system_prompts():
    """Showcase different types of system prompts and their effects."""

    user_question = "How do I optimize my Python code for better performance?"

    # Educational and patient teacher
    teacher_prompt = """
    You are a patient and encouraging computer science professor.
    Your teaching style:
    - Break down complex topics into digestible parts
    - Use analogies and real-world examples
    - Encourage questions and curiosity
    - Build understanding step by step
    - Celebrate learning milestones
    - Be supportive of all skill levels
    """

    # Concise and direct senior engineer
    engineer_prompt = """
    You are a senior software engineer with 15 years of experience.
    Your communication style:
    - Be direct and to the point
    - Focus on practical solutions
    - Use technical terminology appropriately
    - Provide actionable advice
    - Prioritize performance and efficiency
    - Don't waste words on unnecessary explanations
    """

    # Humorous and engaging tech enthusiast
    enthusiast_prompt = """
    You are a passionate tech enthusiast who loves programming.
    Your personality:
    - Use humor and relatable analogies
    - Be enthusiastic and energetic
    - Share personal anecdotes from coding experience
    - Make complex topics fun and accessible
    - Use emojis occasionally to add personality
    - Encourage experimentation and learning through doing
    """

    # Academic and rigorous researcher
    researcher_prompt = """
    You are a computer science researcher and professor.
    Your approach:
    - Provide thorough, academically rigorous explanations
    - Cite relevant algorithms and data structures
    - Discuss time/space complexity analysis
    - Reference academic papers and best practices
    - Encourage critical thinking about trade-offs
    - Focus on fundamental computer science principles
    """

    prompts = [
        ("Teacher", teacher_prompt),
        ("Engineer", engineer_prompt),
        ("Enthusiast", enthusiast_prompt),
        ("Researcher", researcher_prompt)
    ]

    for persona_name, prompt in prompts:
        print(f"\n=== {persona_name} ===")

        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ]
        )

        print(response.choices[0].message.content)

if __name__ == "__main__":
    demonstrate_system_prompts()
```
            
                
                
            
        






#pagebreak()


= Conclusion

Congratulations! You've completed this comprehensive guide to OpenAI and Python. You now have the knowledge and skills to build sophisticated AI-powered applications.

== What You've Learned

Throughout this book, you've gained expertise in:

- Understanding artificial intelligence and large language models
- Setting up Python for AI development
- Working with the OpenAI Python library
- Creating conversational AI applications
- Implementing streaming responses
- Working with images and vision models
- Using function calling and tools
- Mastering prompt engineering
- Building practical AI applications

== Next Steps

Continue your AI journey by:

- Building projects that interest you
- Exploring advanced topics like fine-tuning and RAG
- Joining AI communities and forums
- Staying current with AI research and developments
- Experimenting with new models and techniques

== Resources

- OpenAI Documentation: platform.openai.com
- GitHub Repository: github.com/janbodnar
- AI Communities: Reddit r/OpenAI, Discord servers
- Research: arXiv.org, AI conference papers

Keep learning, keep building, and enjoy the journey into artificial intelligence!

#pagebreak()

= Appendix: Quick Reference

== Essential Code Patterns

*Basic Chat:*
```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="anthropic/claude-3-haiku",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

*Streaming:*
```python
stream = client.chat.completions.create(
    model="anthropic/claude-3-haiku",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

*Error Handling:*
```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"Error: {e}")
```

== Common Parameters

- `model`: Which AI model to use
- `messages`: List of conversation messages
- `temperature`: Creativity (0.0-2.0)
- `max_tokens`: Maximum response length
- `stream`: Enable streaming responses
- `top_p`: Alternative to temperature
- `presence_penalty`: Topic repetition control
- `frequency_penalty`: Word repetition control

== Troubleshooting

*Authentication Error:* Check API key is correct and loaded
*Rate Limit:* Implement retry with backoff
*Token Limit:* Reduce max_tokens or message length
*Network Timeout:* Check connection, increase timeout

== Best Practices

- Always use environment variables for API keys
- Implement proper error handling
- Monitor token usage for cost control
- Use streaming for better UX
- Test with different models
- Keep system prompts clear and specific
- Validate and sanitize user inputs
- Log interactions for debugging



= Exercises and Practice Problems

This chapter provides hands-on exercises to reinforce your learning. Work through these problems to build practical skills.

== Chapter Exercises

=== Exercise 1: Build a Simple Chatbot

Create a command-line chatbot that maintains conversation history.

Requirements:
- Accept user input in a loop
- Maintain conversation context
- Handle quit command
- Save conversation to file

=== Exercise 2: Token Counter

Build a utility that estimates token costs before making API calls.

Requirements:
- Calculate approximate tokens
- Estimate costs for different models
- Compare model pricing
- Suggest most cost-effective option

=== Exercise 3: Prompt Template System

Create a system for managing reusable prompt templates.

Requirements:
- Load templates from files
- Support variable substitution
- Validate required variables
- Test with multiple models

== Key Takeaways

- Practice builds proficiency
- Start with simple projects
- Gradually increase complexity
- Learn from errors
- Share your work with the community

#pagebreak()

= Final Summary

You've completed a comprehensive journey through OpenAI and Python! You're now equipped to build sophisticated AI applications.

== Core Concepts Mastered

- Artificial intelligence and large language models
- Python programming for AI development
- OpenAI library and compatible APIs
- Chat completions and streaming
- Working with images and vision models
- Function calling and tools
- Prompt engineering mastery
- System prompts and personas
- Cost optimization strategies
- Security and best practices
- Error handling and monitoring
- Testing and deployment

== Next Steps in Your AI Journey

Continue learning and building. The AI field evolves rapidly - stay curious, experiment often, and contribute to the community.

Best of luck with your AI projects!



= Advanced Prompt Engineering Techniques

Prompt engineering is the art and science of crafting effective instructions for AI models. This chapter covers advanced techniques for getting better results.

== Zero-Shot vs Few-Shot Prompting

Zero-shot prompting asks the model to perform tasks without examples:

```python
prompt = "Classify this email as spam or not spam: {email_text}"
```

Few-shot prompting provides examples:

```python
prompt = """
Classify emails as spam or not spam.

Email: "Win a free iPhone now!"
Classification: spam

Email: "Meeting at 3pm tomorrow"
Classification: not spam

Email: "{email_text}"
Classification:"""
```

Few-shot generally produces better, more consistent results.

== Chain-of-Thought Prompting

Guide the model to show its reasoning:

```python
prompt = """
Solve this problem step by step:

Problem: If apples cost $0.50 each and I buy 12 apples with a $20 bill, how much change do I get?

Step 1: Calculate total cost
Step 2: Subtract from amount paid
Step 3: State the answer

Let's solve it:"""
```

This technique improves accuracy on complex reasoning tasks.

== Role-Based Prompting

Assign specific roles to guide behavior:

```python
system_prompts = {
    "teacher": "You are a patient teacher explaining to a 10-year-old.",
    "expert": "You are a PhD expert in this field. Be technical and precise.",
    "critic": "You are a critical reviewer. Point out flaws and weaknesses.",
    "creative": "You are a creative writer. Be imaginative and original."
}
```

== Prompt Templates

Create reusable templates:

```python
class PromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

# Usage
summary_template = PromptTemplate("""
Summarize the following {document_type} in {length} words:

{content}

Focus on: {focus_areas}
""")

prompt = summary_template.format(
    document_type="research paper",
    length="100",
    content=paper_text,
    focus_areas="methodology and findings"
)
```

== Prompt Optimization

Iteratively improve prompts:

```python
def test_prompt_variations(base_prompt, variations, test_cases):
    """Test different prompt variations."""
    results = {}
    
    for var_name, variation in variations.items():
        prompt = base_prompt + "\n" + variation
        scores = []
        
        for test_case in test_cases:
            response = get_ai_response(prompt, test_case)
            score = evaluate_response(response, test_case['expected'])
            scores.append(score)
        
        results[var_name] = {
            'average_score': sum(scores) / len(scores),
            'scores': scores
        }
    
    return results
```

== Key Takeaways

- Use few-shot prompting for better consistency
- Apply chain-of-thought for complex reasoning
- Assign roles to guide model behavior
- Create reusable prompt templates
- Test and optimize prompts iteratively
- Be specific and clear in instructions
- Provide relevant context
- Specify desired output format

#pagebreak()

= Working with Embeddings

Embeddings are vector representations of text that enable semantic search and similarity comparison.

== Understanding Embeddings

Embeddings convert text into numerical vectors:

```python
from openai import OpenAI
import os

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for text."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Example
text = "The quick brown fox jumps over the lazy dog"
embedding = get_embedding(text)
print(f"Embedding dimensions: {len(embedding)}")
print(f"First few values: {embedding[:5]}")
```

== Semantic Search

Find similar documents using embeddings:

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class SemanticSearch:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text):
        """Add document to search index."""
        embedding = get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def search(self, query, top_k=5):
        """Search for similar documents."""
        query_embedding = get_embedding(query)
        
        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {
                'document': self.documents[i],
                'similarity': similarities[i]
            }
            for i in top_indices
        ]
        
        return results

# Usage
search = SemanticSearch()
search.add_document("Python is a programming language")
search.add_document("Machine learning uses algorithms")
search.add_document("Dogs are loyal pets")

results = search.search("What is Python?")
for r in results:
    print(f"{r['similarity']:.3f}: {r['document']}")
```

== Clustering Documents

Group similar documents:

```python
from sklearn.cluster import KMeans

def cluster_documents(documents, n_clusters=3):
    """Cluster documents by similarity."""
    # Get embeddings
    embeddings = [get_embedding(doc) for doc in documents]
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Group by cluster
    clusters = {}
    for doc, label in zip(documents, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(doc)
    
    return clusters

# Usage
documents = [
    "Python programming tutorial",
    "Machine learning basics",
    "Python data science",
    "Deep learning with PyTorch",
    "JavaScript web development"
]

clusters = cluster_documents(documents)
for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

== Recommendation System

Build a simple recommendation engine:

```python
class RecommendationEngine:
    def __init__(self):
        self.items = []
        self.embeddings = []
        self.user_preferences = {}
    
    def add_item(self, item_id, description):
        """Add item to catalog."""
        self.items.append({'id': item_id, 'description': description})
        self.embeddings.append(get_embedding(description))
    
    def record_preference(self, user_id, item_id, liked=True):
        """Record user preference."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'liked': [], 'disliked': []}
        
        if liked:
            self.user_preferences[user_id]['liked'].append(item_id)
        else:
            self.user_preferences[user_id]['disliked'].append(item_id)
    
    def recommend(self, user_id, n=5):
        """Generate recommendations for user."""
        if user_id not in self.user_preferences:
            return []
        
        prefs = self.user_preferences[user_id]
        
        # Calculate user profile from liked items
        liked_embeddings = [
            self.embeddings[i] for i, item in enumerate(self.items)
            if item['id'] in prefs['liked']
        ]
        
        if not liked_embeddings:
            return []
        
        # Average of liked item embeddings
        user_profile = np.mean(liked_embeddings, axis=0)
        
        # Find similar items not yet interacted with
        seen_ids = set(prefs['liked'] + prefs['disliked'])
        
        recommendations = []
        for i, item in enumerate(self.items):
            if item['id'] not in seen_ids:
                similarity = cosine_similarity(user_profile, self.embeddings[i])
                recommendations.append({
                    'item': item,
                    'score': similarity
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n]

# Usage
engine = RecommendationEngine()
engine.add_item(1, "Introduction to Python programming")
engine.add_item(2, "Advanced machine learning techniques")
engine.add_item(3, "Web development with JavaScript")

engine.record_preference("user1", 1, liked=True)
recommendations = engine.recommend("user1")
for rec in recommendations:
    print(f"{rec['score']:.3f}: {rec['item']['description']}")
```

== Key Takeaways

- Embeddings represent text as numerical vectors
- Cosine similarity measures semantic similarity
- Semantic search finds relevant documents
- Clustering groups similar content
- Recommendation systems leverage user preferences
- Different embedding models offer different trade-offs
- Store embeddings for fast retrieval
- Update embeddings when content changes

#pagebreak()

= Building Production-Ready Applications

This chapter covers essential practices for deploying AI applications in production environments.

== Application Architecture

Structure your application for scalability:

```python
# app/models.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatResponse:
    message: str
    tokens_used: int
    model: str
    finish_reason: str

# app/ai_client.py
class AIClient:
    """Centralized AI client with configuration."""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
    
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None
    ) -> ChatResponse:
        """Send chat request."""
        model = model or self.config.default_model
        
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages
        )
        
        return ChatResponse(
            message=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=model,
            finish_reason=response.choices[0].finish_reason
        )

# app/services.py
class ChatService:
    """Business logic for chat functionality."""
    
    def __init__(self, ai_client, cache, monitor):
        self.ai_client = ai_client
        self.cache = cache
        self.monitor = monitor
    
    def process_message(self, user_id, message):
        """Process user message with caching and monitoring."""
        # Check cache
        cached = self.cache.get(user_id, message)
        if cached:
            return cached
        
        # Get AI response
        messages = self._build_messages(user_id, message)
        response = self.ai_client.chat(messages)
        
        # Monitor
        self.monitor.record_call(
            user_id=user_id,
            tokens=response.tokens_used,
            model=response.model
        )
        
        # Cache result
        self.cache.set(user_id, message, response.message)
        
        return response.message
```

== API Design

Create a clean API for your AI service:

```python
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/v1/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat():
    """Chat endpoint."""
    try:
        data = request.json
        
        # Validate input
        if 'message' not in data:
            return jsonify({'error': 'message required'}), 400
        
        if len(data['message']) > 1000:
            return jsonify({'error': 'message too long'}), 400
        
        # Process message
        response = chat_service.process_message(
            user_id=request.headers.get('X-User-ID'),
            message=data['message']
        )
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'internal error'}), 500

@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check endpoint."""
    health_status = health_check()
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

== Asynchronous Processing

Handle high loads with async:

```python
import asyncio
from openai import AsyncOpenAI

class AsyncAIService:
    """Asynchronous AI service."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    async def process_batch(self, messages_list):
        """Process multiple messages concurrently."""
        tasks = [
            self._process_single(messages)
            for messages in messages_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append({
                    'index': i,
                    'error': str(result),
                    'status': 'failed'
                })
            else:
                processed.append({
                    'index': i,
                    'response': result,
                    'status': 'success'
                })
        
        return processed
    
    async def _process_single(self, messages):
        """Process a single message."""
        response = await self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages
        )
        return response.choices[0].message.content

# Usage
async def main():
    service = AsyncAIService()
    
    batch = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "What's the weather?"}],
        [{"role": "user", "content": "Tell me a joke"}]
    ]
    
    results = await service.process_batch(batch)
    
    for result in results:
        print(f"{result['index']}: {result.get('response', result.get('error'))}")

# Run
asyncio.run(main())
```

== Database Integration

Store conversations in a database:

```python
import sqlite3
from datetime import datetime

class ConversationDB:
    """Database for storing conversations."""
    
    def __init__(self, db_path="conversations.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens_used INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        self.conn.commit()
    
    def create_conversation(self, user_id):
        """Create a new conversation."""
        cursor = self.conn.execute(
            "INSERT INTO conversations (user_id) VALUES (?)",
            (user_id,)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def add_message(self, conversation_id, role, content, tokens=None):
        """Add a message to conversation."""
        self.conn.execute(
            """INSERT INTO messages 
            (conversation_id, role, content, tokens_used)
            VALUES (?, ?, ?, ?)""",
            (conversation_id, role, content, tokens)
        )
        self.conn.commit()
    
    def get_conversation(self, conversation_id):
        """Get all messages in a conversation."""
        cursor = self.conn.execute(
            """SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at""",
            (conversation_id,)
        )
        
        return [
            {
                'role': row[0],
                'content': row[1],
                'timestamp': row[2]
            }
            for row in cursor.fetchall()
        ]

# Usage
db = ConversationDB()
conv_id = db.create_conversation("user123")
db.add_message(conv_id, "user", "Hello!", tokens=5)
db.add_message(conv_id, "assistant", "Hi there!", tokens=8)

messages = db.get_conversation(conv_id)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")
```

== Key Takeaways

- Structure code into clear layers (models, services, API)
- Design RESTful APIs with proper validation
- Use async processing for better performance
- Implement rate limiting to prevent abuse
- Store conversations in a database
- Monitor all API calls
- Handle errors gracefully
- Document your API thoroughly

#pagebreak()


= Comprehensive Reference Guide

This chapter provides quick reference material for common tasks and patterns.

== Common Code Patterns

=== Basic Chat Completion

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="anthropic/claude-3-haiku",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

=== Streaming Response

```python
stream = client.chat.completions.create(
    model="anthropic/claude-3-haiku",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # New line at end
```

=== Multi-turn Conversation

```python
messages = []

def chat(user_message):
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=messages
    )
    
    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message

# Usage
print(chat("Hello!"))
print(chat("How are you?"))
print(chat("What's the weather?"))
```

=== Error Handling

```python
import time

def safe_api_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=messages
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise
```

=== Image Analysis

```python
response = client.chat.completions.create(
    model="openrouter/sonoma-sky-alpha",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

=== Function Calling

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Execute function
    result = get_weather(arguments["location"])
    
    # Send result back
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })
```

== Configuration Examples

=== Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
OPENROUTER_API_KEY=sk-or-your-key-here
DEEPSEEK_API_KEY=sk-your-key-here
GROQ_API_KEY=gsk_your-key-here
```

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Endpoints
    OPENROUTER_URL = "https://openrouter.ai/api/v1"
    DEEPSEEK_URL = "https://api.deepseek.com"
    
    # Defaults
    DEFAULT_MODEL = "anthropic/claude-3-haiku"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
```

=== Project Structure

```
project/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── config.py
├── src/
│   ├── __init__.py
│   ├── client.py
│   ├── prompts.py
│   ├── utils.py
│   └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   └── test_prompts.py
└── examples/
    ├── basic_chat.py
    ├── streaming.py
    └── function_calling.py
```

== Best Practices Checklist

=== Security
- [ ] API keys in environment variables
- [ ] Never commit .env files
- [ ] Validate all user inputs
- [ ] Implement rate limiting
- [ ] Use HTTPS for all requests
- [ ] Log security events
- [ ] Sanitize outputs
- [ ] Handle errors gracefully

=== Performance
- [ ] Use appropriate models for tasks
- [ ] Implement caching where possible
- [ ] Monitor token usage
- [ ] Use async for concurrent requests
- [ ] Batch requests when possible
- [ ] Set appropriate timeouts
- [ ] Optimize prompt lengths
- [ ] Stream long responses

=== Development
- [ ] Use version control (git)
- [ ] Write comprehensive tests
- [ ] Document your code
- [ ] Follow PEP 8 style guide
- [ ] Use type hints
- [ ] Handle exceptions properly
- [ ] Log important events
- [ ] Use virtual environments

=== Production
- [ ] Implement health checks
- [ ] Set up monitoring
- [ ] Configure logging
- [ ] Use circuit breakers
- [ ] Implement retry logic
- [ ] Have rollback plan
- [ ] Document deployment process
- [ ] Test in staging first

== Troubleshooting Guide

=== Common Errors and Solutions

*Authentication Error*

Problem: "Incorrect API key provided"

Solutions:
- Verify API key is correct
- Check key is properly loaded from environment
- Ensure key is active on provider's website
- Try regenerating the key

*Rate Limit Error*

Problem: "Rate limit exceeded"

Solutions:
- Implement exponential backoff
- Reduce request frequency
- Upgrade API plan
- Use different model tiers

*Timeout Error*

Problem: "Request timeout"

Solutions:
- Increase timeout setting
- Check internet connection
- Try again during off-peak hours
- Use faster models

*Token Limit Exceeded*

Problem: "Maximum context length exceeded"

Solutions:
- Reduce input message length
- Decrease max_tokens parameter
- Summarize conversation history
- Split into multiple requests

*Invalid Model*

Problem: "Model not found"

Solutions:
- Check model name spelling
- Verify model is available
- Check provider documentation
- Try alternative model

=== Debugging Tips

1. Enable verbose logging
2. Print request/response data
3. Test with minimal examples
4. Check API status pages
5. Verify environment setup
6. Test network connectivity
7. Review error messages carefully
8. Check provider documentation

== Additional Resources

=== Official Documentation
- OpenAI Platform: platform.openai.com/docs
- Anthropic Claude: docs.anthropic.com
- DeepSeek: platform.deepseek.com
- OpenRouter: openrouter.ai/docs

=== API References
- OpenAI API Reference: platform.openai.com/docs/api-reference
- Python Library: github.com/openai/openai-python

=== Community Resources
- Reddit: r/OpenAI, r/MachineLearning
- Discord: Various AI community servers
- GitHub: Search for "openai" and "llm"
- Stack Overflow: Tag "openai-api"

=== Learning Resources
- DeepLearning.AI courses
- Fast.ai practical courses
- University ML courses online
- YouTube tutorials and talks
- Technical blog posts
- Research papers on arXiv

=== Tools and Libraries
- LangChain: Framework for LLM applications
- Gradio: Web UI for ML models
- Streamlit: Data apps framework
- Instructor: Structured outputs
- Tiktoken: Token counting

== Conclusion

Congratulations on completing this comprehensive guide to OpenAI and Python! You've learned:

- Fundamentals of AI and large language models
- Python programming for AI development
- OpenAI library and multiple providers
- Chat completions and streaming
- Vision models and image processing
- Function calling and tools
- Prompt engineering mastery
- System prompts and personas
- Embeddings and semantic search
- Production deployment practices
- Testing and monitoring
- Security and best practices

== Your Next Steps

1. *Build Projects* - Start with simple applications and gradually increase complexity

2. *Join Communities* - Engage with other developers, share your work, ask questions

3. *Stay Updated* - Follow AI research, try new models, learn new techniques

4. *Contribute* - Open source your projects, help others, write tutorials

5. *Keep Learning* - AI evolves rapidly, continuous learning is essential

== Final Thoughts

The field of AI is transforming how we interact with technology and solve problems. You now have the foundation to build intelligent applications that can understand language, analyze images, and assist users in countless ways.

Remember that AI is a tool to amplify human creativity and capability. Use it ethically, responsibly, and for the benefit of users. Build applications that solve real problems and add genuine value.

Start small, experiment often, and don't be afraid to fail. Every error teaches a lesson, every project builds experience. The AI community is collaborative and welcoming - participate, share, and learn together.

The future of AI is being written now, and you're part of it. Build something amazing!

Happy coding, and may your AI applications bring value to the world!

#v(2em)

*Jan Bodnar*

#datetime.today().display("[month repr:long] [day], [year]")

#pagebreak()

= Index of Key Terms

- *API (Application Programming Interface)*: Interface for interacting with AI services

- *Asynchronous Programming*: Executing operations concurrently for better performance

- *Chain-of-Thought*: Prompting technique that shows reasoning steps

- *Chat Completions*: Conversational AI interactions with context

- *Circuit Breaker*: Pattern to prevent cascading failures

- *Embeddings*: Vector representations of text for similarity comparison

- *Few-Shot Learning*: Providing examples to guide model behavior

- *Function Calling*: Allowing models to execute predefined functions

- *Large Language Model (LLM)*: AI model trained on vast text data

- *Prompt Engineering*: Crafting effective instructions for AI models

- *Rate Limiting*: Controlling request frequency to prevent abuse

- *Streaming*: Receiving AI responses incrementally

- *System Prompt*: Instructions that set model behavior and context

- *Temperature*: Parameter controlling response randomness

- *Tokens*: Units used to measure API usage and costs

- *Vision Models*: AI models that can analyze images

- *Zero-Shot Learning*: Performing tasks without examples

#pagebreak()



= Appendix B: Extended Examples

This appendix provides complete, working examples for common use cases.

== Example 1: Customer Support Bot

A complete customer support chatbot with conversation management:

```python
import os
from openai import OpenAI
from datetime import datetime

class SupportBot:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.conversations = {}
        self.system_prompt = """You are a helpful customer support agent.
Be polite, professional, and empathetic.
Help customers with their issues efficiently.
If you cannot help, direct them to human support."""
    
    def start_conversation(self, customer_id):
        self.conversations[customer_id] = [{
            "role": "system",
            "content": self.system_prompt
        }]
        return "Hello! How can I help you today?"
    
    def send_message(self, customer_id, message):
        if customer_id not in self.conversations:
            self.start_conversation(customer_id)
        
        # Add user message
        self.conversations[customer_id].append({
            "role": "user",
            "content": message
        })
        
        # Get AI response
        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=self.conversations[customer_id],
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        
        # Add assistant response
        self.conversations[customer_id].append({
            "role": "assistant",
            "content": reply
        })
        
        return reply
    
    def save_conversation(self, customer_id, filename):
        if customer_id in self.conversations:
            with open(filename, 'w') as f:
                for msg in self.conversations[customer_id]:
                    if msg['role'] != 'system':
                        f.write(f"{msg['role']}: {msg['content']}\n\n")

# Usage
bot = SupportBot()
customer = "customer123"

print(bot.start_conversation(customer))
print(bot.send_message(customer, "I need help with my order"))
print(bot.send_message(customer, "My tracking number is ABC123"))

bot.save_conversation(customer, f"conversation_{customer}.txt")
```

== Example 2: Content Moderation System

Implement content moderation:

```python
class ContentModerator:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def moderate_content(self, text):
        prompt = f"""Analyze this content for:
1. Inappropriate language
2. Hate speech
3. Violence
4. Sensitive topics
5. Spam

Content: {text}

Provide a JSON response with:
- is_safe (boolean)
- issues (list of strings)
- severity (low/medium/high)
- recommendation (approve/review/reject)"""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        import json
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            return {
                "is_safe": False,
                "issues": ["Failed to parse"],
                "severity": "high",
                "recommendation": "review"
            }
    
    def batch_moderate(self, contents):
        results = []
        for content in contents:
            result = self.moderate_content(content)
            results.append({
                "content": content[:50] + "...",
                "moderation": result
            })
        return results

# Usage
moderator = ContentModerator()

test_contents = [
    "This is a normal message",
    "Check out this amazing product!",
    "I disagree with your opinion"
]

results = moderator.batch_moderate(test_contents)
for r in results:
    print(f"Content: {r['content']}")
    print(f"Safe: {r['moderation']['is_safe']}")
    print(f"Recommendation: {r['moderation']['recommendation']}")
    print()
```

== Example 3: Document Summarizer

Summarize documents with customizable length:

```python
class DocumentSummarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def summarize(self, text, length="medium", style="neutral"):
        length_guides = {
            "short": "in 2-3 sentences",
            "medium": "in 1-2 paragraphs",
            "long": "in 3-4 paragraphs"
        }
        
        style_guides = {
            "neutral": "in a neutral tone",
            "technical": "in a technical style",
            "simple": "in simple, easy-to-understand language",
            "formal": "in a formal, academic style"
        }
        
        prompt = f"""Summarize the following text {length_guides[length]} {style_guides[style]}:

{text}

Summary:"""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def extract_key_points(self, text, num_points=5):
        prompt = f"""Extract the {num_points} most important key points from this text:

{text}

Provide them as a numbered list."""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
summarizer = DocumentSummarizer()

document = """
Artificial intelligence has transformed numerous industries...
[Long document text]
"""

# Different summary styles
print("Short summary:")
print(summarizer.summarize(document, "short"))

print("\nKey points:")
print(summarizer.extract_key_points(document, 3))
```

== Example 4: Language Tutor

Interactive language learning assistant:

```python
class LanguageTutor:
    def __init__(self, target_language):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.target_language = target_language
        self.conversation = []
    
    def practice_conversation(self, user_message):
        system_msg = f"""You are a friendly language tutor teaching {self.target_language}.
- Respond in {self.target_language}
- Correct mistakes gently
- Provide translations in parentheses
- Encourage the learner
- Suggest better ways to say things"""

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        
        self.conversation.append({"role": "user", "content": user_message})
        self.conversation.append({"role": "assistant", "content": reply})
        
        return reply
    
    def explain_grammar(self, topic):
        prompt = f"""Explain this {self.target_language} grammar concept clearly with examples:
        
Topic: {topic}

Include:
1. Clear explanation
2. 3-4 example sentences
3. Common mistakes to avoid
4. Practice exercises"""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

# Usage
tutor = LanguageTutor("Spanish")

print(tutor.practice_conversation("Hola! Cómo estás?"))
print(tutor.practice_conversation("Yo quiero aprende más español"))

print("\nGrammar explanation:")
print(tutor.explain_grammar("subjunctive mood"))
```

== Example 5: Code Explanation Tool

Explain code in natural language:

```python
class CodeExplainer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def explain_code(self, code, language, level="beginner"):
        level_guides = {
            "beginner": "for someone new to programming",
            "intermediate": "for someone with basic programming knowledge",
            "advanced": "for an experienced programmer"
        }
        
        prompt = f"""Explain this {language} code {level_guides[level]}:

```{language}
{code}
```

Include:
1. What the code does (high-level)
2. How it works (step-by-step)
3. Key concepts used
4. Potential improvements"""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def suggest_improvements(self, code, language):
        prompt = f"""Review this {language} code and suggest improvements:

```{language}
{code}
```

Focus on:
1. Performance
2. Readability
3. Best practices
4. Potential bugs

Provide improved code with explanations."""

        response = self.client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
explainer = CodeExplainer()

code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
"""

print(explainer.explain_code(code, "python", "beginner"))
print("\nSuggested improvements:")
print(explainer.suggest_improvements(code, "python"))
```

== Key Takeaways

- Complete examples demonstrate real-world applications
- Each example includes full working code
- Examples cover common use cases
- Code is production-ready with proper error handling
- Examples can be extended for specific needs
- All patterns follow best practices
- Examples demonstrate different AI capabilities
- Code is well-documented and maintainable

#pagebreak()

= Glossary of Terms

*API (Application Programming Interface)*: A set of rules and protocols for building and interacting with software applications.

*Asynchronous Programming*: A programming paradigm that allows operations to run concurrently without blocking execution.

*Base URL*: The foundational URL for API endpoints, different for each AI provider.

*Chat Completion*: An AI API call that generates conversational responses based on message history.

*Circuit Breaker*: A design pattern that prevents cascading failures by stopping requests to failing services.

*Embedding*: A numerical vector representation of text that captures semantic meaning.

*Environment Variable*: A dynamic value that can affect running processes, commonly used for configuration.

*Few-Shot Learning*: Providing a model with examples to guide its behavior on new tasks.

*Function Calling*: Allowing AI models to invoke predefined functions to extend their capabilities.

*JSON (JavaScript Object Notation)*: A lightweight data interchange format that is easy for humans to read and machines to parse.

*Large Language Model (LLM)*: A neural network trained on vast amounts of text data to understand and generate human language.

*Latency*: The time delay between making a request and receiving a response.

*Max Tokens*: The maximum number of tokens (words/word pieces) the model can generate in a response.

*Prompt*: The input text or instruction given to an AI model to generate a response.

*Prompt Engineering*: The practice of designing effective prompts to get desired outputs from AI models.

*Rate Limiting*: Controlling the frequency of requests to prevent abuse and manage resources.

*Semantic Search*: Finding information based on meaning rather than exact keyword matches.

*Streaming*: Receiving AI responses incrementally as they are generated rather than waiting for completion.

*System Prompt*: Instructions that define the AI model's behavior, personality, and constraints.

*Temperature*: A parameter controlling randomness in AI responses (0=deterministic, higher=more creative).

*Token*: The basic unit of text processing, roughly equivalent to a word or word fragment.

*Top P (Nucleus Sampling)*: An alternative to temperature for controlling response diversity.

*Virtual Environment*: An isolated Python environment with its own dependencies.

*Vision Model*: An AI model capable of analyzing and understanding images.

*Zero-Shot Learning*: Performing tasks without providing examples, relying only on the model's training.

#pagebreak()



= About the Author

Jan Bodnar is a programmer, teacher, and technical writer with extensive experience in software development and AI technologies. He has authored numerous programming tutorials and guides, helping thousands of developers learn new technologies and best practices.

With a passion for making complex technical concepts accessible to beginners, Jan focuses on creating clear, practical educational materials. His work spans multiple programming languages and technologies, with recent focus on artificial intelligence and machine learning applications.

Jan believes in learning by doing and emphasizes hands-on practice in all his educational materials. This ebook reflects his commitment to providing practical, actionable knowledge that readers can immediately apply to their own projects.

When not writing or coding, Jan enjoys exploring new AI models, contributing to open-source projects, and engaging with the developer community through forums, social media, and local meetups.

Connect with Jan:
- GitHub: github.com/janbodnar
- Website: zetcode.com
- Twitter: @janbodnar

#pagebreak()

= Acknowledgments

This book would not have been possible without the contributions and support of many individuals and organizations.

Special thanks to:

- The OpenAI team for creating groundbreaking AI models and accessible APIs
- The Anthropic team for Claude and their focus on AI safety
- The DeepSeek, Groq, and OpenRouter teams for making AI accessible
- The open-source community for countless tools and libraries
- Early readers and reviewers who provided valuable feedback
- The Python community for an amazing programming language and ecosystem
- All the developers sharing knowledge and building with AI

Thank you to everyone pushing the boundaries of what's possible with AI while working to ensure these technologies benefit humanity.

#pagebreak()

= Copyright and License

OpenAI with Python: A Beginner's Guide

Copyright © 2025 Jan Bodnar

All rights reserved.

This book is protected by copyright law. No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the author, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law.

The code examples in this book are provided for educational purposes. You are free to use, modify, and distribute the code examples in your own projects, both commercial and non-commercial, with or without attribution.

Disclaimer:

The information in this book is provided "as is" without warranty of any kind, either expressed or implied. The author and publisher shall not be liable for any damages arising from the use of this information.

API keys, model names, and pricing mentioned in this book were current at the time of writing but may change. Always refer to official documentation for the most current information.

For permission requests, corrections, or other inquiries, please contact the author through the channels listed in the "About the Author" section.

First Edition: December 2025

#pagebreak()

= Frequently Asked Questions

This chapter answers common questions about working with OpenAI and Python.

== General Questions

*What Python version do I need?*

Python 3.8 or higher is required. We recommend Python 3.10 or later for the best compatibility with modern libraries and features.

*Do I need to install TensorFlow or PyTorch?*

No. When using OpenAI APIs, you don't need machine learning frameworks. The OpenAI library handles all the communication with remote models.

*How much does it cost to use OpenAI APIs?*

Costs vary by provider and model. Free tiers are available from OpenRouter and Groq. OpenAI charges per token, with rates varying by model. Check each provider's pricing page for current rates.

*Can I use OpenAI offline?*

No. OpenAI APIs require internet connectivity. However, you can use local models with libraries like LLaMA.cpp or GPT4All for offline usage.

*What's the difference between ChatGPT and the OpenAI API?*

ChatGPT is a web interface for conversing with GPT models. The OpenAI API lets you programmatically integrate AI into your applications with more control and customization.

== Technical Questions

*How do I handle rate limits?*

Implement exponential backoff retry logic. If you hit rate limits frequently, upgrade your API plan or distribute requests over time.

*Why are my responses inconsistent?*

AI models are probabilistic. Lower the temperature parameter (closer to 0) for more consistent outputs. Also ensure your prompts are clear and specific.

*How can I make responses faster?*

- Use smaller, faster models
- Reduce max_tokens
- Enable streaming for perceived speed
- Cache frequent requests
- Use async for multiple requests

*How do I count tokens accurately?*

Use the `tiktoken` library for precise token counting:

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Your text here")
token_count = len(tokens)
```

*Can I fine-tune models?*

Some providers support fine-tuning (OpenAI, for example). This requires training data and additional costs. For most use cases, prompt engineering is sufficient.

== Troubleshooting

*Why do I get authentication errors?*

- Verify your API key is correct
- Check the key is loaded from environment variables
- Ensure the key is active on the provider's dashboard
- Try regenerating the key

*My code hangs or times out. What should I do?*

- Check your internet connection
- Increase timeout settings
- Try a different model or provider
- Check if the API service is operational

*Responses are cut off mid-sentence. Why?*

You've hit the max_tokens limit. Increase the `max_tokens` parameter in your API call.

*How do I debug API issues?*

- Enable verbose logging
- Print request and response data
- Test with minimal examples
- Check API status pages
- Review provider documentation

== Best Practices

*Should I cache responses?*

Yes, for frequently asked questions or repeated queries. This saves costs and improves response time. Implement TTL (time-to-live) for cache entries.

*How do I handle sensitive data?*

- Never send sensitive data to AI APIs unless necessary
- Sanitize inputs before sending
- Consider on-premise models for highly sensitive data
- Review provider privacy policies
- Implement data anonymization where possible

*What's the best way to test my AI application?*

- Write unit tests with mocked responses
- Create integration tests with test API keys
- Use smaller, cheaper models for testing
- Implement logging and monitoring
- Test edge cases and error conditions

*How often should I update my prompts?*

Review and refine prompts based on:
- User feedback
- Model performance
- New model capabilities
- Changing requirements

Test prompt changes before deployment.

== Development

*Should I use sync or async code?*

Use async for:
- Multiple concurrent requests
- High-traffic applications
- Better resource utilization

Use sync for:
- Simple scripts
- Learning and prototyping
- Sequential operations

*How do I manage conversation context?*

Maintain a message history array. Include all previous messages in each API call. Trim old messages if you hit token limits.

*What's the best way to structure my code?*

- Separate concerns (API client, business logic, UI)
- Use environment variables for configuration
- Implement proper error handling
- Write reusable functions
- Document your code
- Follow Python best practices (PEP 8)

== Deployment

*Can I deploy AI apps for free?*

Yes, using platforms like:
- Heroku (free tier)
- Railway
- Fly.io
- Render

Note: You still pay for API calls to AI providers.

*How do I scale my AI application?*

- Implement caching
- Use load balancers
- Deploy multiple instances
- Optimize database queries
- Monitor performance
- Use CDNs for static content

*Should I use containers (Docker)?*

Yes, for:
- Consistent deployment environments
- Easy scaling
- Better resource isolation
- Simplified CI/CD

== Security

*How do I secure my API keys?*

- Use environment variables
- Never commit keys to git
- Rotate keys regularly
- Use different keys for different environments
- Implement key access controls

*Is it safe to expose AI functionality to users?*

Yes, but:
- Validate all inputs
- Implement rate limiting
- Monitor for abuse
- Set spending limits
- Filter inappropriate content

*How do I prevent prompt injection attacks?*

- Validate and sanitize inputs
- Use system prompts wisely
- Implement content filtering
- Monitor unusual patterns
- Keep user inputs separate from instructions

== Performance

*How can I reduce latency?*

- Choose geographically closer servers
- Use smaller models when possible
- Implement request queuing
- Cache frequently requested data
- Use streaming for long responses

*What affects token costs?*

- Input length (prompt + context)
- Output length (completion)
- Model selected (larger = more expensive)
- Provider pricing

Optimize by:
- Crafting concise prompts
- Limiting max_tokens
- Choosing appropriate models
- Implementing caching

== Advanced Topics

*Can I use multiple models together?*

Yes, combine models based on strengths:
- Fast model for initial processing
- Powerful model for complex tasks
- Vision model for images
- Specialized models for specific domains

*How do I implement RAG (Retrieval-Augmented Generation)?*

1. Store documents in a vector database
2. Convert user query to embedding
3. Retrieve relevant documents
4. Include retrieved context in prompt
5. Generate response with context

*What about model hallucinations?*

Models sometimes generate false information. Mitigate by:
- Using lower temperature
- Requesting citations
- Implementing fact-checking
- Cross-referencing important facts
- Being transparent with users about AI limitations

*Can I build agents that use tools?*

Yes, using function calling. Define tools the AI can use, let it decide when to call them, execute the functions, and return results to the AI.

== Learning Resources

*Where can I learn more?*

- Official documentation (platform.openai.com)
- AI safety research (Anthropic, DeepMind)
- Online courses (DeepLearning.AI, Fast.ai)
- Community forums (Reddit, Discord)
- Research papers (arXiv.org)
- GitHub repositories
- Technical blogs

*Are there AI communities I can join?*

Yes:
- Reddit: r/OpenAI, r/MachineLearning
- Discord: Various AI servers
- Twitter/X: #AI, #LLM, #OpenAI
- Local AI/ML meetups
- Online hackathons

*How do I stay updated?*

- Subscribe to provider newsletters
- Follow AI researchers on social media
- Read AI news websites
- Attend virtual conferences
- Join mailing lists
- Watch YouTube channels focused on AI

== Final Thoughts

These FAQs cover common scenarios, but AI development is a rapidly evolving field. Stay curious, keep experimenting, and engage with the community. When stuck, don't hesitate to ask for help—the AI community is welcoming and collaborative.

Remember: there are no stupid questions. Everyone started as a beginner. Your questions help others who face similar challenges.

Keep learning, keep building, and enjoy the journey of working with AI!

#pagebreak()

= Recommended Reading

For deeper understanding of AI and related topics, consider these resources:

== Books

*Artificial Intelligence*
- "Artificial Intelligence: A Modern Approach" by Russell & Norvig
- "Life 3.0" by Max Tegmark
- "The Master Algorithm" by Pedro Domingos

*Machine Learning*
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Pattern Recognition and Machine Learning" by Christopher Bishop

*Python Programming*
- "Python Crash Course" by Eric Matthes
- "Fluent Python" by Luciano Ramalho
- "Effective Python" by Brett Slatkin

*AI Ethics*
- "Weapons of Math Destruction" by Cathy O'Neil
- "The Alignment Problem" by Brian Christian
- "Human Compatible" by Stuart Russell

== Research Papers

*Foundational*
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3

*Recent Advances*
- Browse arXiv.org for latest AI research
- Follow conference proceedings (NeurIPS, ICML, ACL)
- Read provider research blogs (OpenAI, Anthropic, DeepMind)

== Online Courses

*Beginner*
- "AI For Everyone" (DeepLearning.AI)
- "Introduction to AI with Python" (Harvard CS50)
- "Python for Everybody" (University of Michigan)

*Intermediate*
- "Machine Learning Specialization" (DeepLearning.AI)
- "Practical Deep Learning" (Fast.ai)
- "Natural Language Processing" (Stanford CS224N)

*Advanced*
- "Deep Learning Specialization" (DeepLearning.AI)
- "Advanced Machine Learning" (HSE University)
- "Reinforcement Learning" (David Silver)

== Websites and Blogs

*Official Documentation*
- OpenAI Platform Docs
- Anthropic Claude Docs
- Hugging Face Transformers

*Technical Blogs*
- OpenAI Blog
- Anthropic Blog
- Google AI Blog
- DeepMind Blog

*Community Resources*
- Towards Data Science
- Machine Learning Mastery
- Papers With Code
- AI Alignment Forum

== Podcasts

- "The AI Podcast" (NVIDIA)
- "Machine Learning Street Talk"
- "Lex Fridman Podcast"
- "TWiML AI"

== YouTube Channels

- Andrej Karpathy
- Two Minute Papers
- StatQuest
- 3Blue1Brown (for math foundations)

#pagebreak()


= Final Words

As you close this book and embark on your AI development journey, remember that technology is a tool for solving problems and creating value. The AI models you work with are powerful assistants that amplify human creativity and capability.

Start with simple projects. Build a chatbot that helps users find information. Create a tool that summarizes articles. Develop an application that analyzes customer feedback. Each project teaches new lessons and builds confidence.

Don't be discouraged by initial challenges. Every developer has struggled with API errors, unexpected model behavior, and debugging complex issues. These challenges are opportunities to learn and grow.

Contribute to the community. Share your projects, write tutorials, answer questions in forums, and help others learn. The knowledge you gain becomes more valuable when shared.

Think about the ethical implications of your work. Build applications that respect user privacy, handle data responsibly, and provide genuine value. Consider accessibility, fairness, and transparency in your designs.

Stay current with AI developments, but don't feel pressured to learn every new model or technique immediately. Focus on mastering fundamentals first. Deep understanding of core concepts serves you better than surface knowledge of many tools.

Remember that AI is evolving rapidly. Models improve, new capabilities emerge, and best practices evolve. Continuous learning is essential. Follow research, try new techniques, and adapt your skills.

But most importantly, build something that matters to you. Whether it's a tool for your own use, an application that helps others, or an experimental project that pushes boundaries—make it meaningful. Passion drives learning and sustains effort through challenges.

The future of AI is being written now. Your projects, however small they may seem, contribute to this evolving story. The applications you build today become the foundations for tomorrow's innovations.

You have the knowledge, the tools, and the resources. The only thing left is to begin. Open your editor, write your first few lines of code, make that first API call, and start building.

The AI revolution isn't something happening somewhere else—it's happening wherever developers like you choose to apply these tools to solve real problems.

So go forth and create. Build applications that amaze users, solve difficult problems, and make the world a little better. The possibilities are limitless, and your journey is just beginning.

Thank you for reading this book. I hope it serves you well as you build the future with AI.

Happy coding, and may your applications bring value and delight to all who use them!

---

*Jan Bodnar*

#datetime.today().display("[month repr:long] [day], [year]")

#align(center)[
  #v(3em)
  #text(size: 14pt, style: "italic")[
    "The best way to predict the future is to invent it."  
    — Alan Kay
  ]
  #v(2em)
  #text(size: 12pt)[
    End of Book
  ]
]


#pagebreak()

= Appendix C: Quick Start Checklist

Use this checklist to verify you've completed all setup steps:

== Environment Setup
- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] OpenAI library installed (`pip install openai`)
- [ ] Additional libraries installed (python-dotenv, requests, etc.)

== API Keys
- [ ] OpenRouter account created
- [ ] OpenRouter API key obtained
- [ ] .env file created in project root
- [ ] API keys added to .env file
- [ ] .env file added to .gitignore
- [ ] python-dotenv installed
- [ ] Environment variables loading correctly

== First Project
- [ ] Test script created
- [ ] Import statements working
- [ ] API client initialized
- [ ] First API call successful
- [ ] Response parsed correctly
- [ ] Error handling implemented

== Project Structure
- [ ] Project directory created
- [ ] requirements.txt file created
- [ ] README.md file created
- [ ] Source code directory created
- [ ] Configuration file created
- [ ] Git repository initialized
- [ ] .gitignore properly configured

== Code Quality
- [ ] Functions documented
- [ ] Error handling in place
- [ ] Logging implemented
- [ ] Code follows PEP 8
- [ ] No hardcoded secrets
- [ ] Comments added where needed

== Testing
- [ ] Test cases written
- [ ] Tests passing
- [ ] Edge cases considered
- [ ] Error conditions tested

== Next Steps
- [ ] Choose first real project
- [ ] Design application architecture
- [ ] Implement core functionality
- [ ] Add user interface
- [ ] Test thoroughly
- [ ] Deploy application
- [ ] Monitor and iterate

== Learning Path
- [ ] Complete all chapter exercises
- [ ] Build at least 3 practice projects
- [ ] Read official documentation
- [ ] Join AI community forums
- [ ] Share your work
- [ ] Help other learners

Use this checklist each time you start a new AI project to ensure you haven't missed any important steps!

