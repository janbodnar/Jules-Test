# Python OpenAI Library - Introduction to Programming AI Models

In this article, you'll learn how to program large language models in Python using the OpenAI library. We'll show you basic communication with AI models and creating simple programs.

We'll need to have Python installed, VS Code, and an account on the OpenRouter platform and/or directly on the platform of one of the model creators, such as Anthropic, DeepSeek, OpenAI, or xAI. All examples can be run either for free or with minimal costs.

## Large Language Models

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

## Communicating with Models via REST API

Models typically provide access via REST API, which allows sending requests and receiving responses using HTTP.

```
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

## OpenAI Library for Python

The [OpenAI library](https://platform.openai.com/docs/quickstart/quickstart?context=python) for Python is an official SDK (Software Development Kit) that simplifies access to REST API from OpenAI and other compatible providers of large language models. This library is designed to provide a unified and consistent interface for working with various AI models, regardless of whether they are located on OpenAI servers, with other providers like DeepSeek, OpenRouter, or are locally hosted models.

The main advantage of this library is its flexibility and versatility. It allows developers to easily integrate advanced AI capabilities into their applications including text generation, conversational interfaces, natural language processing, image analysis, and function calling. The library abstracts the complexity of HTTP communication and JSON response processing, allowing developers to focus on application logic instead of low-level technical details.

We install the OpenAI library using the pip package manager.

```
$ pip install openai
```

OpenAI is built on several key concepts:

- **Client** - the main object that manages communication with the API
- **Messages** - an array of messages that form a conversation with the model
- **Models** - identifiers of specific AI models
- **Completions** - responses generated by models
- **Functions/Tools** - external functions that the model can have called

Before using the library, it's necessary to create an instance of the OpenAI client. The client is configured using an API key and optionally a base URL if we're using a provider other than OpenAI.

```
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

```
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

## OpenRouter

OpenRouter is an ecosystem in the field of artificial intelligence that functions as a centralized platform for accessing dozens of different models from multiple providers. It's something like an "App Store" for AI models - a place where you can easily discover, test, and use the latest and best models from the entire AI market, all through a unified and consistent API. OpenRouter solves one of the biggest problems in the AI ecosystem: fragmentation and incompatibility between different model providers.

One of the biggest advantages is access to models that aren't publicly available anywhere else. So-called "stealth" or testing models often appear here, which are only available for a limited time and often for free. These models may offer top performance or specialized capabilities that aren't yet part of the mainstream.

We can register at [openrouter.ai](https://openrouter.ai) via Google, GitHub, or email. After registration, we create an API key that we can start using immediately.

Here's a simple example:

```
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

## Example with DeepSeek API

The OpenAI library allows changing the `base_url` to connect to other compatible APIs, such as DeepSeek. This way you can use various models from different providers with the same code.

```
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

## Token Consumption

Tokens are the basic unit used to measure consumption when working with models. Every text you send to the model or that the model generates is divided into tokens. Understanding tokens is crucial for cost optimization and efficient use of the API.

Tokens are text units that can be words, parts of words, characters, or even just spaces. For example, the word "programming" may be divided into multiple tokens, while short words like "a" or "the" are often one token. Different models use different tokenizers, so the same text can have a different number of tokens depending on the model used.

Monitoring token consumption is important for several reasons:

- **Costs** - API calls are billed based on the number of tokens
- **Limits** - each model has a maximum number of tokens per request
- **Performance** - longer token sequences can slow down responses
- **Optimization** - monitoring tokens helps optimize applications

In the following example, we'll show how to monitor token consumption during an API call:

```
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

## Completions

The most common interaction with a model is in the form of question and answer or simple chat. For common questions without the need for context or conversation history, it's often more effective to use `completions` instead of `chat.completions`. Completions are designed for direct text generation based on a single prompt, which is ideal for simple questions, text summarization, or content generation.

When to use completions vs. chat.completions:

- `completions` - for simple questions, text generation, summarization
- `chat.completions` - for conversations with history, multi-question interactions

In this example, we'll use completions because it's a simple question without needing to preserve context of previous messages.

```
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

## Chat Completions

Chat completions are the most commonly used function of the OpenAI library because they allow creating natural, context-rich conversations with AI models. Unlike simple completions, chat completions preserve conversation history and allow the model to understand previous messages, leading to much more relevant and coherent responses.

The latest versions of OpenAI are transitioning to the newer Responses API, but since these aren't yet supported by other providers, in this guide we're still using the older Completions API.

Chat completions are ideal for scenarios where we need interactive conversations, context preservation, or use of system prompts to define model behavior. It's the right choice for chatbots, virtual assistants, or any applications requiring continuity in communication.

Each chat completions API call consists of an array of messages (`messages`), where each message has a specific role and content. There are three main types of roles:

- **system** - defines behavior and context for the model (optional but recommended)
- **user** - messages from the user
- **assistant** - responses generated by the model

The following practical example creates an interactive chat assistant:

```
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
