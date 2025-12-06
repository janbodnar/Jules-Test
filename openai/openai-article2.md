# OpenAI and Python: Streaming Responses, Working with Images, and System Prompts

This article follows the introduction to programming large language models in Python. We discuss streaming responses, working with images, and utilizing system prompts.

## Streaming Responses

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

## Multi-turn Conversation

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

```
py test.py
First question response:
"The capital of France is Paris."
Second question response: "The capital of Slovakia is Bratislava."
```

## Working with Images

The OpenAI library allows working with images, including their analysis and generation. For these tasks, it's necessary to use models capable of processing visual input (vision models), such as GPT-4o or, in this example, `meta-llama/llama-4-maverick-17b-128e-instruct`.

The example demonstrates how to use the OpenAI library to analyze an image through the Groq service. The model is sent a request containing the URL address of the image and a text prompt. Subsequently, the model returns a textual description of the image's content.

When using a model capable of processing visual input, it's necessary to structure the message content as a list that contains parts of type `text` and `image_url`.

Groq is a specialized [platform for artificial intelligence inference](https://groq.com/) (executing trained models) that achieves extremely high speed and low latency thanks to its own chips. (Do not confuse this with the Grok model from Elon Musk's xAI.)

Its strategy is focused exclusively on accelerating open-source large language models, such as Llama, Mixtral, or GPT-OSS, thereby supporting the community and transparency in AI. For developers and testing projects, Groq offers a generous free tier of access to its API, which has set daily and minute limits on the number of requests and tokens. These limits allow users to test the platform's speed for free before deciding on paid programs.

### Image Description

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

### Image Generation

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

## Audio Transcription

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

## Example Using the Gradio Library

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

## System Prompts and Persona Changing

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

```
You are a Python programming tutor. When students ask for help:
- Explain concepts step by step
- Provide code examples
- Point out common mistakes
- Suggest best practices
- Ask clarifying questions when needed
```

A behavioral prompt:

```
You are a friendly and patient kindergarten teacher. Always:
- Use simple, age-appropriate language
- Be encouraging and positive
- Use analogies and examples from children's lives
- Break down complex ideas into tiny steps
- Celebrate small victories and progress
```

A contextual prompt:

```
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

```
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
