#!/usr/bin/env python3
"""
Generate comprehensive OpenAI Python ebook in Typst format.
Transforms markdown source material into a 100-120 page beginner's guide.
"""

def generate_ebook():
    """Generate the complete ebook content."""
    
    ebook = """// Document setup
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
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.")

// Add spacing after headings  
#show heading: it => { 
  it
  v(0.8em) 
}

// Code blocks styling
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

// Table of contents
#outline(
  title: "Table of Contents",
  indent: auto,
  depth: 2,
)

#pagebreak()

= Introduction

Welcome to this comprehensive guide on using OpenAI's API with Python. This book is designed for beginners who want to harness the power of artificial intelligence in their Python applications. Whether you're a student, developer, or curious enthusiast, this guide will take you from zero to creating sophisticated AI-powered applications.

Throughout this book, you'll learn how to integrate large language models into your Python projects, work with various AI capabilities, and build practical applications. Each chapter builds upon the previous one, with progressive complexity that ensures you develop a solid understanding of both the concepts and their practical implementation.

== What You'll Learn

This guide covers essential topics for working with OpenAI and Python:

- Understanding artificial intelligence and large language models
- Setting up your Python development environment for AI projects  
- Working with the OpenAI Python library and compatible APIs
- Creating conversational AI applications with chat completions
- Generating and analyzing images with vision models
- Implementing streaming responses for better user experience
- Using function calling and tools for advanced interactions
- Mastering prompt engineering and system prompts
- Building practical AI agents and applications
- Best practices for cost optimization and error handling

== Who This Book Is For

This book is designed for beginners with basic Python knowledge. You should be familiar with fundamental Python concepts like variables, functions, and basic control structures. No prior experience with AI or machine learning is required. We'll introduce all AI concepts from scratch and build them up progressively.

== How to Use This Book

Each chapter follows a consistent structure with clear explanations, code examples, and practical exercises. We recommend reading the chapters in order, as later chapters build on concepts introduced earlier. All code examples are complete and ready to run, making it easy to experiment and learn by doing.

At the end of each chapter, you'll find a summary of key takeaways and best practices. These summaries help reinforce what you've learned and provide quick reference material for future projects.

#pagebreak()

= Understanding Artificial Intelligence

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI has evolved from simple rule-based systems in the 1950s to sophisticated neural networks capable of processing vast amounts of data.

The history of AI begins with Alan Turing's seminal work in the 1950s, followed by the development of expert systems in the 1970s, machine learning in the 1990s, and deep learning in the 2010s. Today, AI powers numerous applications across various domains.

Key domains of AI include natural language processing (NLP), computer vision, robotics, expert systems, and reinforcement learning. These domains enable machines to understand human language, recognize images, make autonomous decisions, and continuously improve through experience.

== Early Foundations

The foundations of AI were laid in the 1940s and 1950s, when mathematicians and scientists began exploring whether machines could think. Alan Turing, a British mathematician, made groundbreaking contributions with his 1950 paper "Computing Machinery and Intelligence," which posed the famous question "Can machines think?" He proposed the Turing Test as a criterion for machine intelligence.

In 1956, the Dartmouth Conference marked the official birth of AI as a field. John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon organized this historic summer workshop, where they coined the term "Artificial Intelligence."

Early AI research focused on symbolic AI, also called "Good Old-Fashioned AI" (GOFAI). This approach assumed that human intelligence could be reduced to symbol manipulation. Researchers developed programs like the Logic Theorist (1956) and Arthur Samuel's checkers program (1952), which demonstrated early machine learning concepts.

== Machine Learning and Deep Learning

Machine learning is a subset of AI that enables systems to learn from data without explicit programming. It includes supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards).

These algorithms identify patterns, make predictions, and improve their performance over time. Common machine learning algorithms include decision trees, random forests, support vector machines, and clustering algorithms.

Neural networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts during training, allowing the network to learn complex patterns from data.

Deep learning emerged as the dominant paradigm in the 2010s. In 2012, AlexNet, a deep convolutional neural network, won the ImageNet competition by a large margin, demonstrating that deep neural networks trained on GPUs could dramatically outperform traditional computer vision approaches.

== Large Language Models

Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like language. LLMs learn statistical patterns, grammar, facts, and reasoning abilities from billions of text examples, enabling them to perform various language tasks without task-specific training.

The foundation for modern LLMs came from the Transformer architecture, introduced by Google researchers in the paper "Attention Is All You Need" (2017). Transformers use self-attention mechanisms to process entire sequences simultaneously, making them more efficient and effective than previous architectures.

OpenAI's GPT (Generative Pre-trained Transformer) series demonstrated the power of scaling language models. GPT-3 (2020), with 175 billion parameters, showed remarkable few-shot learning abilities, performing diverse tasks from translation to code generation with minimal examples.

ChatGPT, released by OpenAI in November 2022, brought LLMs to mainstream attention, gaining 100 million users in just two months. This demonstrated the incredible potential of LLMs for practical applications.

== Applications of AI

AI has transformed creative industries and technical fields alike. In creative writing, AI assists authors with generating ideas, drafting content, and even completing stories based on prompts. Tools like GPT models can produce coherent narratives, poetry, and technical documentation.

Image generation has become remarkably sophisticated with models like DALL-E, Midjourney, and Stable Diffusion. These systems create original artwork, photorealistic images, and design concepts from text descriptions, enabling artists and designers to rapidly prototype visual ideas.

Music composition using AI involves generating melodies, harmonies, and even complete musical pieces. AI models analyze patterns in existing music to create new compositions in various styles, assisting musicians in the creative process.

Code development has been revolutionized by AI assistants like GitHub Copilot, which suggests code completions, generates functions, and helps debug issues. These tools accelerate development cycles and help programmers learn new frameworks and languages more efficiently.

Healthcare has been revolutionized by AI. Machine learning models analyze medical images to detect cancer, diabetic retinopathy, and other conditions with accuracy matching or exceeding human radiologists. AI accelerates drug discovery by predicting molecular properties and identifying promising compounds.

== Modern AI Chatbots

Modern AI chatbots use large language models to engage in natural conversations, answer questions, and assist with various tasks.

*ChatGPT* (OpenAI) revolutionized conversational AI with its ability to engage in nuanced dialogue, explain complex topics, write code, and assist with creative projects. It supports extended conversations with context awareness.

*Claude* (Anthropic) is known for safety-focused AI with strong reasoning capabilities and helpful, harmless, and honest responses.

*Gemini* (Google) combines multiple modalities (text, images, audio) to provide comprehensive assistance, leveraging Google's vast knowledge base.

*DeepSeek* (China) represents a new generation of efficient AI models, optimized for performance and cost-effectiveness.

*Copilot* (Microsoft) integrates AI assistance directly into development environments and productivity tools, helping with code completion and task automation.

== Key Takeaways

- AI has evolved from simple rule-based systems to sophisticated neural networks
- Large language models understand and generate human-like text  
- LLMs are trained on massive datasets and have billions of parameters
- You can use pre-trained models via APIs without training them yourself
- Different models excel at different tasks - choose based on your needs
- AI applications span creative industries, healthcare, code development, and more
- The Transformer architecture revolutionized natural language processing
- Modern chatbots like ChatGPT have brought AI to mainstream adoption

#pagebreak()
"""

    # Write to file
    with open('openai/openai-ebook.typ', 'w') as f:
        f.write(ebook)
    
    print("Ebook header and first chapters generated successfully")
    print(f"Current size: {len(ebook)} characters")

if __name__ == "__main__":
    generate_ebook()
