// Document setup
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
  size: 12pt,
)

#set heading(numbering: "1.")

// Add spacing after headings
#show heading: it => { it; v(1em) }

// Title page
#align(center + horizon)[
  #text(size: 24pt, weight: "bold")[OpenAI with Python]
  #v(1em)
  #text(size: 18pt)[A Beginner's Guide]
  #v(2em)
  #text(size: 14pt)[by Jan Bodnar]
  #v(1em)
  #text(size: 12pt)[#datetime.today().display("[month repr:long] [day], [year]")]
]

#pagebreak()

// Table of contents
#outline(
  title: "Table of Contents",
  indent: auto,
)

#pagebreak()

// Introduction
= Introduction

Welcome to this ebook on using OpenAI's API with Python. This guide covers the basics of integrating AI models into your Python applications.

#lorem(100)

== Artificial Intelligence Overview
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. In recent years, AI has made significant advancements, particularly in natural language processing and generation.

== History of AI

== OpenAI library

== Large Language Models (LLMs)

#pagebreak()

// Chapter 1
= Getting Started

#lorem(100)

== Installing Dependencies

To begin, install the OpenAI Python library:

```bash
pip install openai
```

== Setting Up API Key


Obtain an API key from OpenAI and set it as an environment variable.

```Python
import os
import openai

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY")
```

#pagebreak()

// Chapter 2
= Making Your First API Call

== Completions Endpoint

Here's a simple example:

```Python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role": "user", "content": "Hello, world!"}]
)

print(response.choices[0].message.content)
```
#lorem(150)

// Chapter 3
= Basic Examples

#lorem(50)

== Streaming Responses

#lorem(50)

== Multi-turn Conversations

#lorem(50)

== Semantic Analysis

#lorem(50)

== Classification Tasks

#lorem(50)

= Working With Images

#lorem(150)

// Chapter 5
= System Prompts

#lorem(150)

// Chapter x
= Function Calling

#lorem(150)
