# --- Completions API ---

# API Question 1
# Set up your OpenAI client and make your first chat completion call. Use the model "gpt-4o-mini" and send this prompt: "What is one thing that makes Python a good language for beginners?". Print the model's response.

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

# Print just the text of the response (not the whole object). Then print the name of the model that responded and the total number of tokens used. Label each output.
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total Tokens Used:", response.usage.total_tokens)

# API Question 2
# Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
print("--- Responses for Different Temperatures ---")
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"Temperature {temp}: {response.choices[0].message.content}")

# Add a comment in your code answering: What do you notice about how the outputs differ? Which temperature would you use if you needed a consistent, reproducible output?
#
# The outputs differ in creativity and variability. Lower temperature (0) produces more consistent and precise responses, and higher temperature (1.5) shows more diverse and creative results.
# If I needed a consistent, reproducible output, I would use a lower temperature setting (0 or close to it).

# API Question 3
# Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three.

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
# Iterate over response.choices and print each one.
print("--- Three Different Completions ---")
for i, choice in enumerate(response.choices):
    print(f"Completion {i+1}: {choice.message.content}")


# API Question 4
# Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work.").

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
# Print the result. Add a comment: What happened, and why might you want to use max_tokens in a real application?
print("Response with max_tokens=15:", response.choices[0].message.content)
# The response is truncated due to the max_tokens limit, resulting in an incomplete explanation.
# Using max_tokens can help control the length of responses, which is useful for applications where brevity is important


# --- System Messages and Personas ---

# System Question 1
# Use a system message to give the model a personality, then ask it a question. Print the response.

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print("Response from Python tutor:", response.choices[0].message.content)

# Now change the system message to give the model a completely different personality (your choice) and ask the same question. Print that response too. Add a comment noting what changed.
messages = [
    {"role": "system", "content": "You are a sarcastic and witty Python tutor. You make fun of mistakes but still provide correct explanations."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print("Response from sarcastic Python tutor:", response.choices[0].message.content)
# The response changed in tone and style, becoming more humorous and teasing while still explaining the concept of list comprehensions.


# System Question 2

# The completions API is stateless — it has no memory of previous calls. The way to give a model context is to pass the conversation history yourself as a list of messages.

# Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# Print the model's response. Add a comment: Why does the model know Jordan's name, even though it's stateless?
print("Model's response:", response.choices[0].message.content)
# The model knows Jordan's name because we provided the conversation history in the messages list, so the model can reference previous interactions and maintain context within that single API call.


# --- Prompt Engineering ---

# Prompt Question 1 — Zero-Shot
# Ask the model to classify the sentiment of each review below as positive, negative, or mixed. Give it no examples — just the task description and the reviews. Print each result labeled with the review number.

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
print("--- Zero-Shot Sentiment Classification ---")
for i, review in enumerate(reviews):
    prompt = f"""
Classify the sentiment of the review below.
Respond with exactly one word: positive, negative, or mixed. No other text.

Review: "{review}"
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i+1} text: {review}")
    print(f"Review {i+1} Sentiment (zero-shot): {response.choices[0].message.content.strip()}")

# Prompt Question 2 — One-Shot
# Repeat the same task, but this time add one example before the reviews to show the model the format you want:
# Example:
# Review: "Fast shipping but the item arrived damaged."
# Sentiment: mixed

one_shot_example = (
    'Review: "Fast shipping but the item arrived damaged."\n'
    'Sentiment: mixed\n\n'
)
print("--- One-Shot Sentiment Classification ---")
for i, review in enumerate(reviews):
    prompt = f"""
Classify the sentiment of each review as exactly one word: positive, negative, or mixed.
Follow the format of the example below exactly. Respond with only the sentiment word.

{one_shot_example}Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    # Print the results.
    print(f"Review {i+1} text: {review}")
    print(f"Review {i+1} Sentiment (one-shot): {response.choices[0].message.content.strip()}")

# Add a comment: Did adding one example change the format or consistency of the output compared to Q1?
# Adding one example helps guide the model to produce output in the desired format and can improve consistency, especially if the task is complex or ambiguous.
# It provides a clear reference for how to structure the response, which helps with more accurate responses than with zero-shot prompting.


# Prompt Question 3 — Few-Shot
# Repeat the task again, this time with three examples.
# At least one example should be positive, one negative, and one mixed.
few_shot_examples = (
    'Review: "Fast shipping but the item arrived damaged."\n'
    'Sentiment: mixed\n\n'
    'Review: "The product quality is fantastic and the customer service was excellent."\n'
    'Sentiment: positive\n\n'
    'Review: "The app is very buggy and crashes frequently. Not recommended."\n'
    'Sentiment: negative\n\n'
)
print("--- Few-Shot Sentiment Classification ---")
for i, review in enumerate(reviews):
    prompt = f"""
Classify the sentiment of each review as exactly one word: positive, negative, or mixed.
Follow the format of the examples below exactly. Respond with only the sentiment word.

{few_shot_examples}Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    # Print the results
    print(f"Review {i+1} text: {review}")
    print(f"Review {i+1} Sentiment (few-shot): {response.choices[0].message.content.strip()}")

# Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
# Zero-shot is useful when you want a quick answer without needing to format examples, but it may be less accurate for complex tasks.
# One-shot can improve accuracy by providing a single example, which is helpful for tasks that require specific formatting or understanding.
# Few-shot is best when the task is complex or ambiguous, as multiple examples can help the model learn patterns and produce more consistent results.


# Prompt Question 4 — Chain of Thought

# Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. Label the final answer clearly.

problem_prompt = (
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "What is her final annual salary? Show your reasoning step by step before giving the final answer."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": problem_prompt}]
)
# Print the full response including the reasoning.
print("Model's response with reasoning:", response.choices[0].message.content)

# Add a comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# Asking the model to reason step by step encourages it to break down the problem into smaller, more manageable parts.
# This process can help the model avoid missing key details or making logical mistakes, which often leads to more accurate answers, especially for multi-step problems.


# Prompt Question 5 — Structured Output
# Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence).

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f"""
Analyze the following review and return the result as valid JSON only.

Respond ONLY with a valid JSON object. Do not include markdown code fences,
explanations, or any text before or after the JSON.

The JSON object must have exactly these keys:
- "sentiment": one of "positive", "negative", or "mixed"
- "confidence": a float from 0 to 1 representing how confident you are in your sentiment classification
- "reason": one sentence explaining your classification

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
# Print the raw response, then parse it with json.loads() and print each field separately, labeled.
print("Raw response:", response.choices[0].message.content)
# Add a try/except block to handle the case where the response is not valid JSON. If it fails, print the raw response so you can debug the prompt.
try:
    result = json.loads(response.choices[0].message.content)
    print("Sentiment:", result['sentiment'])
    print("Confidence:", result['confidence'])
    print("Reason:", result['reason'])
except json.JSONDecodeError:
    print("Failed to parse JSON. Raw response was:", response.choices[0].message.content)


# Prompt Question 6 — Delimiters
# Use triple backticks as delimiters to clearly separate the user's text from your instructions. Send the prompt below and print the result.

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("Response with instructions:", response.choices[0].message.content)

# Then send a second prompt using a passage that is not a set of instructions (any sentence or two of regular prose).
# Confirm that the model returns "No steps provided."
non_instruction_text = "The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of ATP, which is used as a source of chemical energy."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("Response with non-instruction text:", response.choices[0].message.content)

# Add a comment: What problem do delimiters help prevent?
# Delimiters help prevent confusion between the instructions and the content being analyzed.
# They clearly show which part of the input is user-provided one and which part is the prompt or instructions for the model.
# This reduces the chances of misinterpretation and improves the accuracy of the response.


# --- Local Models with Ollama ---

# Ollama Question 1

# In your terminal, run the following prompt using Ollama (you installed it during the lesson):

# ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

# Then run the same prompt using the OpenAI API in Python (as you've been doing above). Print the OpenAI response.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)
print("OpenAI response:", response.choices[0].message.content)
# OpenAI response: A large language model is an artificial intelligence system designed to understand and generate human language by analyzing vast amounts of text data. It uses deep learning techniques to predict and produce coherent text based on the patterns it has learned, enabling it to perform various language tasks such as translation, summarization, and conversation.

# Paste the Ollama output as a multi-line string comment in your code.
"""(.venv) ➜  python200-homework git:(assignments_05) ✗ ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
Thinking...
Okay, the user wants me to explain what a large language model        
is in two sentences. Let me start by breaking down the basics.        
First, a large language model is a type of artificial                 
intelligence that can understand and generate human-like text.        
That's a good start.                                                  
                                                                      
Now, I need to make sure the second sentence adds more details.       
Maybe mention that they process and understand vast amounts of        
text, which makes them powerful for various tasks. Also,              
including that they can generate text in different languages          
would be helpful. Let me check if I'm not missing anything. Oh,       
and maybe emphasize their ability to perform tasks like               
answering questions or writing essays. That should cover it in        
two sentences.                                                        
...done thinking.                                                     
                                                                      
A large language model is an AI system designed to understand 
and generate human-like text, enabling it to process and 
comprehend vast amounts of information. It can generate text in 
multiple languages and perform tasks such as answering questions 
or writing essays, making it highly versatile.  

**Two sentences**:  
"A large language model is a system that can understand and 
generate human-like text, enabling it to process and analyze 
large volumes of information. It can perform tasks like 
answering questions or creating content in various languages, 
making it highly versatile."
"""

# Then add another comment answering: What differences did you notice between the two responses?
# What is one advantage and one disadvantage of running a model locally?

# The OpenAI response is more concise and direct, while the Ollama response provides a more detailed explanation and includes the reasoning process.
# One advantage of running a model locally is that it can provide more control over data privacy and security since the data does not leave the local environment. This one is one of the most important for me personally.
# One disadvantage is that local models may require significant computational resources and may not be as up-to-date or optimized as cloud-based models like OpenAI's offerings. Unfortunately, this is my casem cause I have a laptop without GPU.
