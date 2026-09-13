# Step 1: Setup
# Add your imports at the top of the file. Load your API key from .env and print a confirmation message. Add an assert statement to verify that the groundwork_docs/ directory exists before your code tries to use it.
# An assert statement stops the program early with a clear error message if a condition is not met. For example:
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
# Adjust the path as needed.

if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Step 2: Load the Documents
# Load all documents from groundwork_docs/ using SimpleDirectoryReader.
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(str(docs_dir)).load_data()
# Print:
# How many documents were loaded
print(f"Loaded {len(documents)} documents.")
# The file name of each document
for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown")
    print(f"Document: {file_name}")
# Hint: each Document object has a metadata dictionary with a "file_name" key.


# Step 3: Build the Index and Query Engine
# Build a VectorStoreIndex from the loaded documents and create a query engine with similarity_top_k=3.
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
# Print a short confirmation message once the index is ready, such as:
# Index built successfully. Ready to answer questions.
print("Index built successfully. Ready to answer questions.")


# Step 4: Query the Assistant
#
# Run the five queries below through your query engine.
# For each one, print:
# The question
# The answer from the model
# The top retrieved source node: document name, similarity score, and the first 200 characters of the chunk text
#
# Use a loop — do not repeat the same code block five times.
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for question in questions:
    print(f"Question: {question}")
    response = query_engine.query(question)
    print(f"Answer: {response}")
    # The top retrieved source node: document name, similarity score, and the first 200 characters of the chunk text
    source_nodes = response.source_nodes
    if source_nodes:
        top_node = source_nodes[0]
        doc_name = top_node.node.metadata.get("file_name", "Unknown")
        similarity_score = top_node.score
        chunk_text = top_node.node.get_content()[:200]
        print(f"Top Source Node: {doc_name}, Similarity: {similarity_score}, Chunk Text: {chunk_text}")
    else:
        print("No source nodes retrieved.")
# Question: What are Groundwork's hours on weekends?
# Answer: Groundwork's hours on weekends are 8:00 AM to 5:00 PM.
# Top Source Node: faq.txt, Similarity: 0.8137201480087354, Chunk Text: Frequently Asked Questions
#
# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
# - Saturday and Sunday: 8:00 AM to 5:00 PM
# - We are closed on Thanksgiving Day and Christmas Day.
#
# Locations
# - Downtown: 42
# Question: Do you offer any dairy-free milk options?
# Answer: All dairy-free options are available at no extra charge.
# Top Source Node: seasonal_specials.txt, Similarity: 0.7858166491992239, Chunk Text: Seasonal Specials — Current Menu
#
# These drinks are available for a limited time only.
#
# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre
# Question: How does the loyalty program work?
# Answer: The loyalty program is free to join and allows customers to earn one point for every dollar spent. Once a customer accumulates 100 points, they can redeem them for any free drink on the menu. Customers can sign up for the loyalty program either at the register or on the company's website.
# Top Source Node: faq.txt, Similarity: 0.7659387112084327, Chunk Text: Frequently Asked Questions
#
# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
# - Saturday and Sunday: 8:00 AM to 5:00 PM
# - We are closed on Thanksgiving Day and Christmas Day.
#
# Locations
# - Downtown: 42
# Question: How did Groundwork Coffee get started?
# Answer: Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala, while Sam had managed a community center in his hometown. They believed in the connection between good coffee and strong communities, leading them to establish Groundwork with a commitment to sourcing only fair-trade, sustainably grown beans directly from small farms.
# Top Source Node: our_story.txt, Similarity: 0.9005474240563793, Chunk Text: Our Story
#
# Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala. S
# Question: Do you offer catering or wholesale orders?
# Answer: Yes, we offer both catering and wholesale orders.
# Top Source Node: wholesale_catering.txt, Similarity: 0.8578859221511365, Chunk Text: Wholesale and Catering
#
# Wholesale Coffee
# We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds or

# After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate? Did any of the answers surprise you?
#
# - The assistant sounded confident and accurate for all five queries. No hedging language, even when the retrieval score was relatively low (0.77 for the loyalty program question).
# - What surprised me was the answer for question #2: "Do you offer any dairy-free milk options?".
# It only said "All dairy-free options are available at no extra charge.". I think the reason is that the top source was seasonal_specials.txt (a limited-time drinks menu), and not a dedicated milk/menu document.


# Step 5: Find a Failure
#
# Ask the assistant a question you expect it to struggle with. Good candidates include: something vague or ambiguous, something that requires combining information from more than one document, or a question where the answer is simply not in the documents.
# Print the full response and all three retrieved source nodes (document name, similarity score, and first 200 characters of text).
struggle_question = "What are your most popular drinks?"
print(f"Struggle question: {struggle_question}")
struggle_response = query_engine.query(struggle_question)
print(f"Struggle answer: {struggle_response}")
failure_source_nodes = struggle_response.source_nodes
if failure_source_nodes:
    for i, node in enumerate(failure_source_nodes):
        doc_name = node.node.metadata.get("file_name", "Unknown")
        similarity_score = node.score
        chunk_text = node.node.get_content()[:200]
        print(f"Source Node {i+1}: {doc_name}, Similarity: {similarity_score}, Chunk Text: {chunk_text}")
else:
    print("No source nodes retrieved.")
# Struggle question: What are your most popular drinks?
# Struggle answer: The most popular drinks are likely the ones listed on the menu that are commonly ordered by customers. These include Espresso, Americano, Latte, Cappuccino, Cold brew, Pour-over, Chai latte, and Matcha latte.
# Source Node 1: seasonal_specials.txt, Similarity: 0.7725373341014866, Chunk Text: Seasonal Specials — Current Menu
#
# These drinks are available for a limited time only.
#
# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre
# Source Node 2: menu.txt, Similarity: 0.7267544114547906, Chunk Text: Groundwork Coffee Co. — Menu
#
# Drinks
# - Espresso (single or double): $2.50 / $3.00
# - Americano: $3.00
# - Latte (hot or iced): $4.50
# - Cappuccino: $4.00
# - Cold brew: $4.50
# - Pour-over (rotating single or
#                       Source Node 3: faq.txt, Similarity: 0.7212023400603342, Chunk Text: Frequently Asked Questions
#
# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
#                                          - Saturday and Sunday: 8:00 AM to 5:00 PM
#                                                                                 - We are closed on Thanksgiving Day and Christmas Day. \
#
#                                                                                                                                   Locations \
#                                                                                                                                   - Downtown: 42

# Then add a comment explaining:
# What you asked and why you expected it to be hard
# - I asked "What are your most popular drinks?" because this information is not explicitly stated in any of the documents.
#
# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# - The model retrieved relevant documents (seasonal specials, menu, and FAQ), but there is no exact information about the popularity of the drinks.
# - Thus, the model had to guess and imagine based on common knowledge about coffee shop menus. This led to a vague answer that may not be accurate for this exact Groundwork Coffee shop.
#
# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# - For my view, the model still sounded confident in its response, despite full guess mode.
# - This suggests that AI-generated responses can be misleading, even with a full confidence provided by AI. We need to always think critically and check the information from LLMs.
#
# What you would change about the system to improve it
# - To improve the system, I would implement a detection system to check when results do not contain proper information to answer the question.
# - This can be a threshold for similarity scores and/or checking for the presence of key phrases related to popularity.


# Step 6: Reflection
# Add a comment block at the end of project_06.py answering the following:
#
# The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?
# - The equivalent LlamaIndex implementation took around 20 lines of code, and it is way too less than the manual implementation of semantic RAG that I'm very glad.
# - This highlights the value of using a framework like LlamaIndex. This works not only with Data science frameworks, but in general.
#
# You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
# - A different use case for this approach could be in the legal industry. Law firms often have vast amounts of legal documents, case files, and research materials.
# A RAG system could be used to quickly retrieve relevant information from this extensive database.
# Moreover, I have a friend we work with on a similar system for internation laws.
#
# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
# - One failure mode that RAG cannot fully prevent is the generation of confident yet incorrect answers by the LLM as we can't control it directly.
# So, even when the retrieval is working correctly, but LLM still is lacking context, it may try to fill in the gaps with generic info from pre-training.
