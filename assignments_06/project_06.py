# Step 1: Setup
# Add your imports at the top of the file. Load your API key from .env and print a confirmation message. Add an assert statement to verify that the groundwork_docs/ directory exists before your code tries to use it.
# An assert statement stops the program early with a clear error message if a condition is not met. For example:
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
# Adjust the path as needed.

if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")

# No explicit OpenAI client is needed here - LlamaIndex reads OPENAI_API_KEY from the
# environment itself for both the embedding model and the LLM.


# Step 2: Load the Documents
# Load all documents from groundwork_docs/ using SimpleDirectoryReader.
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
# Actual output:
#
# Question: What are Groundwork's hours on weekends?
# Answer: Groundwork's hours on weekends are 8:00 AM to 5:00 PM.
# Top Source Node: faq.txt, Similarity: 0.8146095829639313, Chunk Text: Frequently Asked Questions
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
# Top Source Node: seasonal_specials.txt, Similarity: 0.7850362596353715, Chunk Text: Seasonal Specials — Current Menu
#
# These drinks are available for a limited time only.
#
# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre
# Question: How does the loyalty program work?
# Answer: The loyalty program is free to join and allows customers to earn one point for every dollar spent. Once a customer accumulates 100 points, they can redeem them for any free drink on the menu. Customers can sign up for the loyalty program either at the register or on the company's website.
# Top Source Node: faq.txt, Similarity: 0.7664594242791599, Chunk Text: Frequently Asked Questions
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
# Top Source Node: our_story.txt, Similarity: 0.9010782481555117, Chunk Text: Our Story
#
# Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala. S
# Question: Do you offer catering or wholesale orders?
# Answer: Yes, catering and wholesale orders are available.
# Top Source Node: wholesale_catering.txt, Similarity: 0.8574578657681237, Chunk Text: Wholesale and Catering
#
# Wholesale Coffee
# We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds or

# After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate? Did any of the answers surprise you?
#
# - The assistant sounded confident on all five queries, and four of the five answers are genuinely accurate.
#   There is no hedging anywhere, and crucially the tone is identical whether the top score is 0.90 (our_story.txt) or 0.77 (the loyalty question) -
#   confidence in the wording tells you nothing about how well retrieval actually went.
#
# - Question #2 ("Do you offer any dairy-free milk options?") is the weak one and it surprised me.
#   The answer was just "All dairy-free options are available at no extra charge." - which answers a question about pricing, not the question I asked.
#   It never lists the actual options (oat, almond, soy). The cause is retrieval: the top node was seasonal_specials.txt at 0.7850, a limited-time drinks menu
#   that happens to repeat the phrase "Dairy-free" several times, so it embeds close to the query. menu.txt, which has the real list of milks, did not come top.
#   So a keyword-ish coincidence ("dairy-free" appearing often) beat the document that actually answers the question.
#
# - Question #3 is interesting in the opposite direction: the top node is faq.txt and the 200-character preview shows only hours and locations,
#   which looks wrong at first glance. But the answer is correct and detailed, because the loyalty section is further down inside that same chunk.
#   That is a good reminder that the preview I print is not the whole chunk the model actually received.
#
# - Question #5's answer ("Yes, catering and wholesale orders are available.") is correct but thin - it confirms rather than explains,
#   even though wholesale_catering.txt came back at 0.8575 and contains the minimum order sizes and lead times it could have summarised.


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
# Actual output:
#
# Struggle question: What are your most popular drinks?
# Struggle answer: The most popular drinks at the coffee shop are Espresso, Americano, Latte (hot or iced), Cappuccino, Cold brew, Pour-over (rotating single origin), Chai latte, and Matcha latte.
# Source Node 1: seasonal_specials.txt, Similarity: 0.7722280121058696, Chunk Text: Seasonal Specials — Current Menu
#
# These drinks are available for a limited time only.
#
# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre
# Source Node 2: menu.txt, Similarity: 0.7284946817910829, Chunk Text: Groundwork Coffee Co. — Menu
#
# Drinks
# - Espresso (single or double): $2.50 / $3.00
# - Americano: $3.00
# - Latte (hot or iced): $4.50
# - Cappuccino: $4.00
# - Cold brew: $4.50
# - Pour-over (rotating single or
# Source Node 3: faq.txt, Similarity: 0.7218209751331671, Chunk Text: Frequently Asked Questions
#
# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
# - Saturday and Sunday: 8:00 AM to 5:00 PM
# - We are closed on Thanksgiving Day and Christmas Day.
#
# Locations
# - Downtown: 42

# Then add a comment explaining:
# What you asked and why you expected it to be hard
# - I asked "What are your most popular drinks?" because this information is not explicitly stated in any of the documents.
#
# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# - Retrieval did not fail in the usual sense: it returned the three most sensible documents available
#   (seasonal_specials.txt 0.7722, menu.txt 0.7285, faq.txt 0.7218). Those really are the drink-related documents.
# - The failure is that popularity data does not exist anywhere in the corpus. Nothing in these documents ranks or counts anything.
# - The model guessed anyway, and it guessed in the most misleading way possible: it took the full drinks list from menu.txt and relabelled it.
#   The answer was "The most popular drinks at the coffee shop are Espresso, Americano, Latte (hot or iced), Cappuccino, Cold brew, Pour-over, Chai latte, and Matcha latte."
#   That is simply the entire menu presented as if it were a popularity ranking. It did not invent fake drinks - every item is real - which makes it harder to spot.
#   It invented the *relationship* ("most popular") rather than the facts, and a reader who trusted it would come away believing Groundwork had told them something it never said.
#
# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# - The tone did not change at all. This answer is phrased with exactly the same flat confidence as the our_story.txt answer that scored 0.90 and was fully correct.
# - Note also that the top score here (0.7722) is not far below the score for the dairy-free question (0.7850), which the system did attempt to answer normally.
#   So the similarity score alone does not cleanly separate "answerable" from "not answerable" either.
# - What this suggests: confidence is a property of the writing style, not of whether the system knows anything. There is no signal in the output itself
#   that distinguishes a grounded answer from an invented one, so the check has to come from outside the model - from citations the reader can follow, or from the system refusing to answer.
#
# What you would change about the system to improve it
# - A similarity floor: if the top node scores below roughly 0.80, return "I don't have that information" instead of answering.
#   This one change would have caught the failure, though as noted above the margin is thin, so the threshold needs tuning against real queries rather than guessing.
# - A stricter prompt telling the model to answer only from the provided context and to say so explicitly when the context does not contain the answer.
#   The warmup showed this works: "What is BrightLeaf's stock price?" correctly returned "not provided in the context information" rather than a guess.
# - Show the source document and score alongside every answer in the UI, so a user can see the answer came from a menu rather than from sales data.
# - Longer term, the honest fix is to close the data gap: if customers keep asking what is popular, add the sales figures to the document set.
#   RAG can only retrieve what someone has written down.


# Step 6: Reflection
# Add a comment block at the end of project_06.py answering the following:
#
# The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?
# - Three lines do the actual work here: SimpleDirectoryReader(...).load_data(), VectorStoreIndex.from_documents(documents), and index.as_query_engine(similarity_top_k=3).
#   Those three replace the manual version's file reading, chunking, embedding calls, vector storage, cosine similarity search and prompt assembly.
# - What that tells me is less "frameworks save typing" and more that the framework encodes decisions I would otherwise have to make badly:
#   chunk size and overlap, batching embedding requests, how the retrieved chunks get formatted into the prompt. I got sensible defaults for free.
# - The trade-off is that those defaults are now invisible. I could not tell you what chunk size this used without looking it up, and Step 4 question #3
#   showed me a chunk boundary I did not choose and could not see. Building it manually once first is what makes the abstraction safe to use.
#
# You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
# - A different use case for this approach could be in the legal industry. Law firms often have vast amounts of legal documents, case files, and research materials.
# A RAG system could be used to quickly retrieve relevant information from this extensive database.
# Moreover, I have a friend we work with on a similar system for internation laws.
#
# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
# - The model filling a gap in the retrieved context with something that sounds plausible, and stating it with full confidence.
#   Step 5 is exactly this: retrieval worked and returned the right documents, but the specific fact (which drinks are popular) was not in them,
#   so the model repurposed the menu into a popularity ranking. Retrieval did its job; the generation step still produced an unsupported claim.
# - RAG narrows what the model can get wrong, but it cannot make the model say "this is not in what you gave me" unless you push it to.
#   The gap between "the documents do not contain X" and "the model declines to answer about X" has to be closed deliberately -
#   through prompting, similarity thresholds, or evaluation - and even then it is a reduction in risk, not a guarantee.


# --- Extension C: Add a New Document ---
#
# What document you added and what information it contains
# - I added seasonal_specials.txt to groundwork_docs/. It is a limited-time drinks menu listing four rotating specials with prices and descriptions:
#   Iced Lavender Lemonade ($5.00), Horchata Latte ($5.50), Pumpkin Maple Oat Latte ($5.50) and Honey Cardamom Latte ($5.00),
#   plus a note that the specials rotate roughly every eight weeks. Three of the four are marked dairy-free.
#
# What query you used to test it and whether the assistant retrieved the correct content
# - The document is picked up automatically by Step 2 (the loader now reports 5 documents, including seasonal_specials.txt) with no code change at all.
# - It is retrieved as the top source node for "Do you offer any dairy-free milk options?" at 0.7850 and for "What are your most popular drinks?" at 0.7722,
#   so the assistant is clearly indexing and searching the new content.
# - Honestly, it retrieves it a bit too eagerly: for the dairy-free question it outranked menu.txt, which is the document that actually lists the milk options,
#   because the specials text repeats "Dairy-free" several times. So the new document is definitely in the index - it is just competing with the better source.
#
# Why this demonstrates an advantage of RAG over fine-tuning
# - Adding knowledge cost one text file and a rebuild of the index. No training run, no labelled examples, no waiting, no risk of degrading anything the model already did well.
# - That matters most for exactly this kind of content: the specials rotate every eight weeks, so a fine-tuned model would be out of date almost immediately
#   and would need retraining each cycle. With RAG, updating the menu is editing a file.
# - The knowledge also stays inspectable and removable. I can see which document an answer came from, and if a special ends I delete the file and the assistant
#   stops mentioning it. Knowledge baked into weights cannot be pointed at or cleanly taken back out.
