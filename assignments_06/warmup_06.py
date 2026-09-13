from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PDFReader
from openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

# Concepts Question 1
# #
# Three teams at a software company are each building a different AI project.
# Add a comment block to your code that identifies the best approach — prompt engineering, fine-tuning, or RAG — for each scenario, and gives a 1-2 sentence explanation of your reasoning.
#
# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
#
# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online.
# They have 3,000 examples their in-house writers produced over the years.
#
# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received.
# She does not need this to work for any other document.

# Concepts Question 2
#
# AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. Add a comment to your code answering this:
#
# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
#
# Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?

# Concepts Question 3
#
# The steps below make up a complete RAG pipeline, but they are out of order.
# Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.

steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt"
    "Split text into chunks",
    "Embed the user's query",
]

# "Extract text from source documents" - the text in each document in the list is extracted and stored in the text variable.
# "Split text into chunks" - the system breaks down larger documents into smaller manageable chunks for processing.
# "Convert text chunks into embeddings" - the system converts each chunk into a vector embedding that captures its meaning.
# "Receive the user's query" - the bot gets the user's question they want an answer for. It repeats in an infinity loop, or until "quit" is entered.
# "Embed the user's query" - the script converts the user's query into a vector embedding to then compare with documents embeddings.
# "Retrieve the most relevant chunks" - the system compares the query embeddings with document chunks' embeddings in order to find the most relevant pieces of information to add to then the prompt.
# "Inject retrieved chunks into the prompt - the script injects the relevant chunks of data into a prompt that will be then sent to the LLM.
# "Generate a response from the LLM" - the LLM processes the given prompt which includes both the user's query and the relevant document chunks, and generates a response.


# --- Keyword RAG ---

# The following questions use the keyword retrieval function from the lesson. Copy the function below into your warmup_06.py — you will call it in the questions that follow.

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Question 1
# Run simple_keyword_retrieval with verbose=True on the query and documents below. Print the name of the selected document.
query = "What are your hours on weekends?"
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
print(f"\nSelected document: {simple_keyword_retrieval(query, documents, verbose=True)}")
# Selected best match: loyalty.txt
# Selected document: [('loyalty.txt', 'Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.')]

# After running the function, add a comment explaining which document was selected and why.
# The function selected "loyalty.txt" based on reverse sorted array of scores that is a misleading, cause the word "weekends" from the query overlaps with the content of "hours.txt".
# BUT, cause of keyword-based retrieval the scores of "loyalty.txt", "hours.txt" and "hiring.txt" are the same (1), so it shows "loyalty.txt" as the first option among 3 equil ones.
# Not a reliable approach at all.


# Keyword Question 2
# Run the same function with this second query using the same documents from Q1:

query = "Do you have anything without caffeine?"
print(f"\nSelected document: {simple_keyword_retrieval(query, documents, verbose=True)}")
# No overlapping keywords found.
# Selected document: [('None found', 'No relevant content.')]

# Add a comment explaining:
# Which document was selected
# Whether keyword RAG got this right — and why or why not
# What kind of retrieval would do better here

# - No document was selected, cause there were no overlapping keywords between the query and the documents.
# - Keyword RAG did not get this right, cause it relies on exact word overlap to retrieve relevant documents that is not very reliable. In this case the query is asking about "caffeine", but there is no "caffeine" word in documents despite "menu.txt" might have relevant information about caffeinated and non-caffeinated options.
# - A semantic retrieval would do better here, cause it could understand the meaning of the query and retrieve relevant information from "menu.txt" about caffeinated and non-caffeinated options.


# Keyword Question 3
#
# Before running any code, predict which document will be selected for the query below. Write your prediction and your reasoning as a comment first, then run the code to check.

# faq.txt, cause of "Sign up" in it
query = "How do I sign up for rewards?"
print(f"\nSelected document: {simple_keyword_retrieval(query, documents, verbose=True)}")
# No overlapping keywords found.
# Selected document: [('None found', 'No relevant content.')]

# Was your prediction correct? If the result surprised you, add a comment explaining what happened.
#
# My prediction was incorrect, as no document was selected.
# I expected "faq.txt" to be selected due to the phrase "Sign up" being present in it, but then I realized that "faq.txt" is not present in the "documents" list.


# --- Semantic RAG Concepts ---

# Semantic Question 1

# Add a comment block answering the following in your own words. Try not to just copy the definitions from the lesson — explaining a concept in your own words is a good sign that you have understood it.
#
# What is a vector embedding? (1-2 sentences)
# A vector embedding is how data is represented as a list of numbers (a vector) in a multi-dimensional space. The closer the vector to each other and their directions the more similar are data examples.
#
# Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
# The chunk with a cosine similarity score of 0.85 is more relevant to the given query, cause their directions are closer and the data relationship with the chunk are closer, meaning they have similar meanings.
#
# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# Semantic search can find a relevant chunk even when none of the exact words from the query appear in the chunk, cause it relies on the meaning and context (cosine similarity scores), and not just matching specific keywords.


# Semantic Question 2
#
# Keyword RAG and semantic RAG handle the same problem differently. Copy this table into your code as a comment and fill in the right column:
#
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|---------------------------------------------------------------------------------|
# | What is compared?          | Exact word overlap                | The meaning and context (cosine similarity scores)                              |
# | What is retrieved?         | Full document                     | The most relevant chunks of the document based on meaning and context           |
# | Can it handle synonyms?    | No                                | Yes                                                                             |
# | Storage format             | Plain text dictionary             | Vector embeddings in a vector database                                          |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score (0 to 1, where 1 is very similar and 0 is not similar)  |



# --- LlamaIndex ---

# For this section you will build a small LlamaIndex pipeline using the Brightleaf Solar PDFs from the lesson. These documents should already be familiar from the lesson material.
# Path note: The brightleaf_pdfs/ directory is in the lesson folder, not the assignments folder. Point SimpleDirectoryReader to it using a path relative to where you run your script — for example:

# Adjust this path as needed based on your local folder structure.
# API note: These questions make a small number of calls to the OpenAI embeddings API to build the vector index. The cost is very low (typically less than one cent), but make sure your .env file has a valid key before running.

if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LlamaIndex Question 1
#
# Build an in-memory LlamaIndex pipeline using the Brightleaf Solar PDFs and run the two queries below.
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# Load documents directly from PDFs in the folder

def process_in_memory_llama_index_pipeline(docs_dir_path, questions, similarity_top_k=3):
    docs = SimpleDirectoryReader(docs_dir_path, file_extractor={".pdf": PDFReader()}).load_data()
    # Build a vector index automatically (handles chunking + embeddings)
    index = VectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    # For each query, print:
    # The question
    # The answer from the model
    # For each of the 3 retrieved source nodes: the similarity score and the first 150 characters of the chunk text
    # Use similarity_top_k=3.
    for q in questions:
        print(f"\nQ: {q}")
        response = query_engine.query(q)
        print("A:", response)

        for node_with_score in response.source_nodes:
            print(f"Node ID: {node_with_score.node.node_id}")
            print(f"Similarity Score: {node_with_score.score:.4f}")
            print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
            print("-" * 30)

docs_dir_path = "assignments_06/resources/brightleaf_pdfs"
assert Path(docs_dir_path).exists(), f"Document directory not found: {docs_dir_path}"

process_in_memory_llama_index_pipeline(docs_dir_path, questions)

# After printing the results, add a comment for each query answering:
#     Do the retrieved chunks look relevant to the question?
# The ones with the highest Similarity Score definitely yes.
#
#     Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# The model's response sounds confident and specific, it does not hedge with phrases like "based on the context" or "I'm not sure". It provides pretty clear and focused answers to the questions.
#
#     Did anything unexpected get retrieved?
# Nothing specific, except I needed to fix the PDF parsing to support encoding format.
# Also, there were irrelevant snippets on lower Similarity Score retrieved, but that is expected as they are less relevant to the query.


# LlamaIndex Question 2
# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5. Print the response and source node scores for both runs.
similarity_top_k_values = [1, 5]
for k in similarity_top_k_values:
    print(f"\nRunning query with similarity_top_k={k}")
    process_in_memory_llama_index_pipeline(docs_dir_path, [questions[0]], similarity_top_k=k)

# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.
#
# TBH, the response did not change significantly between similarity_top_k=1 and similarity_top_k=5.
# I think, it is cause of the most relevant information was already retrieved with similarity_top_k=1.
# Maybe a bit more retrieved context (with similarity_top_k=5) which provided additional information that might be useful for a more comprehensive answer, but not in this case.


# LlamaIndex Question 3
#
# Try a query you think the pipeline might struggle with — something vague, something that spans multiple documents, or something where the information might not be in the documents at all.
# Print the response and all retrieved chunks.
test_query = "What is BrightLeaf's mission and how does it approach sustainability?"
print(f"\nRunning 'struggle' query: {test_query}")
process_in_memory_llama_index_pipeline(docs_dir_path, [test_query])

# Running 'struggle' query: What is BrightLeaf's mission and how does it approach sustainability?
#
# Q: What is BrightLeaf's mission and how does it approach sustainability?
# A: BrightLeaf's mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. The company approaches sustainability by not only focusing on engineering and building solar installations but also by being educators, partners, and advocates for a more resilient and equitable power grid. They emphasize community well-being, energy literacy, local empowerment, and workforce training programs to ensure a long-lasting impact beyond just energy savings.
# Node ID: b60834df-9043-4dc2-a645-c51db576eb76
# Similarity Score: 0.8859
# Text Snippet: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract...
# ------------------------------
# Node ID: 7dbf0056-9578-4796-8cea-9173406dd58c
# Similarity Score: 0.8522
# Text Snippet: EcoVolt Energy (2022 Partnership)
# BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid
# solutions to ru...
# ------------------------------
# Node ID: d8dd4f63-5762-4f5c-8262-9fb9c6111946
# Similarity Score: 0.8459
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------

# Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle this kind of query better.
#
# - I expected the model to struggle a bit with this query, cause it is asking about the company's mission and approach to sustainability which might vague (at least from what I thought) and info split across multiple documents and not directly in a single chunk.
# BUT, the model provided a pretty comprehensive and relevant answer. I suppose, that it was able to retrieve and pack information from multiple relevant chunks effectively.
# - TBH, I don't know how to handle this kind of query better, but I heard about a graph-based retrieval that can be applicable for complex RAGs and vague questions.


# LlamaIndex Question 4
#
# Using the same index and query engine you built in Q1, evaluate one response using LlamaIndex's built-in evaluators.
# Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as the judge LLM (refer to the "RAG Evaluation using LlamaIndex" section of lesson 4 for the exact import and setup pattern). Run them on this query:

q = "What employee benefits does BrightLeaf offer?"

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Create Judge LLM
llm = LlamaIndexOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

docs = SimpleDirectoryReader(docs_dir_path, file_extractor={".pdf": PDFReader()}).load_data()
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

# Get response to query
response = query_engine.query(q)

# Print both scores
print(f"Test query: {q}")
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Then run the evaluators again on a query you expect to produce a lower-quality response — for example, a question about something that is clearly not in the Brightleaf documents.
test_q = "What is BrightLeaf's stock price?"
test_response = query_engine.query(test_q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=test_q, response=response)
print(f"Test query: {test_q}")
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=test_q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Test query: What employee benefits does BrightLeaf offer?
# Faithfulness Evaluation: 1.0
# Relevancy Result: 1.0
#
# Test query: What is BrightLeaf's stock price?
# Faithfulness Evaluation: 1.0
# Relevancy Result: 0.0

# After printing both sets of scores, add a comment block answering:
#
# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A faithfulness score of 1.0 means that the response is completely faithful to the source documents.
# In other words, all the information in the response can be supported by real information without any hallucinations or imaginary data.
# A score of 0.0 would show that the response is not faithful at all.
# In other words, it contains information that cannot be supported by the real information and likely includes hallucinations.

# What does a relevancy score measure, and how is it different from faithfulness
# - A relevancy score measures how relevant the response is to the user's query. It means how well the information in the response addresses the question asked.
# - This is different from faithfulness, that measures how the information in the response is accurate and supported by the source documents

# Did the scores change between your two queries? If so, why do you think that happened?
# - The faithfulness score did not change between the two queries, it remained at 1.0 for both.
# I think this is cause the model did not hallucinate anything in both responses.
# - BUT, the relevancy score changed significantly between the two queries (from 1.0 to 0.0).
# For the first query about employee benefits, the relevancy score was 1.0, so the response is highly relevant to the question.
# For the second query about stock price, the relevancy score dropped to 0.0, so the response is not relevant at all (stock price information was not present in the source documents).

# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# - The "LLM-as-a-judge" approach uses LLM to check the quality of responses from another LLM in a RAG system.
# - This approach is used for RAG evaluation, cause it helps with a more holistic and detailed assessment of the responses.
# It is useful when simple metrics can't capture for example a partial correctness or relevance.
