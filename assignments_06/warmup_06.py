from dotenv import load_dotenv
import os
import string
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.readers.file import PDFReader

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
#
# My answers:
#
# Scenario A is RAG.
# The content changes every quarter, so anything baked into the model goes stale the moment the policies are updated.
# With RAG I can re-index the new PDFs and the assistant is the current. It can cite the exact policy it got answers from - which is exactly what a legal team need.
#
# Scenario B is Fine-tuning.
# The problem here is a style and not facts. The style is rare enough that the model didn't train on a lot of it.
# 3,000 in-house examples is exactly the kind of dataset that teaches a consistent voice, which you cannot reliably get by describing "dry and minimalist" in a prompt.
#
# Scenario C is Prompt engineering.
# A two-page report suits well in the context window, so she can just paste it in and ask.
# Building a retrieval pipeline or fine-tuning a model for one usage document is way too much work than the problem deserves.

# Concepts Question 2
#
# AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. Add a comment to your code answering this:
#
# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
#
# Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?
#
# My answer:
#
# A confidently wrong answer is more harmful because it removes the signal that would have made me go and check.
# "I am not sure" hands the problem back to me and I open the source document. A flat, specific, well-formatted answer tells me the work is already done, so I just use it.
# The damage is not only that the answer is wrong - it is that the wrong answer gets acted on without review.
#
# Tone matters because we read fluency as competence. We are used to humans hedging when they are unsure, so confident phrasing reads as a signal of knowledge.
# An LLM does not work that way: the tone comes from the training data's style, not from any internal measure of certainty, so a fabricated answer sounds exactly as sure as a correct one.
# That broken link between confidence and correctness is what makes hallucinations hard to catch.
#
# Real example: a nurse asks an AI assistant for the maximum dose of a drug for a pediatric patient.
# The model confidently states an adult dose with no warning. Because it sounded authoritative and specific, nobody double-checks the reference, and the patient is overdosed.
# It should have said "I am not sure, please confirm against the formulary", the reference would have been checked and the harm avoided.

# Concepts Question 3
#
# The steps below make up a complete RAG pipeline, but they are out of order.

steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]

# Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.

# The list below is the same eight steps, now in the correct order, with a one-sentence description of each.
# Steps 1-3 happen once. Steps 4-8 happen every time a user asks something (querying):

# 1. "Extract text from source documents" - the text in each document in the list is extracted and stored in the text variable.
# 2. "Split text into chunks" - the system breaks down larger documents into smaller manageable chunks for processing.
# 3. "Convert text chunks into embeddings" - the system converts each chunk into a vector embedding that captures its meaning.
# 4. "Receive the user's query" - the bot gets the user's question they want an answer for. It repeats in an infinity loop, or until "quit" is entered.
# 5. "Embed the user's query" - the script converts the user's query into a vector embedding to then compare with documents embeddings.
# 6. "Retrieve the most relevant chunks" - the system compares the query embeddings with document chunks' embeddings in order to find the most relevant pieces of information to add to then the prompt.
# 7. "Inject retrieved chunks into the prompt" - the script injects the relevant chunks of data into a prompt that will be then sent to the LLM.
# 8. "Generate a response from the LLM" - the LLM processes the given prompt which includes both the user's query and the relevant document chunks, and generates a response.


# --- Keyword RAG ---

# The following questions use the keyword retrieval function from the lesson. Copy the function below into your warmup_06.py — you will call it in the questions that follow.

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
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nSelected document: {result[0][0]}")

# Actual output:
# Query tokens (filtered): ['hours', 'weekends', 'what', 'your']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=1 -> ['weekends']
# [hiring.txt] overlap=1 -> ['your']
# [loyalty.txt] overlap=1 -> ['your']
#
# Selected best match: loyalty.txt
# Selected document: loyalty.txt

# After running the function, add a comment explaining which document was selected and why.
#
# "loyalty.txt" was selected, and it is the wrong answer - the question is about weekend hours, so "hours.txt" is the document a human would pick.
#
# Why it happened: three documents tie on score.
# - "hours.txt" scores 1 because it genuinely contains "weekends".
# - "hiring.txt" and "loyalty.txt" both score 1 only because they contain the word "your" ("your resume", "your choice").
#   "your" is not in the stopword list (the list has "you", not "your"), so it survives filtering and counts as a real keyword match.
# With a three-way tie at score 1, `scores.sort(reverse=True)` falls through to the second element of the tuple - the file name - and sorts it descending.
# "loyalty.txt" > "hours.txt" > "hiring.txt" alphabetically, so loyalty.txt ends up first and wins.
#
# So the retrieval is decided by an alphabetical tie-break on the file name, not by relevance.
# This shows two weaknesses of keyword RAG: a single incidental stopword-like word ("your") is enough to score as high as the genuinely relevant match,
# and the integer score is too coarse to separate documents, so ties are common and are broken arbitrarily.


# Keyword Question 2
# Run the same function with this second query using the same documents from Q1:

query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nSelected document: {result[0][0]}")

# Actual output:
# Query tokens (filtered): ['anything', 'caffeine', 'do', 'have', 'without']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=0 -> []
# [hiring.txt] overlap=0 -> []
# [loyalty.txt] overlap=0 -> []
#
# No overlapping keywords found.
# Selected document: None found

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

# My prediction (written before running the code):
# I expect "loyalty.txt". The query is about signing up for rewards, and loyalty.txt is the document about the rewards
# program - it talks about joining, earning points and redeeming them for a free drink. It is for sure the right document for a human reader.

query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nSelected document: {result[0][0]}")

# Actual output:
# Query tokens (filtered): ['do', 'how', 'i', 'rewards', 'sign', 'up']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=0 -> []
# [hiring.txt] overlap=0 -> []
# [loyalty.txt] overlap=0 -> []
#
# No overlapping keywords found.
# Selected document: None found

# Was your prediction correct? If the result surprised you, add a comment explaining what happened.
#
# My prediction was wrong, and this is an interesting part: nothing at all was selected, not even a bad match.
#
# loyalty.txt is obviously the correct document by meaning, but it shares zero words with the query.
# The query says "sign up" and "rewards"; the document says "join", "loyalty program", "points" and "free drink".
# Every one of those is a synonym or paraphrase, and keyword matching cannot see synonyms - it only compares literal tokens.
# The remaining query words ("how", "do", "i", "up") are common words that happen not to be in this particular stopword list, but they do not appear in the documents either.
#
# This is a really strange failure in warmup exercise: the answer is here in loyalty.txt and keyword RAG returns nothing.
# Semantic retrieval would handle it, because "sign up for rewards" and "join our loyalty program" embed to nearby vectors even with no shared words.


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

# LlamaIndex Question 1
#
# Build an in-memory LlamaIndex pipeline using the Brightleaf Solar PDFs and run the two queries below.
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

docs_dir_path = "assignments_06/resources/brightleaf_pdfs"
assert Path(docs_dir_path).exists(), f"Document directory not found: {docs_dir_path}"


def build_index(docs_dir_path):
    """Load the PDFs and build the vector index (handles chunking + embeddings)."""
    docs = SimpleDirectoryReader(docs_dir_path, file_extractor={".pdf": PDFReader()}).load_data()
    return VectorStoreIndex.from_documents(docs)


def run_queries(query_engine, questions):
    """Print the question, the answer, and each source node's score + first 150 chars."""
    for q in questions:
        print(f"\nQ: {q}")
        response = query_engine.query(q)
        print("A:", response)

        for node_with_score in response.source_nodes:
            print(f"Similarity Score: {node_with_score.score:.4f}")
            print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
            print("-" * 30)


# Build the index ONCE and reuse it for Q1-Q4. Embedding the PDFs is the expensive part
# (it is the only step that costs API calls), so there is no reason to redo it per question.
index = build_index(docs_dir_path)
query_engine = index.as_query_engine(similarity_top_k=3)

run_queries(query_engine, questions)

# Actual output:
#
# Q: What employee benefits does BrightLeaf offer?
# A: BrightLeaf offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, and wellness programs. They also provide financial security benefits like life insurance, disability insurance, and a 401(k) retirement plan with a company match. Additionally, BrightLeaf offers parental leave, work flexibility options, professional development opportunities, mentorship programs, and access to free online courses through their Learning Hub.
# Similarity Score: 0.9114
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------
# Similarity Score: 0.8189
# Text Snippet: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract...
# ------------------------------
# Similarity Score: 0.8187
# Text Snippet: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f...
# ------------------------------
#
# Q: What are BrightLeaf's security policies?
# A: BrightLeaf's security policies include maintaining layered defenses for networks, requiring multi-factor authentication and VPN with device certificates for critical system access, rotating credentials every 90 days, encrypting customer data in transit and at rest, enforcing least privilege with perimeter firewalls and cloud security groups, centralizing logs with anomaly detection, following an incident response plan based on NIST 800-61 guidance, conducting tabletop exercises, providing employee training on security, conducting quarterly RBAC reviews, ensuring vendor security compliance, aligning with ISO 27001 practices, tracking security metrics, and fostering a culture of accountability and continuous improvement.
# Similarity Score: 0.8848
# Text Snippet: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f...
# ------------------------------
# Similarity Score: 0.8407
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------
# Similarity Score: 0.8231
# Text Snippet: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract...
# ------------------------------

# After printing the results, add a comment for each query answering:
#
# --- Query 1: "What employee benefits does BrightLeaf offer?" ---
#
#     Do the retrieved chunks look relevant to the question?
# Only the top one really is. Node 1 (0.9114) is the benefits document and it carries the whole answer.
# Nodes 2 and 3 score far lower (0.8189 and 0.8187) and are the mission statement and the network security policy - neither is about benefits.
# The gap between 0.91 and 0.82 is the useful signal here: the top chunk is clearly on-topic and the other two are just "the next closest thing in the index".
#
#     Does the model's response sound confident and specific, or does it hedge?
# Completely confident and specific. It lists medical, vision, wellness, life and disability insurance, 401(k) with match, parental leave, professional development and mentorship,
# with no hedging language at all - no "based on the context", no "I'm not sure". The tone reads like a benefits page written by HR.
#
#     Did anything unexpected get retrieved?
# Yes - the security policy chunk ("Network and Data Security") came back for a benefits question.
# It is a good example of top_k forcing the retriever to return 3 chunks whether or not 3 relevant chunks exist.
# Also worth noting: the PDF text extraction produces a mangled character ("multi■f" instead of "multi-f"), which is an encoding artefact of the source PDF, not of LlamaIndex.
#
# --- Query 2: "What are BrightLeaf's security policies?" ---
#
#     Do the retrieved chunks look relevant to the question?
# Same pattern. Node 1 (0.8848) is the security document and answers the question fully.
# Nodes 2 and 3 (0.8407, 0.8231) are the employee benefits intro and the mission statement - both off-topic.
#
#     Does the model's response sound confident and specific, or does it hedge?
# Again fully confident, and this answer is dense with specifics: MFA, VPN with device certificates, 90-day credential rotation, least privilege, NIST-based incident response, ISO 27001.
# No hedging at all.
#
#     Did anything unexpected get retrieved?
# The benefits document showed up in the security query and the security document showed up in the benefits query - the two queries retrieved each other's documents as filler.
# This is the mirror image of the first query and reinforces the same point: with a six-document corpus and top_k=3, ranks 2 and 3 are frequently noise.
# The saving grace is that the LLM mostly ignored the irrelevant chunks rather than trying to work them into the answer.


# LlamaIndex Question 2
# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5. Print the response and source node scores for both runs.
similarity_top_k_values = [1, 5]
for k in similarity_top_k_values:
    print(f"\nRunning query with similarity_top_k={k}")
    # Same index, only the retriever's top_k changes - no re-embedding needed.
    run_queries(index.as_query_engine(similarity_top_k=k), [questions[0]])

# Actual output:
#
# Running query with similarity_top_k=1
#
# Q: What employee benefits does BrightLeaf offer?
# A: BrightLeaf offers a comprehensive benefits package that includes health benefits such as medical insurance, vision benefits, and wellness programs. They also provide financial security through life, disability, and retirement benefits, including a 401(k) plan with a company match. Additionally, BrightLeaf offers parental leave, work flexibility options, professional development opportunities, mentorship programs, and resources for diversity, equity, and inclusion.
# Similarity Score: 0.9114
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------
#
# Running query with similarity_top_k=5
#
# Q: What employee benefits does BrightLeaf offer?
# A: BrightLeaf offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, and wellness programs. They also provide financial security benefits like life insurance, disability insurance, and a 401(k) retirement plan with a company match. Additionally, BrightLeaf offers parental leave, work flexibility, professional development opportunities, mentorship programs, and access to free online courses through their Learning Hub.
# Similarity Score: 0.9114
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------
# Similarity Score: 0.8189
# Text Snippet: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract...
# ------------------------------
# Similarity Score: 0.8187
# Text Snippet: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f...
# ------------------------------
# Similarity Score: 0.8133
# Text Snippet: EcoVolt Energy (2022 Partnership)
# BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid
# solutions to ru...
# ------------------------------
# Similarity Score: 0.7880
# Text Snippet: Overview
# This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The period
# includes a growth phase, a temporary dip i...
# ------------------------------

# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.
#
# The facts did not change, but the level of detail did.
#
# With top_k=1 the answer is short and stays at the category level: "health, vision, and wellness benefits, financial security and retirement benefits,
# parental leave and work flexibility, as well as learning, inclusion, and career growth opportunities". Correct, but it just names the section headings.
#
# With top_k=5 the answer names the actual benefits: medical insurance, life and disability insurance, 401(k) with company match, mentorship programs, the Learning Hub.
# That extra specificity comes from the same benefits document - the retriever returned more chunks of it, not new sources.
#
# Is more context always better? No, and this run shows why.
# Going from 1 to 5 pulled in the EcoVolt partnership (0.8133) and the earnings report (0.7880), which have nothing to do with employee benefits.
# Here the model ignored them, so the answer improved. But every extra chunk is more tokens (more cost, more latency) and one more chance for the model
# to anchor on something irrelevant. Notice the scores flatten out - 0.9114 then 0.8189, 0.8187, 0.8133, 0.7880 - so after rank 1 the retriever is
# mostly guessing. The right top_k depends on whether an answer is concentrated in one chunk or genuinely spread across several;
# for a small corpus like this, a similarity threshold would be a better control than a fixed k.


# LlamaIndex Question 3
#
# Try a query you think the pipeline might struggle with — something vague, something that spans multiple documents, or something where the information might not be in the documents at all.
# Print the response and all retrieved chunks.
test_query = "What is BrightLeaf's mission and how does it approach sustainability?"
print(f"\nRunning 'struggle' query: {test_query}")
run_queries(query_engine, [test_query])

# Actual output:
#
# Running 'struggle' query: What is BrightLeaf's mission and how does it approach sustainability?
#
# Q: What is BrightLeaf's mission and how does it approach sustainability?
# A: BrightLeaf's mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. The company approaches sustainability by not only focusing on engineering and building solar installations, but also by being educators, partners, and advocates for a more resilient and equitable power grid. They emphasize community well-being, stable job creation, and lowering household electricity costs through initiatives like neighborhood microgrids, community solar arrays, and local workforce training programs. Additionally, BrightLeaf invests in energy literacy, local empowerment, and community engagement to ensure a long-lasting impact beyond just technology deployment.
# Similarity Score: 0.8858
# Text Snippet: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract...
# ------------------------------
# Similarity Score: 0.8523
# Text Snippet: EcoVolt Energy (2022 Partnership)
# BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid
# solutions to ru...
# ------------------------------
# Similarity Score: 0.8455
# Text Snippet: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m...
# ------------------------------

# Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle this kind of query better.
#
# What I expected:
# I picked this query because it is two questions in one ("what is the mission" and "how does it approach sustainability") and because I assumed
# sustainability would be spread across the mission statement, the partnerships document and possibly the product specs.
# I expected the retriever to fetch one document well and miss the rest, giving a lopsided answer.
#
# What actually happened:
# It handled it better than I expected. The mission statement came back at 0.8858 and carried most of the answer, and the EcoVolt partnership chunk (0.8523)
# genuinely contributed the microgrid and community-solar details. The model merged the two into a coherent answer covering both halves of the question.
# The third chunk (employee benefits, 0.8455) was irrelevant and was ignored.
#
# But the "success" is partly luck, and this is the real lesson:
# the mission statement already discusses sustainability in its own text, so a single well-matched chunk happened to answer both halves.
# If the sustainability information had genuinely lived only in a separate document, a single embedding of the combined query could easily have matched
# neither half well - a two-part question averages into one vector that is not especially close to either topic.
#
# What I would change:
# - Query decomposition: split a multi-part question into sub-questions, retrieve for each, then synthesise. LlamaIndex has SubQuestionQueryEngine for exactly this.
# - A similarity floor so clearly off-topic chunks (like the benefits one here) are dropped instead of padding out top_k.
# - Reranking the retrieved chunks with a cross-encoder, which judges query-chunk pairs directly and is much better than raw cosine at pushing noise down.


# LlamaIndex Question 4
#
# Using the same index and query engine you built in Q1, evaluate one response using LlamaIndex's built-in evaluators.
# Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as the judge LLM (refer to the "RAG Evaluation using LlamaIndex" section of lesson 4 for the exact import and setup pattern). Run them on this query:

q = "What employee benefits does BrightLeaf offer?"

# Create Judge LLM
llm = LlamaIndexOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Reuse the same index and query engine built for Q1 - no need to re-read and re-embed the PDFs.

# Get response to query
response = query_engine.query(q)

# Print both scores (and the answer being judged, so the scores can be interpreted)
print(f"\nTest query: {q}")
print(f"Answer: {response}")
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Then run the evaluators again on a query you expect to produce a lower-quality response — for example, a question about something that is clearly not in the Brightleaf documents.
test_q = "What is BrightLeaf's stock price?"
test_response = query_engine.query(test_q)

print(f"\nTest query: {test_q}")
print(f"Answer: {test_response}")
faithfulness_result = faithfulness_evaluator.evaluate_response(query=test_q, response=test_response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=test_q, response=test_response)
print("Relevancy Result: " + str(relevancy_result.score))

# Actual output:
#
# Test query: What employee benefits does BrightLeaf offer?
# Answer: BrightLeaf Solar offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, and wellness programs. They also provide financial security through life, disability, and retirement benefits. Additionally, BrightLeaf offers parental leave, work flexibility, professional development opportunities, mentorship programs, and access to online courses for continuous education.
# Faithfulness Evaluation: 1.0
# Relevancy Result: 1.0
#
# Test query: What is BrightLeaf's stock price?
# Answer: BrightLeaf's stock price is not provided in the context information.
# Faithfulness Evaluation: 1.0
# Relevancy Result: 1.0

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
#
# No - both queries scored 1.0 on faithfulness and 1.0 on relevancy, which is not what I expected when I picked the stock price question.
#
# The reason is that the pipeline handled the out-of-scope question correctly instead of failing.
# The answer to "What is BrightLeaf's stock price?" was: "BrightLeaf's stock price is not provided in the context information."
# - Faithfulness is 1.0 because the answer makes no claim that the retrieved context does not support. Refusing to answer is perfectly faithful;
#   it is only unfaithful if the model invents a number.
# - Relevancy is 1.0 because the answer does address the question that was asked - it responds to the stock price query by correctly reporting that the
#   information is unavailable. Relevancy asks "does this answer respond to this query", not "is this answer useful".
#
# So this is the important lesson: these two evaluators measure the quality of the *generation step*, not whether retrieval found anything worthwhile.
# A well-behaved refusal scores a perfect 1.0/1.0 on both, exactly like a rich, correct answer does.
# To detect "we have nothing to say about this", I would need a different signal - retrieval metrics like context relevance or hit rate,
# or simply checking the top similarity score against a threshold.
#
# I would only have seen a low score here if the model had hallucinated a stock price (faithfulness would drop)
# or answered a different question than the one asked (relevancy would drop).

# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# - The "LLM-as-a-judge" approach uses LLM to check the quality of responses from another LLM in a RAG system.
# - This approach is used for RAG evaluation, cause it helps with a more holistic and detailed assessment of the responses.
# It is useful when simple metrics can't capture for example a partial correctness or relevance.
