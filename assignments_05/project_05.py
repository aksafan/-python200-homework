# --- Mini-Project — Job Application Helper ---

# Task 1: Setup and System Prompt

# Load your API key and initialize the client. Then define a get_completion() helper function (as seen in the prompt engineering lesson) that takes a messages list and returns the model's text response:

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# Next, write a system prompt that sets up the model as a job application coach. Be specific: give it a role, a description of who it's helping, and clear behavioral constraints.
# At a minimum, your system prompt should instruct the model to:
# Stay focused on job application materials
# Always remind the user to review and edit its output before submitting anywhere
# Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment
SYSTEM_PROMPT = """
You are a job application coach.
You help users improve their resumes, cover letters, and other job application materials.
You will stay focused on job application materials and provide constructive feedback.
Always remind the user to review and edit your output before submitting it anywhere. 
Acknowledge that you may not know the user's specific industry norms, and that the user should use their own judgment when applying your suggestions.
"""
print(f"SYSTEM_PROMPT: {SYSTEM_PROMPT}")

# Add a comment explaining at least one deliberate choice you made in writing the system prompt and why.
#
# I chose to explicitly instruct the model to stay focused on job application materials and provide constructive feedback because this ensures that the model does not flee into unrelated topics.
# Additionally, highlighting the need for user review and acknowledging potential gaps in industry knowledge helps manage user expectations and encourages critical check and validation of the results.

# Before you move on — check:
# If you print your system prompt and read it aloud, does it sound like a clear briefing for a specific assistant?
# If it's vague or could apply to almost any task, try adding more specificity.
# The more concrete your system prompt, the more predictable and useful the model's behavior will be throughout the project.


# Task 2: Bullet Point Rewriter

# Write a standalone rewrite_bullets() function that takes a list of resume bullet points and returns improved versions.
# This function will later be called from inside the chatbot loop.

# Your function should:

# Use delimiters to clearly separate the user's bullet points from your instructions
# Ask for the output as a JSON list where each item has "original" and "improved" keys
# Parse the JSON response and print both versions of each bullet side by side

import json


def print_bullet_comparison(rewritten: list[dict]) -> None:
    # Print each original bullet next to its improved version, side by side
    if not rewritten:
        print("No bullet points to compare.")
        return

    print("\n--- Bullet Point Comparison ---")
    for i, item in enumerate(rewritten, start=1):
        print(f"{i}. Original: {item['original']}")
        print(f"   Improved: {item['improved']}\n")


def rewrite_bullets(bullets: list[str], display: bool = True) -> list[dict]:
    # Rewrite resume bullets and return a list of {"original", "improved"} dicts.
    # Always returns a list, so callers can use the return value directly.
    # display=True prints the side-by-side comparison; callers that want to
    # control their own presentation (like the chatbot) can pass display=False.
    if not bullets:
        return []

    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with a valid JSON list, no other text and no markdown code fences.
    Each item must have exactly two keys:
    "original" (the original bullet, copied verbatim) and "improved" (your rewritten version).
    Return one item per bullet point, in the same order they were given.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result

    raw = get_completion(messages).strip()
    # Here I'm checking for "```json" part and strip it
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response was: {raw}")
        return []

    # Keep only well-formed items so callers can trust the returned structure
    parsed = [
        item for item in result
        if isinstance(item, dict) and "original" in item and "improved" in item
    ]

    # Print both versions of each bullet side by side
    if display:
        print_bullet_comparison(parsed)

    return parsed

# Test it with these starter bullets:
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewritten_bullets = rewrite_bullets(bullets)
print("Returned parsed JSON list:", rewritten_bullets)

# Bullet Point Rewriter Result: [{'original': 'Helped customers with their problems', 'improved': 'Resolved customer inquiries and issues efficiently, enhancing satisfaction and loyalty.'}, {'original': 'Made reports for the management team', 'improved': 'Compiled and presented detailed analytical reports for the management team, facilitating data-driven decision-making.'}, {'original': 'Worked with a team to finish the project on time', 'improved': 'Collaborated with a cross-functional team to successfully deliver the project ahead of schedule, ensuring alignment with strategic goals.'}]

# Add a comment: What makes these bullets weak, and what kinds of changes did the model suggest?
#
# The original bullets are weak because they are vague, lack specific details, and do not highlight the impact or results of the actions taken.
# They also use generic phrases like "helped customers" and "made reports" without highlighting the achievements or demonstrating the skills involved.
# The model's suggestions improve the bullets by adding specificity (e.g., "Resolved customer inquiries and issues efficiently"), emphasizing results (e.g., "enhancing satisfaction and loyalty"), and using stronger action verbs (e.g., "Compiled and presented detailed analytical reports").
# The improved versions also provide context that makes the candidate's contributions clearer and more compelling to potential employers.

# Before you move on — check:
# Did json.loads() succeed without raising an error? If not, try adding "Respond ONLY with valid JSON, no other text." to your prompt.
# The model sometimes adds a preamble like "Here is the JSON:" that breaks the parser.
# Are both the original and improved versions printing clearly for each bullet?
# Do the improvements feel meaningfully better, or are they just rearranged words? If the output is weak, try making your prompt more specific about what "strong" looks like.



# Task 3: Cover Letter Generator

# Write a generate_cover_letter() function that takes a job title and a brief description of the user's background, and returns a cover letter opening paragraph.
# Use few-shot prompting: include at least two examples of strong cover letter openings in your prompt before asking for the new one.
# Your examples should demonstrate the tone and style you want — confident, specific, and not generic.

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be exactly 3-4 sentences: confident, specific, and free of clichés.
    Do not exceed 4 sentences. Be concise and direct.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete information — which turns out to be excellent training for data analysis. I recently completed a data analytics program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that combination of clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions get made by people who had never processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

    Now write a 3-4 sentence opening paragraph for this person. Do not add extra commentary or explanation.

    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    return get_completion(messages).strip()

# Test it with:

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
# Print the generated paragraph.
print(f"Job title: {job_title}")
print(f"Background: {background}")
print(f"The generated paragraph: {generate_cover_letter(job_title, background)}")
# The generated paragraph: After five years of guiding middle school students through the complexities of math, I’ve developed a knack for breaking down intricate concepts into digestible pieces — a skill that's directly transferable to data engineering.
# Recently, I immersed myself in a Python course, where I built data pipelines using Prefect and Pandas, transforming raw data into actionable insights.
# I'm eager to leverage my teaching experience and newfound technical expertise to contribute to [Company]'s innovative data solutions and drive impactful results.

# Add a comment: Why did you choose those particular examples? What does the few-shot pattern help control in the output?
# I chose those examples because they demonstrate a strong, confident tone while also being specific about the candidate's background and how it relates to the new role.
# Both examples clearly articulate the candidate's unique value proposition and show a clear connection between their past experience and the job they're applying for.

# Before you move on — check:
#
# Does the output feel tailored to the specific person, or is it generic? (Phrases like "I am excited to bring my unique skills..." are a red flag.)
# Does it avoid inventing credentials the user didn't mention?
# Try changing the job title and background to something very different and see if the output adapts. If it sounds the same regardless of input, your prompt may not be specific enough.



# Task 4: Moderation Check

# Before sending any user input to the model in your chatbot loop, run it through OpenAI's moderation endpoint first.
# Write an is_safe(text) function that:

# Calls client.moderations.create() with model="omni-moderation-latest"
# Returns True if the input is not flagged, False if it is
# Prints a short, respectful message if the input is flagged, asking the user to rephrase

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("Your message was flagged by our moderation system. Please rephrase and try again.")
        categories = [category for category, flagged in vars(result.results[0].categories).items() if flagged]
        print(f"Triggered result categories: {categories}")

        return False
    return True

# Test your function with at least two inputs — one that should pass and one that should be flagged — and print the result of each test.
# You want to confirm this is working correctly before wiring it into the loop.
safe_input = "Can you help me improve my resume bullets?"
flagged_input = "I want to write a cover letter that will kill the employer."

print(f"Testing safe input: '{safe_input}'")
print(f"Is it safe? - {is_safe(safe_input)}")
print(f"Testing flagged input: '{flagged_input}'")
print(f"Is it safe? - {is_safe(flagged_input)}")

# Before you move on — check:
#
# Does your flagged test case actually get caught? If not, try a more explicit phrase.
# Does your safe test case pass without triggering any warning?
# What happens if you test a borderline phrase? Look at result.results[0].categories to see which category was triggered.


# Task 5: The Chatbot Loop

# Now assemble everything into a working chatbot. Use the starter code below as your structure — your job is to fill in the marked sections.

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            # Add user's request to conversation history
            messages.append({"role": "user", "content": user_input})

            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewritten = rewrite_bullets(raw_bullets, display=False)

            if not rewritten:
                reply = "Sorry, I couldn't rewrite those bullet points. Please try again."
                print(f"\nJob Application Helper: {reply}\n")
            else:
                # Present the rewritten bullets inside the chatbot interaction
                print("\nJob Application Helper: Here are your rewritten bullets:")
                print_bullet_comparison(rewritten)
                print("!!! Remember to review and edit these before using them in any real application!\n")
                # Build the assistant turn from the returned data itself
                reply = "Here are your rewritten bullets:\n" + "\n".join(
                    f"- {item['original']} -> {item['improved']}" for item in rewritten
                )

            messages.append({"role": "assistant", "content": reply})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            # Add user's request to conversation history
            messages.append({"role": "user", "content": user_input})

            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter = generate_cover_letter(job_title, background)
            reply = f"Here is a draft of your cover letter opening paragraph for {job_title}:\n\n{cover_letter}"
            print(f"\nJob Application Helper: {reply}")
            print("\n!!! Remember to review and edit it before using it in any real application!")

            # Preserve the assistant's actual reply in history
            messages.append({"role": "assistant", "content": reply})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            reply = get_completion(messages)
            # - Print the reply
            print(f"\nJob Application Helper: {reply}\n")
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# Before you move on — check:
#
# Have a short conversation with your bot (3-4 exchanges) without using the bullet or cover letter features.
# After each turn, add a temporary print(len(messages)) to confirm the history is growing. Remove it when you're done.
# Ask the bot something from turn 1, then reference it in turn 3 (e.g., give your name in the first message, then ask "what did I tell you my name was?").
# If it can't remember, check that you're appending both user and assistant messages to messages after every turn.
#
# Try triggering the bullet rewriter and cover letter generator from inside the loop and confirm they still work.
#
# Type quit and confirm the bot exits cleanly.


# Task 6: Ethics Reflection
# Choose one of the following and add a comment at the top of your reflection noting which format you chose:
# Option A — Comment block: At the bottom of project_05.py, add a comment block responding to the questions below. Write at least 3-5 sentences total.
# Option B — Short video: Record a 2-3 minute Loom or YouTube video walking through the same questions and paste the link as a comment at the bottom of project_05.py. This can be submitted as your second LMS link.
# Respond to at least two of the following three questions:
#
# Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
# What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)

# Option A was chosen for this reflection:
#
# QUESTION 1 - BIAS:
# The bot was trained on text written primarily by and about certain demographics, industries, and communication styles.
# For example, it may have learned heavily from tech industry materials, standard American business English, and examples from people with certain educational backgrounds.
# This could produce advice that favors direct, assertive communication over indirect or respectful communication styles valued in many cultures; tech jargon over traditional industries like healthcare or public service; or Western management concepts over practices common in other business contexts.
#
# QUESTION 2 - USER RISK:
# If a job-seeker submitted the bot's output directly without reviewing it, several real harms could occur:
# The bot might suggest jargon or achievements that don't match the job seeker's actual experience, making them appear dishonest during interviews or skills assessments.
# A generated bullet point could inadvertently include phrases inappropriate for the candidate's industry or target company's culture, causing them to be filtered out by human readers.
# The cover letter examples in the training data may not represent successful applications in all fields; a candidate in academia or non-profits might use advice calibrated for tech startups, damaging their credibility with hiring managers who expect different tone and structure.
#
# GUARDRAIL:
# One concrete guardrail I would add is a mandatory review checklist presented to the user before they export or finalize any output from the tool.
# The checklist would ask: "Does this language sound like you and match your actual background?" "Have you verified any numbers, dates, or achievements?" "Does this match the communication style of your target industry?"
# This design choice makes the tool's advisory nature explicit and shifts final responsibility back to the human, who has the judgment and context the AI lacks.
