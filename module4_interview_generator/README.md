# Module 4 — Interview Question Generator

> This module was built with the help of Claude AI (Anthropic). The implementation was guided and directed by me. Claude wrote the code based on my instructions and decisions.

This module takes a candidate's resume, the job description, and their match result from Module 3 and generates a personalized set of interview questions for that specific candidate. Every candidate gets different questions based on what they have and what they are missing.

The technique used here is RAG (Retrieval Augmented Generation).

## Why RAG

The goal was to make the questions feel targeted and consistent rather than generic. The idea was to not ask the LLM to invent questions from scratch every time. Instead, first retrieve relevant pre-written questions from a database, then ask the LLM to personalize those questions for the specific candidate.

RAG means: retrieve first, then generate.

## How it was built

**Step 1. Created a question bank (question_bank.py)**

* Decided on four categories of questions: technical, experience, gap, and behavioral
* Built 85 questions covering all four categories
* Each question has an id, the question text, a category, skill tags, experience level, and a what to listen for note for the recruiter
* Some questions use placeholders like [SKILL], [COMPANY], [PROJECT] that get replaced with the candidate's real details later
* This file is just data with no libraries and no API calls

**Step 2. Built the index (indexer.py)**

* Chose sentence-transformers with the all-MiniLM-L6-v2 model to convert each question into a vector
* Each question text becomes a list of 384 numbers representing its meaning
* Questions about similar topics produce similar vectors
* Chose ChromaDB as the vector database to store those vectors permanently on disk
* This step runs only once and after that the index lives on disk

**Step 3. Built the retriever (retriever.py)**

* Decided against using a single big search query and instead fire four separate targeted queries, one per category
* Query 1 (technical) is built from the candidate's matched skills
* Query 2 (experience) is built from the candidate's actual company names and project names
* Query 3 (gap) is built from Module 3's missing skills list
* Query 4 (behavioral) is built from the job title and responsibilities
* Each query is embedded into a vector and compared against the stored question vectors in ChromaDB
* Each query only searches within its own category using a metadata filter so gap queries only return gap questions and behavioral queries only return behavioral questions. This is called routing
* Results from all four queries are merged and duplicates are removed
* The output is a pool of around 24 questions

**Step 4. Built the generator (generator.py)**

* The LLM receives the 24 retrieved questions and picks the best 10 to 12
* It replaces every placeholder with the candidate's real details and ensures at least 2 questions come from each category
* Used the same instructor and Groq pattern from Modules 1, 2, and 3 so the output is always structured JSON
* Pydantic enforces the schema and instructor retries automatically if the LLM output does not meet it

**Step 5. Wired everything together (main.py)**

* Single function: generate_interview_questions(candidate, jd, match_result)
* On first call it checks if the index exists and builds it automatically if not
* Then calls the retriever, then the generator, then returns the final result
* That is the only function any outside module needs to call

## Files

| File | What it does |
|---|---|
| question_bank.py | The 85 pre-written questions (data only) |
| indexer.py | Embeds questions and stores vectors in ChromaDB (run once) |
| retriever.py | Runs 4 queries, routes by category, returns a pool of around 24 questions |
| generator.py | Sends the pool and candidate context to the LLM for personalization |
| main.py | Entry point that calls everything in order and returns the final result |

## Libraries used

* sentence-transformers: local embedding model, runs on CPU, no API needed
* chromadb: vector database, stores and searches embeddings on disk
* groq and instructor: LLM call with structured output, same as Modules 1 to 3

## Input and output

Input is three dicts passed to generate_interview_questions():

* Module 1 output (parsed resume)
* Module 2 output (parsed job description)
* Module 3 output (match result for this candidate including matched and missing skills)

Output:

```json
{
  "candidate_name": "Ali Khan",
  "job_title": "Machine Learning Engineer",
  "questions": [
    {
      "category": "technical",
      "question": "You listed TensorFlow. How would you approach migrating an existing model to PyTorch?",
      "what_to_listen_for": "Understanding of both frameworks, migration strategy, awareness of differences"
    },
    {
      "category": "gap",
      "question": "Docker is central to how we deploy here. You have not listed it. Have you worked in a containerized environment at all?",
      "what_to_listen_for": "Honesty about the gap, any adjacent knowledge, learning attitude"
    }
  ]
}
```
