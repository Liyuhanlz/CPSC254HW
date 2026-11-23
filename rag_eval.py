from dotenv import load_dotenv
import os
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

##########################################################
# Step 1. Connect to GPT-OSS
##########################################################

llm = ChatOpenAI(
    model="gpt-oss",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ellm.nrp-nautilus.io/v1",
    temperature=0
)

##########################################################
# Step 2. Load syllabus PDF
##########################################################

pdf_path = "CPSC 254_SYL.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

##########################################################
# Step 3. Split text into chunks
##########################################################
# Chunk size = 500
# Chunk overlap = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)

##########################################################
# Step 4. Create embeddings
##########################################################

embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

##########################################################
# Step 5. Load or Create Chroma Database
##########################################################

if os.path.exists("./chroma_db"):
    print("Loading existing Chroma database...")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
else:
    print("Creating new Chroma database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

##########################################################
# Step 6. Prompt Template
##########################################################

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:

"I do not have enough information."

Context:
{context}

Question:
{question}
"""
)

##########################################################
# Step 7. Evaluation Questions
##########################################################

questions = [
    {
        "question": "How often should students check their email?",
        "accuracy": 2,
        "completeness": 2,
        "relevance": 2,
        "justification": "Answer is directly supported by syllabus text."
    },
    {
        "question": "How quickly does the instructor respond to emails?",
        "accuracy": 2,
        "completeness": 2,
        "relevance": 2,
        "justification": "Response time is explicitly stated in the syllabus email communication section."
    },
    {
        "question": "What is the attendance policy?",
        "accuracy": 2,
        "completeness": 2,
        "relevance": 2,
        "justification": "The answer is directly supported by the attendance and participation section of the syllabus."
    }
]

##########################################################
# Step 8. Run Evaluation
##########################################################

results = []

for item in questions:

    question = item["question"]

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    answer = response.content

    unsupported_claims = "None observed"

    print("\n")
    print("=" * 90)
    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSUPPORTING EVIDENCE:")
    print(context)

    print("\nACCURACY:", item["accuracy"])
    print("COMPLETENESS:", item["completeness"])
    print("RELEVANCE:", item["relevance"])

    print("\nJUSTIFICATION:")
    print(item["justification"])

    print("\nUNSUPPORTED CLAIMS / OMISSIONS:")
    print(unsupported_claims)

    results.append({
        "Question": question,
        "Answer": answer,
        "Supporting Evidence": context,
        "Accuracy": item["accuracy"],
        "Completeness": item["completeness"],
        "Relevance": item["relevance"],
        "Justification": item["justification"],
        "Unsupported Claims": unsupported_claims
    })

##########################################################
# Step 9. Save Evaluation Results
##########################################################

df = pd.DataFrame(results)

df.to_csv(
    "rag_evaluation_results.csv",
    index=False
)

print("\n")
print("=" * 90)
print("Evaluation table saved as:")
print("rag_evaluation_results.csv")
print("=" * 90)