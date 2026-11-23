from dotenv import load_dotenv
import os

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
# Step 4. Create embedding model
##########################################################

embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

##########################################################
# Step 5. Build or Load Chroma vector database
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

##########################################################
# Step 6. Retrieve relevant documents
##########################################################

question = "How often should students check their email?"

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_docs = retriever.invoke(question)

print("\nRETRIEVED DOCUMENTS:\n")

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:1000])

##########################################################
# Step 7. Build context
##########################################################

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

##########################################################
# Step 8. Prompt template
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
# Step 9. Format prompt
##########################################################

messages = prompt.format_messages(
    context=context,
    question=question
)

##########################################################
# Step 10. Ask GPT-OSS
##########################################################

response = llm.invoke(messages)

##########################################################
# Step 11. Print answer
##########################################################

print(response.content)