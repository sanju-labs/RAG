import os
import logging
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class RAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initializes the RAG Pipeline and loads environment variables."""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")
        
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def ingest_document(self, file_path: str) -> Chroma:
        """Loads a document, chunks it, and stores it in a vector database."""
        logging.info(f"Loading document from {file_path}...")
        loader = TextLoader(file_path)
        documents = loader.load()

        logging.info("Splitting document into manageable chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks.")

        logging.info("Generating embeddings and storing in ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        return vectorstore

    def create_chain(self, vectorstore: Chroma):
        """Creates the retrieval-augmented generation chain with a custom prompt."""
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3} # Retrieve the top 3 most relevant chunks
        )

        # ==========================================
        # CUSTOM PROMPT DEFINITION
        # ==========================================
        system_prompt = (
            "You are a highly intelligent and helpful assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you cannot find the answer in the provided context, state clearly that you do not know. "
            "Do not make up information. Keep your answer concise and structured.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Helper to format the retrieved documents into a single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Construct the LCEL (LangChain Expression Language) chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

if __name__ == "__main__":
    # --- Setup Dummy Data for Testing ---
    sample_file = "company_policy.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write("Company Policy 2026: Employees are entitled to 25 days of paid time off (PTO) per year. "
                    "Remote work is fully supported. The company IT helpdesk can be reached at support@example.com.")
    
    try:
        # --- Run the Pipeline ---
        logging.info("Initializing RAG Application...")
        app = RAGPipeline()
        
        # 1. Ingest
        db = app.ingest_document(sample_file)
        
        # 2. Build Chain
        chain = app.create_chain(db)
        
        # 3. Query
        question = "How many days of PTO do I get, and who do I contact for IT issues?"
        logging.info(f"Querying: '{question}'\n")
        
        answer = chain.invoke(question)
        
        print("-" * 50)
        print("🤖 AI RESPONSE:")
        print(answer)
        print("-" * 50)

    except Exception as e:
        logging.error(f"Application failed: {e}")