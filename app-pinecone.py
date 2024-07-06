import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="RAG-LNGCHN-LOGS"

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

os.environ["USER_AGENT"]=os.getenv("USER_AGENT")


## CONNECTION TO PINECONE

index_name = "rag-genai"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print(existing_indexes)

index = pc.Index(index_name)

print(index)

# 1. INDEXING

loader = PyPDFLoader("Aetna Health Insurance.pdf")
docs = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)


## 2. RETRIEVAL

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if index.describe_index_stats()['total_vector_count'] == 0:
    docsearch = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)

else:
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = docsearch.as_retriever(search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke("Explain Maternity Benefits")
# print(f"Number of retrieved documents: {len(retrieved_docs)}")


### 3. GENERATION

template = '''Answer the question based only on the following context:
{context}
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{question}")
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Explain Maternity Care Coverage in detail and percentage of coverage")






