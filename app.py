import os
import bs4
from dotenv import load_dotenv
from langchain.prompts import load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="RAG-LNGCHN-LOGS"

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# SET USER_AGENT

os.environ["USER_AGENT"]=os.getenv("USER_AGENT")

# 1. INDEXING

# 1.1 Load Web Documents
# loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), 
#                        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
#                        )

# blog_docs = loader.load()

# 1.1 Load PDF's
loader = PyPDFLoader("Aetna Health Insurance.pdf")
blog_docs = loader.load_and_split()


# 1.2 Split Documents - Tokens

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=300, chunk_overlap=50)

splits = text_splitter.split_documents(blog_docs)

## 2. RETRIEVAL

# 2.1 Embed

vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke("What are the approaches decomposition ?")
# len(retrieved_docs)
# print(retrieved_docs)

### 3. GENERATION

### 3.1 Prompt

template = '''Answer the question based only on the following context:
{context}
'''
# prompt_template = hub.pull("rlm/rag-prompt")

# prompt = PromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{question}")
    ]
)

### 3.2 LLM

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

### 3.3 Chain & Invoke

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("How to obtain insurance card ?")


## RESOURCES

## 1. Google AI chat models
## https://python.langchain.com/v0.2/docs/integrations/chat/google_generative_ai/

## 2. Build a Retrieval Augmented Generation (RAG) App
## https://python.langchain.com/v0.2/docs/tutorials/rag/

## 3. Prompt Templates
## https://python.langchain.com/v0.2/docs/concepts/#prompt-templates

## TODOS:
## We can have prompt template to Langchain Hub 