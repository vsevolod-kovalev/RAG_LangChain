from keys import OPENAI_key
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# load pdf document
file_path = 'file.pdf'
loader = PyPDFLoader(file_path)
document = loader.load()
print(f"Pages:\t{len(document)}")

# split in slices
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

slices = splitter.split_documents(document)
print(f"Slices:\t{len(slices)}")

# embed the chunks and store them in FAISS
embeddings_instance = OpenAIEmbeddings(openai_api_key=OPENAI_key)
vector_storage = FAISS.from_documents(slices, embeddings_instance)

user_query = 'Do I need to look over the right shoulder?'
print(f"User query:\t{user_query}")


llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_key, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_storage.as_retriever(
        use_mmr=True,
        k=3
    )
)

answer = qa_chain.invoke(user_query)
print(answer['result'])
