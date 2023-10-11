from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

llm = OpenAI(temperature=0)
# Load the text file
loader = TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/text_files/CODE_OF_CONDUCT.txt")  

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
state_of_union_store = Chroma.from_documents(
    texts, embeddings, collection_name="state-of-union"
)


vectorstore_info = VectorStoreInfo(
    name="imageJ_database",
    description="documentation of imageJ",
    vectorstore=state_of_union_store,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

response = agent_executor.run(
    "what is BioImage.IO for? "
)