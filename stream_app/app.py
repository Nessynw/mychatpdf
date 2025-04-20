from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
import streamlit as st
from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#import os

# llm = Ollama(model="phi3:instruct")
llm = Ollama(model="mistral:latest")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=80, length_function=len, is_separator_regex=False
)


template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 

    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
"""


custom_rag_prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def save_file_pdf(file) :
    with open(os.path.join("pdfs",file.name),"wb") as f:
        f.write(file.getbuffer())
    return file


def get_text_from_pdf(file) :
    pdf = "pdfs/"+file.name
    loader = PDFMinerLoader(pdf)
    docs = loader.load_and_split()
    return docs


def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
        )   
    chunks = text_splitter.split_documents(docs)
    return chunks


embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
def embed_and_store(chunks) : 
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory="db"
    )
    vector_store.persist()
    return vector_store


def load_vector_store() :
    vector_store = Chroma(
        persist_directory="db",
        embedding_function=embedding
        )
    return vector_store

def get_retriever(vector_store) :
    retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5
                }
            )
    return retriever

# def set_qa_chain(retriever) :
#     qa_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()} 
#             | custom_rag_prompt
#             | llm
#             | StrOutputParser()
#         )  
#     return qa_chain

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
      )
    contexte = get_retriever(vectorstore)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=contexte,
        memory=memory,
        condense_question_prompt=custom_rag_prompt
      )
    return conversation_chain


# def displayPDF(file):
#     # Opening file from file path
#     with open(os.path.join("pdfs",file.name), "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="1000" type="application/pdf"></iframe>'

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="Chat with your PDF",
                       page_icon="chatbot.png")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDF :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        file = st.file_uploader(
            "Upload your PDF here and click on 'Process'", accept_multiple_files=False, type=('pdf'))
        if st.button("Process"):
            pdf = save_file_pdf(file)
            with st.spinner("Processing"):

                raw_text = get_text_from_pdf(pdf)

                # get the text chunks
                text_chunks = split_docs(raw_text)

                # create vector store
                vectorstore = embed_and_store(text_chunks)
            
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        
if __name__ == '__main__':
    main()
    