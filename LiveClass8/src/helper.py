import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from src.prompt import *

_ = load_dotenv(find_dotenv())

groq_api_key = os.environ["GROQ_API_KEY"]

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )


    document_answer_gen = splitter_ans_gen.split_documents( document_ques_gen )

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.5,
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(document_answer_gen, embeddings)

    llm_ans_gen_pipeline = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.5,
    )

    ques_list = ques.split('\n')
    filtered_ques_list = [element for element in ques_list if element.endswith('.') or element.endswith('?')]

    ans_gen_chain = RetrievalQA.from_chain_type(llm = llm_ans_gen_pipeline,
                                                chain_type = "stuff",
                                                retriever=vectorstore.as_retriever())

    return ans_gen_chain, filtered_ques_list