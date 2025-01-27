import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv, find_dotenv

_= load_dotenv(find_dotenv())
groq_api_key = os.environ['GROQ_API_KEY']

llm = ChatGroq()

parser = StrOutputParser()

system_template = "Translate the following in the language {language}."

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", '{text}'),
])

chain = prompt_template | llm | parser

app = FastAPI(
    titlle="Simple tranlator",
    description="A simple API server using LangChain's Runnable interfaces.",
    version="0.0.1"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)