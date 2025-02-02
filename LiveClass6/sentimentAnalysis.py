import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field

_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage
    Only extract the properties mentioned in the 'Classification' function
    passage:
      {input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(...,enum=['happy','nuteral','sad'])
    tendency: str = Field(...,enum=['conservative','liberal','independent'],
                            description="Poltical tendecy of the user")
    language: str = Field(...,enum=['english','spanish'])

llm = ChatGroq(temperature=0).with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

trump_follower = "I'm confident that President Trump's leadership and track record will once again resonate with Americans. His strong stance on economic growth and national security is exactly what our country needs at this pivotal moment. We need to bring back the proven leadership that can make America great again!"

biden_follower = "I believe President Biden's compassionate and steady approach is vital for our nation right now. His commitment to healthcare reform, climate change, and restoring our international alliances is crucial. It's time to continue the progress and ensure a future that benefits all Americans."

response = tagging_chain.invoke({"input": trump_follower})

print("\n----------\n")

print("Sentiment analysis Trump follower (with a list of options using enums):")

print("\n----------\n")
print(response)

print("\n----------\n")

response = tagging_chain.invoke({"input": biden_follower})

print("\n----------\n")

print("Sentiment analysis Biden follower (with a list of options using enums):")

print("\n----------\n")
print(response)

print("\n----------\n")