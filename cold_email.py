from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = os.environ.get("GROQ_API_KEY"),
)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://jobs.nike.com/job/R-38074?from=job%20search%20funnel")
page_data = loader.load().pop().page_content

from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
)

chain_extract = prompt_extract | llm
res = chain_extract.invoke(input={'page_data':page_data})
print(res.content)

from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
res = json_parser.parse(res.content)
print(res)

import pandas as pd

df = pd.read_csv('portfolio.csv')
print(df)


import chromadb
import uuid


client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name='portfolio')

if not collection.count():
    for _,row in df.iterrows():
        collection.add(documents=row['Techstack'],metadatas={'links':row['Links']},ids=[str(uuid.uuid4())])