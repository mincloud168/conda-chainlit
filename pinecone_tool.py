import numpy as np
import openai
import pickle
import os
import uuid
import pinecone
from tqdm.auto import tqdm
import datetime
from time import sleep
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders import PyPDFLoader
import re

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"

# Preprocess the document library
new_data = []

def read_csv_upsert_pinecone(fileName: str, index: pinecone.Index) -> None:
    #df = "" #read pdf file  #pd.read_csv(fileName)
    df = pd.read_csv(fileName)
    for i,row in df.iterrows():
        print(row)
        title = row["title"]
        heading = row["heading"]
        content = row["content"]
        uid = uuid.uuid4()
        new_data.append({
            'id':str(uid),
            'title':title,
            'heading':heading,
            'content':content
        })
        """
        title: 'MTNet入口網'
        heading: '帳號申請1' 
        text: 'Q1我忘記MTNet密碼了，應該怎麼辦？忘記密碼，請點登入右上方的「忘記密碼」，然後輸入您在 MTNet 的帳號與當初提供密碼寄送的電子郵件及手機號碼後送出。 如您所填入的資料完全正確，在確認了您的身份後，為便於系統安全控管，將以亂數產生一組驗證碼寄送簡訊給您，您可輸入此驗證碼後設定新的密碼。'
        """

        batch_size = 100  # how many embeddings we create and insert at once

        for i in tqdm(range(0, len(new_data), batch_size)):
            # find end of batch
            i_end = min(len(new_data), i+batch_size)
            meta_batch = new_data[i:i_end]
            # get ids
            ids_batch = [x['id'] for x in meta_batch]
            # get texts to encode
            contents = [x['content'] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=contents, engine=EMBEDDING_MODEL)
                #print(res['data'])
            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=contents, engine=EMBEDDING_MODEL)
                        done = True
                    except Exception as e:
                        print("An error occurred while creating the embedding:")
                        print(str(e))
                        pass
            embeds = [record['embedding'] for record in res['data']]
            # cleanup metadata
            meta_batch = [{
                'title': x['title'],
                'heading': x['heading'],
                'content': x['content']
            } for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            index.upsert(vectors=to_upsert)

# initialize connection (get API key at app.pinecone.io)
def initPinecone(index_name:str, pinecone_api_key:str, pinecone_env:str,  dimension_len:int = 1536) -> pinecone.Index:
    pinecone.init(
        api_key=pinecone_api_key,
        #environment="asia-southeast1-gcp"  # find next to API key  #us-central1-gcp #asia-southeast1-gcp
        environment=pinecone_env
    )
        #environment="us-central1-gcp"  # find next to API key
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
    # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=dimension_len,
            metric='cosine',
            metadata_config={
                'indexed': ['title', 'heading']
            }
        )

    # connect to index
    return pinecone.Index(index_name)



def call_api(project_id:str, pinecone_api_key:str, pinecone_env:str,  openai_api_key:str, csv_file:str)->None:


    #project_id = 'openai-chatapi'

    # connect to index
    # pinecone_api_key = get_file_contents('pinecone_apikey')
    index = initPinecone(project_id, pinecone_api_key, pinecone_env) 

    # view index stats
    print(index.describe_index_stats())

    # init openai
    # openai_api_key = get_file_contents('openai_apikey')
    openai.api_key = openai_api_key

    #read_csv_upsert_pinecone('./MTNetQA1.csv', index)
    read_csv_upsert_pinecone(csv_file, index)

    # view index stats
    print(index.describe_index_stats())

def get_part_paragraph(p:str):
    # Using String Methods
    start_keyword = "案情摘要"
    end_keyword = "肇災原因"

    start_index = p.find(start_keyword)
    end_index = p.find(end_keyword)

    if start_index != -1 and end_index != -1:
        extracted_text = p[start_index:end_index]
        print("Extracted Text using String Methods:")
        print(extracted_text)
        return extracted_text
    else:
        print("Keywords not found.")


# parameters: project_id:str, pinecone_api_key:str, pinecone_api_key:str, openai_api_key:str
def main():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    my_project_id = 'mtnet-faq-index'
    my_csv = './osha_case.csv'
    #my_csv = './pcc.csv'
    call_api(my_project_id,PINECONE_API_KEY,PINECONE_ENV, OPENAI_API_KEY,my_csv)
    # Create Index, 
    """
    loader = PyPDFLoader("./110_labor_cases.pdf")
    pages = loader.load_and_split()
    i=0
    for _ in pages:
        if i > 3: #read page 3 - 10
            content = pages[i].page_content
            #p_summary = get_part_paragraph(content)
            print(content)
        i+=1
        if i > 5:
            break
            """



if __name__ == "__main__":
    main()    
