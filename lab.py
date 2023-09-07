import os
from dotenv import load_dotenv

load_dotenv()

#print(PINECONE_API_KEY)
#print(PINECONE_ENV)
#print(os.environ["OPENAI_API_KEY"])

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

import pinecone

def query_pinecone(query:str, index_name:str, text_key:str)->str:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    embeddings = OpenAIEmbeddings()
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )

    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings,text_key=text_key)

    #docs = docsearch.similarity_search(query)
    #print(docs)

    #doc = docsearch.similarity_search_with_score(query)
    #print(doc)

    #found_docs = docsearch.max_marginal_relevance_search(query, k=3, fetch_k=4)
    #for i, doc in enumerate(found_docs):
    #    print(f"{i + 1}.", doc.page_content, "\n")

    retriever = docsearch.as_retriever(search_type="mmr")
    matched_docs = retriever.get_relevant_documents(query)
    context = []
    for i, d in enumerate(matched_docs):
        #print(f"\n## Document {i}\n")
        #print(d.page_content)
        context.append(d.page_content )

    
    return context

def construtPrompt(query:str, contexts: list) -> list:
    limit = 3750
        # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    header = """Answer the question as truthfully as possible using the provided context in traditional Chinese, and if the answer is not contained within the text below, say "這個問題我目前無法回答您，請進一步來電洽訊客服人員."\n\n"""

    return header + prompt

def main()->None:
    query = "MTNet忘記密碼怎麼辦?"
    index_name = "mtnet-faq-index"
    contexts = query_pinecone(query=query,index_name=index_name,text_key="content")
    p = construtPrompt(query=query,contexts=contexts)
    print(p)


if __name__ == "__main__":
    main()