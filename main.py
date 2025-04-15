import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = "AIzaSyDTpDeQpy6Ql6s1Ug6HsZId5jACVPNwJJI"  


app = FastAPI()


pdf_path = r"data\RecipeNLG- Cooking Recipes Dataset.pdf"
db_path = "vector_db/recipes_index"

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if os.path.exists(db_path):
    vectordb = FAISS.load_local(db_path, embeddings=embedding, allow_dangerous_deserialization=True)

else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(db_path)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={"k": 4}), return_source_documents=True)

class RecipeQuery(BaseModel):
    query: str

@app.post("/ask", summary="Ask a recipe question", response_model=dict)
async def ask_recipe(req: RecipeQuery):
    try:
        result = qa_chain.invoke(req.query)
        return {
            "query": req.query,
            "answer": result["result"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
