from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify
import time

# Load environment variables
load_dotenv()

api_key1 = os.getenv("GROK_API_KEY")
api_key2 = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize models and embeddings
llm = ChatGroq(groq_api_key=api_key1, model="gemma-7b-it")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
"""
Your role is a ayurvedic doctor bot SwastVeda now what you have to do is analyze the given text given that is delimited by text 'INPUT TEXT' and analyze the Question delimited by 'Question.
    Where format of INPUT TEXT is as: 
      Brief Introduction of the disease, 
      case definition, 
      types of disease with their characterstics, 
      differential diagnosis, 
      3 Levels of that disease: Each level consist of 
        Clinical Diagnosis, 
        Examination, 
        Investigation, 
        Line of treatment 
        Medicines for each level and also medicines according to each types of that disease with proper dosage.
    Now according to the disease that you identified from question provide the user correct ayurvedic medicines,home remedies and yoga poses with proper dosage and timining to consume it. 
    Also provide the do's, dont's and preventions that user must take to recover from the disease.
Context:
{context}

Questions:
{input}

Please ensure your responses are clear, concise, and directly relevant to the questions asked.
"""
)

# Load vectors, embeddings, and documents (run only once or as needed)
def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./files")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# Create the vector store (run when needed)
vectors = vector_embedding()

# Flask route to handle API requests
@app.route('/api/question', methods=['POST'])
def answer_question():
    prompt1 = request.json.get('question')

    if not prompt1:
        return jsonify({'error': 'No question provided'}), 400
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({'input': prompt1})
    elapsed_time = time.process_time() - start
    return jsonify({
        'question': prompt1,
        'answer': response['answer'],
        'processing_time': elapsed_time
    })

if __name__ == '__main__':
    app.run(debug=True)
