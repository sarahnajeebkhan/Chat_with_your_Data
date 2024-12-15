from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

pdfs = ['path-to-doc1.pdf','path-to-doc2.pdf','path-to-doc3.pdf']

# Loading the PDFs
pages = []
for pdf in pdfs:
   pages.extend(PyPDFLoader(pdf).load())

# Breaking it into chunks for easier retrieval
r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = r_splitter.split_documents(pages)

# Initializing a collection in Chroma
vectordb = Chroma.from_documents(
    documents = docs,
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory = "/content/chroma/"
)

# Text Generation Model Initialization
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Retrieving context from vectorDB
context = vectordb.max_marginal_relevance_search(question,k=2, fetch_k=3)

# Prompting the model to be an assistant
prompt = f"""
You are a helpful assistant. You use the provided context and your knowledge base to answer the question.
context = {context}
question = {question}
answer =
"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)