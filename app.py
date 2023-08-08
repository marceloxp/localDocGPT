import torch
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from chromadb.config import Settings
from pathlib import Path

ROOT_DIRECTORY = Path().resolve()
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/documents"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id,
                                              load_in_8bit=True,
                                              device_map=1, #'auto'
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True
                                              )
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm

def load_ggml_model(device_type, model_id, model_basename):
    print("Using Llamacpp for GGML quantized models")
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    print(model_path)
    max_ctx_size = 2048
    kwargs = {
        "model_path": model_path,
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
    }
    if device_type.lower() == "mps":
        kwargs["n_gpu_layers"] = 1000
    if device_type.lower() == "cuda":
        kwargs["n_gpu_layers"] = 1000
        kwargs["n_batch"] = max_ctx_size
    return LlamaCpp(**kwargs)

device_type = "cuda"
print(f"Running on: {device_type}")
    
embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device_type})
# load the vectorstore
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()

# load the LLM for generating Natural Language responses. 
# llm = load_model()

model_id="TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
llm = load_ggml_model(device_type, model_id, model_basename)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

query = "Como foram as vendas?"

# Get the answer from the chain
res = qa(query)    
answer, docs = res['result'], res['source_documents']

# Print the result
print("\n\n> Question:")
print(query)
print("\n> Answer:")
print(answer)

# # Print the relevant sources used for the answer
print("----------------------------------SOURCE DOCUMENTS---------------------------")
for document in docs:
    print("\n> " + document.metadata["source"] + ":")
    print(document.page_content)
print("----------------------------------SOURCE DOCUMENTS---------------------------")
