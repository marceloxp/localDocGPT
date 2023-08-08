import torch
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
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

device_type = "cuda"
print(f"Running on: {device_type}")
    
embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device_type})
# load the vectorstore
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()

# load the LLM for generating Natural Language responses. 
llm = load_model()

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
