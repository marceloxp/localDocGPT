from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import click

from constants import CHROMA_SETTINGS

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

def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    # model_id = "TheBloke/vicuna-7B-1.1-HF"
    model_id = "TheBloke/orca_mini_3B-GGML"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id,
                                            #   load_in_8bit=True, # set these options if your GPU supports them!
                                            #   device_map=1#'auto',
                                            #   torch_dtype=torch.float16,
                                            #   low_cpu_mem_usage=True
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

@click.command()
@click.option('--device_type', default='cpu', help='device to run on, select gpu or cpu')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    else:
        device='cuda'

    print(f"Running on: {device}")
        
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
    embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    
    model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"
    model_id = "TheBloke/orca_mini_3B-GGML"
    llm = load_ggml_model(device_type, model_id, model_basename)

    # model_id = "s3nh/TinyLLama-v0-GGML"
    # model_basename = "TinyLLama-v0.ggmlv3.q8_0.bin"
    # llm = load_ggml_model(device_type, model_id, model_basename)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
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


if __name__ == "__main__":
    main()