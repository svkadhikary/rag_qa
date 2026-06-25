import os
import time
import requests
import glob
import pandas as pd
import gradio as gr
from utils import load_and_split_pdf, text_splitter
from TransformerUtils.transformer_utils import transform_sentences, transformer_local
from VectorStores.vectorstore import VectorStore, search
from llms.reranker import cohere_rerank
from llms.model_calls import GroqChatModel



index = None
chunks = None

def user(user_message, history: list):
        if not user_message:
             return "", history
        
        return "", history + [{"role": "user", "content": user_message}]

def my_custom_pdf_function(file_obj):
    """
    This is your custom function where you process the PDF.
    Gradio passes file paths as strings or temporary file objects.
    """
    if file_obj is None:
        return "No file uploaded yet."
    
    # file_obj.name contains the actual local path to the temporary PDF file
    pdf_path = file_obj.name
    print(f"Processing PDF file located at: {pdf_path}")
    
    # PLACE YOUR RAG / PDF PARSING CODE HERE
    # load pdf texts into chunks
    global chunks
    chunks = load_and_split_pdf(pdf_path, chunk_size=400, overlap=50)
    yield f"📄 Loaded and split PDF into {len(chunks)} chunks..."
    # transform chunks into vectors
    vectors = transformer_local(chunks)
    yield f"🔍 Transformed chunks into vector representations..."
    # create vector store index
    global index
    index = VectorStore(vectors.shape[1], chunks[0][:10]).create_index(vectors)
    yield f"📚 Created vector store index for the document: {chunks[0][:10]}..."

    # Return a status message to show the user it worked
    return f"✅ Successfully processed: {pdf_path.split('/')[-1]}..."

def read_urls(urls_string):
    # Split the input string into a list of individual URLs
    urls = [url.strip() for url in urls_string.split(",") if url.strip()]
    
    # Bug Fix: Change string to a list to allow .append()
    temp_text_list = [] 
    
    for url in urls:
        try:
            res = requests.get(url)
            temp_text_list.append(res.text)
        except Exception as e:
            yield f"❌ Error fetching {url}: {str(e)}"
            return

    # Combine text for processing
    full_text = "\n".join(temp_text_list)
    
    # Process chunks
    global chunks
    chunks = text_splitter(full_text, 500, 50)
    
    vectors = transformer_local(chunks)
    yield "🔍 Transformed chunks into vector representations..."
    
    # Create vector store index
    global index
    index = VectorStore(vectors.shape[1], chunks[0][:10]).create_index(vectors)
    
    # Bug Fix: Use yield instead of return at the end of a generator
    yield f"📚 Created vector store index for the document: {chunks[0][:10]}..."
    yield f"✅ Successfully processed text sample: {full_text[:10]}..."



def retrieve_relevant_chunks(query, top_k=10):
     # transform query into vector
     query_vector = transformer_local(query)
     # retrieve relevant chunks from vector store
     global index
     distances, indices = search(query_vector, index, top_k)
     return indices[0]

def respond_to_query(query):
     if index is None:
          return "Please upload a PDF document first."
     
     # get enriched query
     enriched_query = GroqChatModel(temperature=0.5).query_rewrite(query)
     # retrieve relevant indices
     relevant_indices = retrieve_relevant_chunks([enriched_query])
     # retrieve relevant chunks
     global chunks
     relevant_chunks = [chunks[i] for i in relevant_indices if i != -1]
     # rerank chunks
     relevant_chunks = cohere_rerank(enriched_query, relevant_chunks, top_k=5)
     # get llm response
     llm_response = GroqChatModel().generate_responses(enriched_query, ''.join(relevant_chunks))
     
     return llm_response, relevant_chunks

def bot(history: list):
    raw_content = history[-1]['content'][-1].get('text', '') if history and history[-1]['role'] == 'user' else "No user message yet."
    bot_message, retrieved_chunks = respond_to_query(raw_content)

    # Format chunks neatly into one string
    formatted_chunks = "\n\n---\n\n".join([f"📌 [Chunk {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    history.append({
         "role": "assistant",
         "content": "",
         })
    
    for character in bot_message:
        history[-1]['content'] += character
        time.sleep(0.01)
        yield history, formatted_chunks

def load_latest_metrics():
    # 1. Find the latest generated evaluation CSV file
    list_of_files = glob.glob('eval_results/custom_scores_*.csv')
    if not list_of_files:
        return 0.0, 0.0, "no_file"
    latest_file = max(list_of_files, key=os.path.getctime)
    
    # 2. Read with pandas and calculate averages
    df = pd.read_csv(latest_file)
    avg_faithfulness = df['faithfulness'].mean()
    avg_recall = df['context_recall'].mean()
    avg_reasoning = df['reasoning'].values[3]
    print(avg_faithfulness, avg_reasoning, avg_recall)
    return (round(avg_faithfulness, 2), round(avg_recall, 2), avg_reasoning)

with gr.Blocks() as demo:
    gr.Markdown("RAG PDF Demo")

    with gr.Row():
         with gr.Column(scale=1):
              gr.Markdown("### Upload your PDF:")
              # pdf uploader component
              pdf_uploader = gr.File(
                   label="Upload PDF",
                   file_types=['.pdf'],
                   file_count="single"
              )
              # Processing funtion
              upload_status = gr.Textbox(
                   label="Processing Status",
                   value="Waiting for document upload...",
                   interactive=False
              )
              with gr.Row():
                    gr.Markdown("### Paste URL(s) to read from")
              
              with gr.Row():
                    # Input component for the user to type or paste URLs
                    url_input = gr.Textbox(
                         label="Enter URLs (separated by commas)", 
                         placeholder="https://example.com, https://another.com"
                    )
              with gr.Row():
                    # Button to trigger the process
                    submit_btn = gr.Button("Process URLs")
                    
              with gr.Row():
                    # Output component to display the streaming yield status messages
                    status_output = gr.Textbox(label="Status updates", interactive=False)
              with gr.Row():
                   # show recent Evaluation metrics
                   gr.Markdown('### Latest Evaluation Metrics')
                   # Load the latest saved metrics
                   faith, recall_val, reasoning = load_latest_metrics()
                   # Display Components
                   faith_metric = gr.Number(label="Average Faithfullness", value=faith)
                   recall_metric = gr.Number(label="Average Recall", value=recall_val)
                   reasoning_metric = gr.Textbox(label="Sample Reasoning", value=reasoning)



         with gr.Column(scale=2):
              gr.Markdown("### Chat With Document")
              chatbot = gr.Chatbot() # chat interface
              msg = gr.Textbox(placeholder="Ask a question relevant to your document") # user input
              clear = gr.Button("Clear Chat")
              with gr.Accordion("🔍 Last Retrieved Document Chunks", open=False):
                   retrieved_chunks_display = gr.Textbox(
                        label="Retrieved Context Chunks", 
                        placeholder="Chunks utilized for the active prompt answer will render here...",
                        lines=8,
                        interactive=False,
                        show_label=False
                   )
    
    # ---- Event Listeners ----
    
    # trigger 1: when PDF is uploaded, process it
    pdf_uploader.upload(
         fn=my_custom_pdf_function,
         inputs=pdf_uploader,
         outputs=upload_status
    )

    # trigger 2: when urls are given
    submit_btn.click(
         fn=read_urls,
         inputs=url_input,
         outputs=status_output
    )

    # trigger 2: when user submits a message
    msg.submit(
         user,
         [msg, chatbot],
         [msg, chatbot],
         queue=False
    ).then(
         bot,
         chatbot,
         outputs=[chatbot, retrieved_chunks_display]
    )

    clear.click(lambda: None, None, chatbot, queue=False) # clear chat history
        


if __name__ == "__main__":
    demo.launch()