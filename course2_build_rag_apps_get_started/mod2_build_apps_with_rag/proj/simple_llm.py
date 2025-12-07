import os
import gradio as gr
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm_model = "google/gemma-3n-e2b-it:free"

# PROMPT = "Who is the current President of the United States?"

chat = ChatOpenAI(
    model_name=llm_model,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0
)

# print(chat.invoke(PROMPT).content)

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = chat.invoke(prompt_txt).content
    return generated_response

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output", lines=10),
    title="Chatbot LLM with Gradio",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)