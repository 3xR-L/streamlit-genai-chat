from dotenv import load_dotenv
import streamlit as st

#Load the env variables
load_dotenv()

# streamlit page setup
st.set_page_config(
    page_title="Optional Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("Generative AI Chatbots")

st.header("Configuración")

temperature = st.slider("Temperatura", 0.0, 1.0, 0.0)

provider = st.selectbox(
    "Proveedor",
    ["Groq", "OpenAI", "Ollama"]
)

if provider == "Groq":
    model_option = st.selectbox(
        "Modelo",
        ["llama-3.1-8b-instant"]
    )

elif provider == "OpenAI":
    model_option = st.selectbox(
        "Modelo",
        ["gpt-5-nano-2025-08-07"]
    )

else:  # Ollama
    model_option = st.selectbox(
        "Modelo",
        ["gemma3:4b "]
    )

if provider == "Groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=model_option,
        temperature=temperature,
    )
elif provider == "OpenAI":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_option,
        temperature=temperature,
    )
else:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=model_option,
        temperature=temperature,
    )

# Initiate chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# show chat history
for message in  st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



st.session_state["provider"] = provider
st.session_state["temperature"] = temperature

user_prompt = st.chat_input("Ask Chatbot...")


if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state["chat_history"].append({"role":"user", "content":user_prompt})

    try:
        response = llm.invoke(
            input=[{"role":"system", "content":"You are a helpful assistant"}, *st.session_state.chat_history]
        )
    except Exception as e:
        st.warning("Hubo un error en la consulta.")

    assistant_response = response.content
    st.session_state.chat_history.append({"role":"assistant", "content":assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)