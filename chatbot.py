import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

# ------------------ STORAGE ------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def get_user_file(username):
    return os.path.join(DATA_DIR, f"{username}.json")


def load_user_data(username):
    path = get_user_file(username)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return {"conversations": {}, "current_chat": None}


def save_user_data(username, data):
    with open(get_user_file(username), "w") as f:
        json.dump(data, f, indent=2)


# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("Generative AI Chatbots")

# ------------------ USER ------------------
username = "default"

if "user_data" not in st.session_state:
    st.session_state["user_data"] = load_user_data(username)

user_data = st.session_state["user_data"]

def get_existing_users():
    files = os.listdir(DATA_DIR)
    users = [f.replace(".json", "") for f in files if f.endswith(".json")]
    return users

existing_users = get_existing_users()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("👤 Usuario")

    # opción especial para crear nuevo
    user_options = existing_users + ["➕ Crear nuevo usuario"]

    selected_user = st.selectbox("Selecciona usuario", user_options)

    if selected_user == "➕ Crear nuevo usuario":
        new_user = st.text_input("Nuevo usuario")

        if st.button("Crear usuario"):
            if new_user.strip() == "":
                st.warning("Nombre inválido")
            else:
                # crear archivo vacío
                save_user_data(new_user, {
                    "conversations": {},
                    "current_chat": None
                })
                st.success(f"Usuario '{new_user}' creado")
                st.rerun()

        st.stop()

    username = selected_user

with st.sidebar:
    st.header("💬 Conversaciones")

    def create_new_chat():
        chat_id = str(uuid.uuid4())
        user_data["conversations"][chat_id] = {
            "title": "Nuevo chat",
            "messages": []
        }
        user_data["current_chat"] = chat_id
        save_user_data(username, user_data)

    if st.button("➕ Nuevo chat"):
        create_new_chat()
        st.rerun()

    for chat_id, chat in user_data["conversations"].items():
        if st.button(chat["title"], key=chat_id):
            user_data["current_chat"] = chat_id
            save_user_data(username, user_data)
            st.rerun()



if "current_username" not in st.session_state:
    st.session_state["current_username"] = username

if st.session_state["current_username"] != username:
    st.session_state["current_username"] = username
    st.session_state["user_data"] = load_user_data(username)
    st.rerun()

if "user_data" not in st.session_state:
    st.session_state["user_data"] = load_user_data(username)

user_data = st.session_state["user_data"]

# ------------------ CONFIG MODEL ------------------0
st.header("Configuración")

provider = st.selectbox("Proveedor", ["Groq", "OpenAI", "Ollama"])
temperature = st.slider("Temperatura", 0.0, 1.0, 0.0)

if provider == "Groq":
    model_option = st.selectbox("Modelo", ["llama-3.1-8b-instant"])

elif provider == "OpenAI":
    model_option = st.selectbox("Modelo", ["gpt-4o-mini"])

else:
    model_option = st.selectbox("Modelo", ["gemma3:4b"])

# ------------------ CACHE LLM ------------------
@st.cache_resource
def get_llm(provider, model, temperature):
    if provider == "Groq":
        return ChatGroq(model=model, temperature=temperature)

    elif provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=temperature)

    elif provider == "Ollama":
        return ChatOllama(model=model, temperature=temperature)


llm = get_llm(provider, model_option, temperature)

# ------------------ CHAT LOGIC ------------------
if not user_data["current_chat"]:
    # crear uno automáticamente
    chat_id = str(uuid.uuid4())
    user_data["conversations"][chat_id] = {
        "title": "Nuevo chat",
        "messages": []
    }
    user_data["current_chat"] = chat_id
    save_user_data(username, user_data)

chat_id = user_data["current_chat"]
chat = user_data["conversations"][chat_id]
messages = chat["messages"]

# ------------------ SHOW HISTORY ------------------
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------ INPUT ------------------
user_prompt = st.chat_input("Ask Chatbot...")

# ------------------ RESPONSE ------------------
if user_prompt:
    messages.append({"role": "user", "content": user_prompt})

    if len(messages) == 1:
        chat["title"] = user_prompt[:30]

    save_user_data(username, user_data)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(
                [{"role": "system", "content": "You are a helpful assistant"}, *messages]
            ):
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            messages.append({
                "role": "assistant",
                "content": full_response
            })

            save_user_data(username, user_data)

        except Exception as e:
            placeholder.markdown(f"❌ Error: {e}")