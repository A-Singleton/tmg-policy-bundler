import os
import streamlit as st
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ==========================================
# 0. ENVIRONMENT SECRETS INJECTION
# ==========================================
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ==========================================
# 1. APPLICATION STATE MANAGEMENT
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = "shop"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I see you are setting up an Auto policy. Tell us what other coverage you're looking for so we can bundle it for you easily!"}
    ]
if "bundled_policy" not in st.session_state:
    st.session_state.bundled_policy = None
if "bundled_price" not in st.session_state:
    st.session_state.bundled_price = 0

# ==========================================
# 2. MOCK RAG VECTOR STORE (Built In-Memory)
# ==========================================
@st.cache_resource
def initialize_vector_store():
    docs = [
        Document(page_content="Homeowners Insurance: Essential coverage for physical damage to your house and property liability. Typical bundle addition is $120/month."),
        Document(page_content="Umbrella Policy: Provides an extra $1M-$5M in liability coverage that sits above your auto and home policies. Crucial for high-net-worth individuals to protect assets from lawsuits. Typical bundle addition is $45/month."),
        Document(page_content="Renters Insurance: Covers personal property inside a rented apartment and provides personal liability. Typical bundle addition is $15/month.")
    ]
    
    # SWAPPED TO LOCAL OPEN-SOURCE EMBEDDINGS (Bypasses API errors entirely)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

vector_db = initialize_vector_store()

# ==========================================
# 3. LLM ORCHESTRATION & DECISION ENGINE
# ==========================================
def process_user_input(user_input):
    docs = vector_db.similarity_search(user_input, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # SWAPPED TO UNIVERSALLY AVAILABLE GEMINI 1.0 PRO
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful insurance bundling assistant for The Mutual Group.
        Use the following retrieved policy documents to answer the user's questions:
        
        {context}
        
        RULES:
        1. Keep answers brief and professional.
        2. If the user explicitly agrees to add a policy or says "let's do it" or "add umbrella", you MUST append this exact string to the very end of your response: [ADD_POLICY: <Policy Name>|<Price>]
        Example: [ADD_POLICY: Umbrella|$45]"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": user_input})
    return response.content

# ==========================================
# 4. FRONTEND UI ROUTING
# ==========================================
st.set_page_config(page_title="TMG Policy Bundler", layout="centered")

if st.session_state.step == "shop":
    st.header("🚗 Your Auto Insurance Quote")
    st.write("**Base Policy:** 2024 Sedan - Full Coverage ($150/month)")
    st.divider()
    
    st.subheader("Popular Bundles to Enhance Your Coverage")
    col1, col2, col3 = st.columns(3)
    with col1: st.button("🏡 Homeowners", use_container_width=True)
    with col2: st.button("☔ Umbrella", use_container_width=True)
    with col3: st.button("🏢 Renters", use_container_width=True)
    
    st.divider()
    
    # Chat Interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask about umbrella coverage, home limits, etc..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Checking mutual policies..."):
                raw_response = process_user_input(prompt)
                
                match = re.search(r"\[ADD_POLICY:\s*(.*?)\|(.*?)\]", raw_response)
                
                if match:
                    clean_response = raw_response.replace(match.group(0), "").strip()
                    st.write(clean_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
                    
                    st.session_state.bundled_policy = match.group(1).strip()
                    st.session_state.bundled_price = match.group(2).strip()
                    st.session_state.step = "checkout"
                    st.rerun()
                else:
                    st.write(raw_response)
                    st.session_state.messages.append({"role": "assistant", "content": raw_response})

elif st.session_state.step == "checkout":
    st.success("Bundle Successfully Configured!")
    st.header("📝 Review Your Final Order")
    
    st.write("### Base Auto Policy")
    st.write("**Premium:** $150 / month")
    
    st.write(f"### Bundled Addition: {st.session_state.bundled_policy}")
    st.write(f"**Premium:** {st.session_state.bundled_price} / month")
    
    st.divider()
    st.subheader("Total Monthly Premium: Pending Final Underwriting")
    
    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()
