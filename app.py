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
        Document(page_content="Homeowners Insurance: Essential coverage for physical damage to your house and property liability. Standard home limits start at $250,000 for dwelling coverage and $100,000 for personal liability. Typical bundle addition is $120/month."),
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
class TMG_Guardrails:
    """Deterministic middleware to validate LLM outputs for a regulated industry."""
    @staticmethod
    def validate_policy_bundle(policy_name, price):
        approved_policies = ["Homeowners", "Umbrella", "Renters"]
        if not any(approved in policy_name for approved in approved_policies):
            return False, f"⚠️ GUARDRAIL BLOCKED: '{policy_name}' is not an approved TMG product."
        
        try:
            numeric_price = int(re.sub(r'[^\d]', '', price))
            if numeric_price <= 0 or numeric_price > 500:
                return False, f"⚠️ GUARDRAIL BLOCKED: Hallucinated premium amount ({price}). Manual underwriting required."
        except ValueError:
            return False, "⚠️ GUARDRAIL BLOCKED: Invalid premium format detected."
            
        return True, "Valid"

def process_user_input(user_input):
    docs = vector_db.similarity_search(user_input, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # 1. ENFORCE DETERMINISM: Temperature 0.0 for regulated spaces
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly governed underwriting assistant for The Mutual Group.
        You must ONLY use the provided context to answer. 
        If the user asks about coverage details that are NOT in the context, politely reply: "I don't have those specific details in my current files, so I must refer you to a licensed agent."
        
        {context}
        
        RULES:
        1. Keep answers conversational but professional.
        2. Never hallucinate prices or policies not in the text.
        3. If the user explicitly agrees to add a policy (e.g., "let's do it", "add umbrella"), you MUST append this exact string to the very end of your response: [ADD_POLICY: <Policy Name>|<Price>]
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
    
    # Wire the buttons to update state and instantly reroute to checkout
    if col1.button("🏡 Homeowners", use_container_width=True):
        st.session_state.bundled_policy = "Homeowners"
        st.session_state.bundled_price = "$120"
        st.session_state.step = "checkout"
        st.rerun()
        
    if col2.button("☔ Umbrella", use_container_width=True):
        st.session_state.bundled_policy = "Umbrella"
        st.session_state.bundled_price = "$45"
        st.session_state.step = "checkout"
        st.rerun()
        
    if col3.button("🏢 Renters", use_container_width=True):
        st.session_state.bundled_policy = "Renters"
        st.session_state.bundled_price = "$15"
        st.session_state.step = "checkout"
        st.rerun()
    
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
                    proposed_policy = match.group(1).strip()
                    proposed_price = match.group(2).strip()
                    
                    # RUN DETERMINISTIC GUARDRAIL CHECK
                    is_valid, guardrail_msg = TMG_Guardrails.validate_policy_bundle(proposed_policy, proposed_price)
                    
                    if is_valid:
                        clean_response = raw_response.replace(match.group(0), "").strip()
                        
                        # ESCAPE DOLLAR SIGNS FOR UI RENDERING
                        display_text = clean_response.replace("$", r"\$")
                        st.write(display_text)
                        st.session_state.messages.append({"role": "assistant", "content": display_text})
                        
                        st.session_state.bundled_policy = proposed_policy
                        st.session_state.bundled_price = proposed_price
                        st.session_state.step = "checkout"
                        st.rerun()
                    else:
                        st.error(guardrail_msg)
                        st.session_state.messages.append({"role": "assistant", "content": guardrail_msg})
                else:
                    # ESCAPE DOLLAR SIGNS FOR UI RENDERING
                    display_text = raw_response.replace("$", r"\$")
                    st.write(display_text)
                    st.session_state.messages.append({"role": "assistant", "content": display_text})

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
