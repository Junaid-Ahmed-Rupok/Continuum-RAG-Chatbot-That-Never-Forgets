import streamlit as st
import time
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Continuum RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1117, #0a0a0f); }
    
    .header { 
        text-align: center; 
        padding: 1.5rem; 
        background: linear-gradient(135deg, #6d28d9, #7c3aed); 
        border-radius: 1rem; 
        margin-bottom: 1.5rem; 
    }
    .header h1 { color: white; margin: 0; }
    .header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; }
    
    [data-testid="stChatMessage"] {
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.75rem;
    }
    
    [data-testid="stChatMessage"][data-testid="user"] {
        background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
        border: none !important;
        color: white !important;
    }
    
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background: #1a1a2e !important;
        border: 1px solid #3a3a5e !important;
        color: white !important;
    }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span {
        color: white !important;
    }
    
    [data-testid="stChatMessage"] strong {
        color: #a78bfa !important;
    }
    
    [data-testid="stSidebar"] {
        background: #1a1a2e;
        color: white;
    }
    
    [data-testid="stMetric"] {
        background: #1a1a2e;
        border: 1px solid #3a3a5e;
        border-radius: 0.75rem;
    }
    
    [data-testid="stMetric"] label {
        color: #a1a1b0;
    }
    
    [data-testid="stMetric"] value {
        color: #a78bfa;
        font-weight: bold;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6d28d9, #7c3aed);
        color: white;
        border: none;
        border-radius: 0.5rem;
    }
    
    .stCaption {
        color: #a1a1b0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧠 Continuum RAG</h1>
    <p>Persistent Memory Assistant · Precise · No API Keys</p>
</div>
""", unsafe_allow_html=True)

class Memory:
    def __init__(self):
        self.file_path = Path("./memories.json")
        self.data = self._load()
    
    def _load(self):
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text())
            except:
                return {"facts": {}, "conversations": []}
        return {"facts": {}, "conversations": []}
    
    def _save(self):
        self.file_path.write_text(json.dumps(self.data, indent=2))
    
    def add_fact(self, key, value):
        self.data["facts"][key] = value
        self._save()
    
    def get_fact(self, key):
        return self.data["facts"].get(key)
    
    def get_all_facts(self):
        return self.data["facts"]
    
    def add_conversation(self, user_msg, bot_msg):
        self.data["conversations"].append({
            "user": user_msg,
            "bot": bot_msg,
            "timestamp": time.time()
        })
        if len(self.data["conversations"]) > 50:
            self.data["conversations"] = self.data["conversations"][-50:]
        self._save()
    
    def count(self):
        return len(self.data["facts"])

if "memory" not in st.session_state:
    st.session_state.memory = Memory()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🧠 Memory")
    st.metric("Facts Stored", st.session_state.memory.count())
    
    st.markdown("---")
    st.markdown("#### 📌 Remembered Facts")
    facts = st.session_state.memory.get_all_facts()
    
    if facts:
        for key, value in facts.items():
            st.caption(f"• {key}: {value}")
    else:
        st.caption("*No facts yet*")
    
    st.markdown("---")
    if st.button("🗑️ Reset Memory", use_container_width=True):
        st.session_state.memory = Memory()
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Continuum v1.0")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def extract_facts(text, memory):
    lower = text.lower()
    updates = []
    
    name_patterns = [
        (r"my name is (\w+)", "Name"),
        (r"i(?:'m| am) (\w+)", "Name"),
        (r"call me (\w+)", "Name")
    ]
    
    job_patterns = [
        (r"i (?:work as|am a|am an) (\w+)", "Job"),
        (r"my job is (\w+)", "Job")
    ]
    
    hobby_patterns = [
        (r"i (?:like|love|enjoy) (\w+)", "Hobby"),
        (r"my hobby is (\w+)", "Hobby")
    ]
    
    location_patterns = [
        (r"i live in (\w+)", "Location"),
        (r"i(?:'m| am) from (\w+)", "Location")
    ]
    
    for pattern, key in name_patterns:
        match = re.search(pattern, lower)
        if match:
            value = match.group(1).capitalize()
            memory.add_fact(key, value)
            updates.append((key, value))
            break
    
    for pattern, key in job_patterns:
        match = re.search(pattern, lower)
        if match:
            value = match.group(1).capitalize()
            memory.add_fact(key, value)
            updates.append((key, value))
            break
    
    for pattern, key in hobby_patterns:
        match = re.search(pattern, lower)
        if match:
            value = match.group(1).capitalize()
            memory.add_fact(key, value)
            updates.append((key, value))
            break
    
    for pattern, key in location_patterns:
        match = re.search(pattern, lower)
        if match:
            value = match.group(1).capitalize()
            memory.add_fact(key, value)
            updates.append((key, value))
            break
    
    return updates

def generate_response(prompt, memory):
    lower = prompt.lower()
    facts = memory.get_all_facts()
    
    # Question patterns with direct answers
    if "what is my name" in lower or "what's my name" in lower or "do you know my name" in lower:
        name = memory.get_fact("Name")
        return f"Your name is {name}." if name else "I don't know your name yet. Tell me: 'My name is [name]'"
    
    if "what is my job" in lower or "what's my job" in lower or "what do i do" in lower:
        job = memory.get_fact("Job")
        return f"You work as {job}." if job else "I don't know your job yet. Tell me: 'I work as [job]'"
    
    if "what is my hobby" in lower or "what's my hobby" in lower or "what do i like" in lower:
        hobby = memory.get_fact("Hobby")
        return f"You enjoy {hobby}." if hobby else "I don't know your hobbies yet. Tell me: 'I like [hobby]'"
    
    if "where do i live" in lower or "where am i from" in lower:
        location = memory.get_fact("Location")
        return f"You live in {location}." if location else "I don't know where you live. Tell me: 'I live in [city]'"
    
    if "what do you know about me" in lower or "what do you remember about me" in lower:
        if facts:
            fact_list = ", ".join([f"{k}: {v}" for k, v in facts.items()])
            return f"I remember: {fact_list}"
        return "I don't know anything about you yet. Tell me your name, job, or hobbies."
    
    # Statement patterns - confirm and store
    if "my name is" in lower:
        return "I'll remember your name."
    
    if "work as" in lower or "my job is" in lower or "am a" in lower:
        return "I'll remember your job."
    
    if "like" in lower or "love" in lower or "enjoy" in lower:
        return "I'll remember that."
    
    if "live in" in lower or "am from" in lower:
        return "I'll remember your location."
    
    # Greetings
    if any(g in lower for g in ["hello", "hi", "hey", "greetings"]):
        name = memory.get_fact("Name")
        return f"Hello {name}." if name else "Hello."
    
    if "how are you" in lower:
        return "I'm functioning well. How can I help?"
    
    if "thank" in lower:
        return "You're welcome."
    
    if "bye" in lower or "goodbye" in lower:
        return "Goodbye."
    
    if "help" in lower or "what can you do" in lower:
        return """Commands:
- Tell me: "My name is X", "I work as X", "I like X", "I live in X"
- Ask: "What's my name?", "What do you know about me?" """
    
    # Default response - short and useful
    if facts:
        return "I've noted that. Anything else you'd like me to remember?"
    else:
        return "Tell me about yourself - your name, job, or hobbies. I'll remember it."

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = generate_response(prompt, st.session_state.memory)
        st.markdown(response)
        
        new_facts = extract_facts(prompt, st.session_state.memory)
        if new_facts:
            for key, value in new_facts:
                st.success(f"✓ Remembered {key.lower()}: {value}")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
