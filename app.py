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
    <p>Persistent Memory Chatbot · No API Keys · Free Forever</p>
</div>
""", unsafe_allow_html=True)

class SimpleMemory:
    def __init__(self):
        self.file_path = Path("./memories.json")
        self.memories = self._load()
    
    def _load(self):
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text())
            except:
                return []
        return []
    
    def _save(self):
        self.file_path.write_text(json.dumps(self.memories, indent=2))
    
    def add(self, text, source="user"):
        memory = {
            "id": hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:8],
            "text": text,
            "timestamp": time.time(),
            "source": source,
            "access_count": 0
        }
        self.memories.append(memory)
        self._save()
        return memory["id"]
    
    def retrieve(self, query, top_k=5):
        query_words = set(query.lower().split())
        scored = []
        for mem in self.memories:
            mem_words = set(mem["text"].lower().split())
            score = len(query_words & mem_words) / max(len(query_words), 1)
            if score > 0:
                scored.append((score, mem))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for score, mem in scored[:top_k]]
    
    def count(self):
        return len(self.memories)

if "memory" not in st.session_state:
    st.session_state.memory = SimpleMemory()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🧠 Memory Stats")
    st.metric("Total Memories", st.session_state.memory.count())
    
    st.markdown("---")
    st.markdown("#### 🔥 Recent Memories")
    for mem in st.session_state.memory.memories[-5:]:
        st.caption(f"• {mem['text'][:50]}...")
    
    st.markdown("---")
    if st.button("🗑️ Reset Memory", use_container_width=True):
        st.session_state.memory = SimpleMemory()
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🧠 Continuum RAG v1.0")
    st.caption("⚡ Local & Private")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        memories = st.session_state.memory.retrieve(prompt)
        
        remembered_name = None
        for mem in st.session_state.memory.memories:
            if mem["text"].startswith("User's name is"):
                remembered_name = mem["text"].replace("User's name is ", "")
                break
        
        lower_prompt = prompt.lower()
        
        if remembered_name and "my name" in lower_prompt:
            response = f"Your name is **{remembered_name}**! I remember."
        elif "what do you do" in lower_prompt or "what can you do" in lower_prompt or "your purpose" in lower_prompt:
            response = "I'm Continuum, a memory chatbot. I remember our conversations and learn about you. Tell me your name, interests, or anything you want me to remember!"
        elif "who are you" in lower_prompt or "what are you" in lower_prompt:
            response = "I'm Continuum - your persistent memory assistant. I remember what you tell me across conversations. Completely free, no API keys needed!"
        elif "how are you" in lower_prompt:
            response = "I'm doing great! Thanks for asking. How can I help you today?"
        elif "thank" in lower_prompt:
            response = "You're welcome! Glad I could help."
        elif "hello" in lower_prompt or "hi" in lower_prompt or "hey" in lower_prompt:
            response = "Hello! Tell me about yourself - your name, interests, or anything you'd like me to remember."
        elif "your name" in lower_prompt:
            response = "I'm Continuum! Nice to meet you."
        elif "remember" in lower_prompt and remembered_name:
            response = f"I remember that your name is **{remembered_name}**. What else would you like me to know?"
        else:
            response = "Thanks for telling me! I'll remember that. Is there anything specific you'd like to know or share?"
        
        st.markdown(response)
        
        st.session_state.memory.add(prompt, source="user")
        
        if "my name is" in lower_prompt:
            name_match = re.search(r"my name is (\w+)", lower_prompt)
            if name_match:
                name = name_match.group(1).capitalize()
                st.session_state.memory.memories = [m for m in st.session_state.memory.memories if not m["text"].startswith("User's name is")]
                st.session_state.memory.add(f"User's name is {name}", source="fact")
                st.success(f"📝 I'll remember your name: **{name}**")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
