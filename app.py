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
    
    /* Fix for chat messages - MAKE TEXT VISIBLE */
    [data-testid="stChatMessage"] {
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.75rem;
    }
    
    /* User message - light purple background, white text */
    [data-testid="stChatMessage"][data-testid="user"] {
        background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
        border: none !important;
        color: white !important;
    }
    
    /* Assistant message - dark gray background, WHITE text */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background: #1a1a2e !important;
        border: 1px solid #3a3a5e !important;
        color: white !important;
    }
    
    /* Make ALL text in chat white */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span {
        color: white !important;
    }
    
    /* Fix markdown bold text in assistant messages */
    [data-testid="stChatMessage"] strong {
        color: #a78bfa !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a1a2e;
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Metrics */
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
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6d28d9, #7c3aed);
        color: white;
        border: none;
        border-radius: 0.5rem;
    }
    
    /* Caption text */
    .stCaption {
        color: #a1a1b0;
    }
    
    /* Success message */
    .stSuccess {
        background: #22c55e;
        color: white;
    }
    
    /* Hide branding */
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

# Simple file-based memory system
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

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Retrieve relevant memories
        memories = st.session_state.memory.retrieve(prompt)
        
        # Check for name memory
        remembered_name = None
        for mem in st.session_state.memory.memories:
            if mem["text"].startswith("User's name is"):
                remembered_name = mem["text"].replace("User's name is ", "")
                break
        
        # Build response
        if remembered_name and "name" in prompt.lower():
            response = f"I remember! Your name is **{remembered_name}**."
        elif memories:
            context = "\n".join([f"• {m['text']}" for m in memories[:3]])
            response = f"**I remember:**\n{context}\n\n**About your message:** I'll remember our conversation."
        else:
            response = "I'm listening! Tell me about yourself, and I'll remember our conversation."
        
        st.markdown(response)
        
        # Store in memory
        st.session_state.memory.add(prompt, source="user")
        
        # Extract simple facts
        if "my name is" in prompt.lower():
            name_match = re.search(r"my name is (\w+)", prompt.lower())
            if name_match:
                name = name_match.group(1).capitalize()
                # Remove old name memory if exists
                st.session_state.memory.memories = [m for m in st.session_state.memory.memories if not m["text"].startswith("User's name is")]
                st.session_state.memory.add(f"User's name is {name}", source="fact")
                st.success(f"📝 I'll remember your name: **{name}**")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
