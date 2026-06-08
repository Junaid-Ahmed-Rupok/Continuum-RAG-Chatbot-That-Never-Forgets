import streamlit as st
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Continuum RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1117, #0a0a0f); }
    .header { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #6d28d9, #7c3aed); border-radius: 1rem; margin-bottom: 1.5rem; }
    .header h1 { color: white; margin: 0; }
    .header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; }
    [data-testid="stChatMessage"] { border-radius: 1rem; margin-bottom: 0.5rem; }
    [data-testid="stChatMessage"][data-testid="user"] { background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.2); }
    [data-testid="stChatMessage"][data-testid="assistant"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧠 Continuum RAG</h1>
    <p>Persistent Memory Chatbot · No API Keys · Free Forever</p>
</div>
""", unsafe_allow_html=True)

# Simple file-based memory system (no chromadb needed)
class SimpleMemory:
    def __init__(self):
        self.file_path = Path("./memories.json")
        self.memories = self._load()
    
    def _load(self):
        if self.file_path.exists():
            return json.loads(self.file_path.read_text())
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

# Display chat
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
        
        # Build response
        if memories:
            context = "\n".join([f"• {m['text']}" for m in memories[:3]])
            response = f"""**I remember:**\n{context}\n\n**About your message:** That's interesting! I'll remember this conversation."""
        else:
            response = "I'm listening! Tell me more, and I'll remember our conversation."
        
        st.markdown(response)
        
        # Store in memory
        st.session_state.memory.add(prompt, source="user")
        
        # Extract simple facts
        if "my name is" in prompt.lower():
            import re
            name_match = re.search(r"my name is (\w+)", prompt.lower())
            if name_match:
                name = name_match.group(1).capitalize()
                st.session_state.memory.add(f"User's name is {name}", source="fact")
                st.success(f"📝 I'll remember your name: {name}")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
