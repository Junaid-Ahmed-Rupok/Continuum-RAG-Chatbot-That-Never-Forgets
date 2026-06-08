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
        cursor: pointer;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124,58,237,0.3);
    }
    
    .stCaption {
        color: #a1a1b0;
    }
    
    .stSuccess {
        background: #22c55e20;
        border-left: 3px solid #22c55e;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧠 Continuum RAG</h1>
    <p>Intelligent Memory Assistant · Precise · No API Keys · Free Forever</p>
</div>
""", unsafe_allow_html=True)

class Memory:
    def __init__(self):
        self.file_path = Path("./memories.json")
        self.data = self._load()
    
    def _load(self):
        if self.file_path.exists():
            try:
                loaded = json.loads(self.file_path.read_text())
                # Ensure required keys exist
                if "facts" not in loaded:
                    loaded["facts"] = {}
                if "conversations" not in loaded:
                    loaded["conversations"] = []
                return loaded
            except:
                return {"facts": {}, "conversations": []}
        return {"facts": {}, "conversations": []}
    
    def _save(self):
        try:
            self.file_path.write_text(json.dumps(self.data, indent=2))
        except:
            pass
    
    def add_fact(self, key, value):
        if not hasattr(self, 'data') or self.data is None:
            self.data = {"facts": {}, "conversations": []}
        self.data["facts"][key] = value
        self._save()
    
    def get_fact(self, key):
        if not hasattr(self, 'data') or self.data is None:
            return None
        return self.data["facts"].get(key)
    
    def get_all_facts(self):
        if not hasattr(self, 'data') or self.data is None:
            return {}
        return self.data["facts"]
    
    def delete_fact(self, key):
        if not hasattr(self, 'data') or self.data is None:
            return
        if key in self.data["facts"]:
            del self.data["facts"][key]
            self._save()
    
    def add_conversation(self, user_msg, bot_msg):
        if not hasattr(self, 'data') or self.data is None:
            self.data = {"facts": {}, "conversations": []}
        self.data["conversations"].append({
            "user": user_msg,
            "bot": bot_msg,
            "timestamp": time.time()
        })
        if len(self.data["conversations"]) > 50:
            self.data["conversations"] = self.data["conversations"][-50:]
        self._save()
    
    def count(self):
        if not hasattr(self, 'data') or self.data is None:
            return 0
        return len(self.data.get("facts", {}))

if "memory" not in st.session_state:
    st.session_state.memory = Memory()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🧠 Memory Dashboard")
    st.metric("Facts Stored", st.session_state.memory.count())
    
    st.markdown("---")
    st.markdown("#### 📌 Stored Facts")
    facts = st.session_state.memory.get_all_facts()
    
    if facts:
        for key, value in facts.items():
            st.markdown(f"**{key}:** {value}")
    else:
        st.caption("*No facts stored yet*")
        st.caption("Try: 'My name is John'")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.memory = Memory()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    st.caption("💡 Tips:")
    st.caption("• 'My name is X'")
    st.caption("• 'I work as X'")
    st.caption("• 'I like X'")
    st.caption("• 'What do you know?'")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def extract_and_store_facts(text, memory):
    lower = text.lower()
    stored = []
    
    # Name extraction
    name_patterns = [
        r"my name is (\w+)",
        r"i(?:'m| am) called (\w+)",
        r"call me (\w+)",
        r"i(?:'m| am) (\w+)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, lower)
        if match:
            name = match.group(1).capitalize()
            if len(name) >= 2 and name.lower() not in ["i", "me", "my", "a"]:
                memory.add_fact("Name", name)
                stored.append(("Name", name))
            break
    
    # Job extraction
    job_patterns = [
        r"i (?:work as|am a|am an) (\w+)",
        r"my job is (\w+)",
        r"i(?:'m| am) (?:a|an) (\w+)"
    ]
    for pattern in job_patterns:
        match = re.search(pattern, lower)
        if match:
            job = match.group(1).capitalize()
            if len(job) >= 2:
                memory.add_fact("Job", job)
                stored.append(("Job", job))
            break
    
    # Hobby extraction
    hobby_patterns = [
        r"i (?:like|love|enjoy) (\w+)",
        r"my hobby is (\w+)",
        r"i(?:'m| am) into (\w+)"
    ]
    for pattern in hobby_patterns:
        match = re.search(pattern, lower)
        if match:
            hobby = match.group(1).capitalize()
            if len(hobby) >= 2:
                memory.add_fact("Hobby", hobby)
                stored.append(("Hobby", hobby))
            break
    
    # Location extraction
    location_patterns = [
        r"i live in (\w+)",
        r"i(?:'m| am) from (\w+)",
        r"from (\w+)"
    ]
    for pattern in location_patterns:
        match = re.search(pattern, lower)
        if match:
            location = match.group(1).capitalize()
            if len(location) >= 2:
                memory.add_fact("Location", location)
                stored.append(("Location", location))
            break
    
    return stored

def generate_intelligent_response(prompt, memory):
    lower = prompt.lower()
    facts = memory.get_all_facts()
    
    # ========== QUESTION HANDLING ==========
    
    # Name questions
    if re.search(r"what('s| is) my name|do you know my name|my name", lower):
        name = memory.get_fact("Name")
        if name:
            return f"Your name is {name}."
        return "I don't know your name yet. Please tell me: 'My name is [your name]'"
    
    # Job questions
    if re.search(r"what('s| is) my job|what do i do for work|what do i do|my job", lower):
        job = memory.get_fact("Job")
        if job:
            return f"You work as a {job}."
        return "I don't know your job yet. Please tell me: 'I work as [your job]'"
    
    # Hobby questions
    if re.search(r"what('s| is) my hobby|what do i like|what do i enjoy|my hobby|what am i into", lower):
        hobby = memory.get_fact("Hobby")
        if hobby:
            return f"You enjoy {hobby}."
        return "I don't know your hobbies yet. Please tell me: 'I like [your hobby]'"
    
    # Location questions
    if re.search(r"where do i live|where am i from|my location|where('s| is) my home", lower):
        location = memory.get_fact("Location")
        if location:
            return f"You live in {location}."
        return "I don't know where you live. Please tell me: 'I live in [city]'"
    
    # General recall
    if re.search(r"what do you (know|remember) about me|what do you know|what do you remember|tell me about me", lower):
        if facts:
            fact_list = []
            for key, value in facts.items():
                if key == "Name":
                    fact_list.append(f"your name is {value}")
                elif key == "Job":
                    fact_list.append(f"you work as a {value}")
                elif key == "Hobby":
                    fact_list.append(f"you enjoy {value}")
                elif key == "Location":
                    fact_list.append(f"you live in {value}")
            
            if fact_list:
                return f"I know that {', '.join(fact_list)}."
            else:
                return "I have some information about you. Ask me specifically about your name, job, hobby, or location."
        return "I don't know anything about you yet. Tell me about yourself."
    
    # ========== STATEMENT HANDLING ==========
    
    # Name statement
    if re.search(r"my name is|i am \w+|i'm \w+", lower) and not re.search(r"how are you", lower):
        name_match = re.search(r"(?:my name is|i am|i'm) (\w+)", lower)
        if name_match:
            name = name_match.group(1).capitalize()
            if len(name) >= 2 and name.lower() not in ["i", "me", "my", "a", "fine", "good", "ok"]:
                memory.add_fact("Name", name)
                return f"I'll remember that your name is {name}."
    
    # Job statement
    if re.search(r"i work as|my job is|i am a|i'm a", lower):
        job_match = re.search(r"(?:i work as|my job is|i am a|i'm a) (\w+)", lower)
        if job_match:
            job = job_match.group(1).capitalize()
            if len(job) >= 2:
                memory.add_fact("Job", job)
                return f"I'll remember that you work as a {job}."
    
    # Hobby statement
    if re.search(r"i (?:like|love|enjoy)", lower):
        hobby_match = re.search(r"i (?:like|love|enjoy) (\w+)", lower)
        if hobby_match:
            hobby = hobby_match.group(1).capitalize()
            if len(hobby) >= 2:
                memory.add_fact("Hobby", hobby)
                return f"I'll remember that you enjoy {hobby}."
    
    # Location statement
    if re.search(r"i live in|i am from", lower):
        location_match = re.search(r"(?:i live in|i am from) (\w+)", lower)
        if location_match:
            location = location_match.group(1).capitalize()
            if len(location) >= 2:
                memory.add_fact("Location", location)
                return f"I'll remember that you live in {location}."
    
    # ========== GENERAL CONVERSATION ==========
    
    # Greetings
    if re.search(r"^(hello|hi|hey|greetings|sup|yo)", lower):
        name = memory.get_fact("Name")
        if name:
            return f"Hello {name}. How can I help you today?"
        return "Hello. How can I help you today?"
    
    # How are you
    if re.search(r"how are you|how's it going|how do you do", lower):
        return "I'm functioning well. What would you like to know?"
    
    # Thanks
    if re.search(r"thank|thanks|appreciate", lower):
        return "You're welcome."
    
    # Goodbye
    if re.search(r"bye|goodbye|see you|exit|quit", lower):
        return "Goodbye. Feel free to return anytime."
    
    # Help
    if re.search(r"help|what can you do|commands|how do i use this", lower):
        return """**Commands:**

**Tell me:**
• "My name is [name]"
• "I work as [job]"  
• "I like [hobby]"
• "I live in [city]"

**Ask me:**
• "What is my name?"
• "What do I do?"
• "What do I like?"
• "Where do I live?"
• "What do you know about me?"

**Example:** "My name is John. I work as a doctor. I like photography." """
    
    # Default response when no pattern matches
    if facts:
        return "I've noted that. Is there anything specific you'd like me to remember or answer?"
    else:
        return "Tell me about yourself. For example: 'My name is John', 'I work as a doctor', or 'I like photography'. I'll remember everything you share."

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Extract and store facts silently
        new_facts = extract_and_store_facts(prompt, st.session_state.memory)
        
        # Generate response
        response = generate_intelligent_response(prompt, st.session_state.memory)
        st.markdown(response)
        
        # Show confirmation for new facts
        for key, value in new_facts:
            st.success(f"✓ Remembered: {key} → {value}")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
