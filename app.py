import streamlit as st
import time
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

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

class SmartMemory:
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
    
    def add(self, text, source="user", category="general"):
        memory = {
            "id": hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:8],
            "text": text,
            "timestamp": time.time(),
            "source": source,
            "category": category,
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
            word_overlap = len(query_words & mem_words)
            word_score = word_overlap / max(len(query_words), 1)
            
            recency_score = min(1.0, (time.time() - mem["timestamp"]) / 86400) * 0.3
            
            importance_score = min(1.0, mem["access_count"] / 10) * 0.2
            
            total_score = (word_score * 0.6) + (recency_score * 0.2) + (importance_score * 0.2)
            
            if word_score > 0:
                scored.append((total_score, mem))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for score, mem in scored[:top_k]]
    
    def get_fact(self, fact_type):
        for mem in self.memories:
            if mem["text"].startswith(fact_type):
                return mem["text"].replace(fact_type, "")
        return None
    
    def get_all_facts(self):
        facts = {
            "name": self.get_fact("User's name is "),
            "job": self.get_fact("User's job is "),
            "hobby": self.get_fact("User enjoys "),
            "location": self.get_fact("User lives in "),
            "pet": self.get_fact("User has a pet "),
            "food": self.get_fact("User likes food "),
            "movie": self.get_fact("User likes movie "),
            "music": self.get_fact("User likes music "),
            "book": self.get_fact("User likes book "),
            "sport": self.get_fact("User likes sport ")
        }
        return {k: v for k, v in facts.items() if v}
    
    def count(self):
        return len(self.memories)
    
    def update_access(self, memory_id):
        for mem in self.memories:
            if mem["id"] == memory_id:
                mem["access_count"] += 1
                self._save()
                break

if "memory" not in st.session_state:
    st.session_state.memory = SmartMemory()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🧠 Memory Stats")
    st.metric("Total Memories", st.session_state.memory.count())
    
    st.markdown("---")
    st.markdown("#### 📝 What I Remember")
    facts = st.session_state.memory.get_all_facts()
    
    fact_icons = {
        "name": "👤",
        "job": "💼",
        "hobby": "🎨",
        "location": "📍",
        "pet": "🐕",
        "food": "🍕",
        "movie": "🎬",
        "music": "🎵",
        "book": "📖",
        "sport": "⚽"
    }
    
    if facts:
        for fact_type, value in facts.items():
            icon = fact_icons.get(fact_type, "📌")
            st.success(f"{icon} {value}")
    else:
        st.caption("*No facts yet. Tell me about yourself!*")
    
    st.markdown("---")
    if st.button("🗑️ Reset Memory", use_container_width=True):
        st.session_state.memory = SmartMemory()
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🧠 Continuum RAG v3.0")
    st.caption("⚡ Local & Private")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_all_facts(text, memory):
    text_lower = text.lower()
    extracted = []
    
    patterns = {
        "name": [
            (r"my name is (\w+)", "User's name is {}"),
            (r"i(?:'m| am) (\w+)", "User's name is {}"),
            (r"call me (\w+)", "User's name is {}"),
            (r"you can call me (\w+)", "User's name is {}")
        ],
        "job": [
            (r"i (?:work as|am a|am an) (\w+)", "User's job is {}"),
            (r"my job is (\w+)", "User's job is {}"),
            (r"i(?:'m| am) (?:a|an) (\w+)(?: engineer|developer|designer|manager|teacher|doctor|lawyer|writer|artist)", "User's job is {}")
        ],
        "location": [
            (r"i live in (\w+)", "User lives in {}"),
            (r"i(?:'m| am) from (\w+)", "User lives in {}"),
            (r"from (\w+)", "User lives in {}")
        ],
        "hobby": [
            (r"i (?:like|love|enjoy) (\w+)", "User enjoys {}"),
            (r"my hobby is (\w+)", "User enjoys {}"),
            (r"i(?:'m| am) into (\w+)", "User enjoys {}")
        ],
        "pet": [
            (r"i have a (?:dog|cat|bird|fish|hamster|rabbit)(?:\s+called\s+(\w+))?", "User has a pet"),
            (r"my pet is (\w+)", "User has a pet {}"),
            (r"i have (\d+) (?:dogs|cats)", "User has pets")
        ],
        "food": [
            (r"i (?:like|love|enjoy) (?:eating|food|pizza|burger|pasta|sushi|ice cream)", "User likes food"),
            (r"my favorite food is (\w+)", "User likes food {}")
        ],
        "movie": [
            (r"i (?:like|love|enjoy) (\w+) movie", "User likes movie {}"),
            (r"my favorite movie is (\w+)", "User likes movie {}")
        ],
        "music": [
            (r"i (?:like|love|enjoy) (\w+) music", "User likes music {}"),
            (r"my favorite (?:band|artist|singer) is (\w+)", "User likes music {}")
        ]
    }
    
    for category, pattern_list in patterns.items():
        for pattern, template in pattern_list:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1).capitalize() if match.groups() else "yes"
                fact_text = template.format(value) if "{}" in template else template
                
                memory.memories = [m for m in memory.memories if not m["text"].startswith(f"User's {category}")]
                memory.add(fact_text, source="fact", category=category)
                extracted.append((category, value if value != "yes" else fact_text))
                break
    
    return extracted

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        memories = st.session_state.memory.retrieve(prompt)
        
        for mem in memories:
            st.session_state.memory.update_access(mem["id"])
        
        facts = st.session_state.memory.get_all_facts()
        
        remembered_name = facts.get("name")
        remembered_job = facts.get("job")
        remembered_hobby = facts.get("hobby")
        remembered_location = facts.get("location")
        
        lower_prompt = prompt.lower()
        response = ""
        
        question_words = ["what", "who", "where", "when", "why", "how", "do you", "can you", "will you", "are you"]
        is_question = any(prompt.lower().startswith(q) for q in question_words) or "?" in prompt
        
        if is_question:
            if "my name" in lower_prompt or "what's my name" in lower_prompt or "what is my name" in lower_prompt:
                response = f"Your name is **{remembered_name}**!" if remembered_name else "You haven't told me your name yet. What should I call you?"
            
            elif "my job" in lower_prompt or "what do i do" in lower_prompt or "what's my work" in lower_prompt:
                response = f"You work as **{remembered_job}**!" if remembered_job else "You haven't told me about your job yet. What do you do for work?"
            
            elif "where do i live" in lower_prompt or "where am i from" in lower_prompt:
                response = f"You live in **{remembered_location}**!" if remembered_location else "You haven't told me where you live. Want to share?"
            
            elif "hobby" in lower_prompt or "like to do" in lower_prompt or "interests" in lower_prompt:
                response = f"You enjoy **{remembered_hobby}**!" if remembered_hobby else "What do you enjoy doing in your free time?"
            
            elif "what do you remember about me" in lower_prompt or "what do you know about me" in lower_prompt:
                if facts:
                    fact_list = "\n".join([f"- **{k.capitalize()}**: {v}" for k, v in facts.items()])
                    response = f"Here's what I know about you:\n{fact_list}"
                else:
                    response = "I don't know much about you yet. Tell me your name, job, hobbies, or where you live!"
            
            elif "what can you do" in lower_prompt or "how can you help" in lower_prompt:
                response = """I can:
- Remember facts about you (name, job, hobbies, location, pets, etc.)
- Answer questions about what you've told me
- Have natural conversations
- Recall information across sessions

**Try:** "My name is Alex", "I work as a designer", "I love hiking", or "What do you know about me?" """
            
            elif "who are you" in lower_prompt or "what are you" in lower_prompt:
                response = "I'm Continuum, a persistent memory assistant. I remember our conversations so you never have to repeat yourself!"
            
            elif "how are you" in lower_prompt:
                responses = ["I'm doing great, thanks for asking!", "Fantastic! How are you today?", "All good here! What about you?"]
                response = responses[hash(prompt) % len(responses)]
            
            else:
                if remembered_name:
                    response = f"That's a good question, **{remembered_name}**. Could you tell me more about what you're asking?"
                else:
                    response = "That's interesting. Could you tell me more about what you mean?"
        
        elif "my name is" in lower_prompt or "i am" in lower_prompt and len(lower_prompt.split()) < 6:
            name_match = re.search(r"(?:my name is|i am|i'm) (\w+)", lower_prompt)
            if name_match:
                name = name_match.group(1).capitalize()
                response = f"Nice to meet you, **{name}**! I'll remember that. What do you do for work or what are your hobbies?"
        
        elif "i work as" in lower_prompt or "my job is" in lower_prompt or "i am a" in lower_prompt:
            response = f"**{remembered_job}** sounds fascinating! Tell me more about what you enjoy in your free time." if remembered_job else "That's awesome! I'll remember your job. What do you enjoy doing outside work?"
        
        elif "i like" in lower_prompt or "i love" in lower_prompt or "i enjoy" in lower_prompt:
            response = f"**{remembered_hobby}** is wonderful! I'll remember that. Is there anything else you'd like me to know about you?" if remembered_hobby else "That's great! I'll remember your interests. Want to tell me about your job or where you live?"
        
        elif "i live in" in lower_prompt or "i am from" in lower_prompt:
            response = f"**{remembered_location}** is a nice place! I'll remember where you live. What do you do for fun?" if remembered_location else "That's cool! I'll remember your location. What kind of hobbies do you enjoy?"
        
        elif "hello" in lower_prompt or "hi" in lower_prompt or "hey" in lower_prompt:
            if remembered_name:
                responses = [f"Hey **{remembered_name}**! Good to see you again. What's new?", f"Hello **{remembered_name}**! How can I help you today?", f"Hi **{remembered_name}**! Ready to chat?"]
                response = responses[hash(prompt) % len(responses)]
            else:
                response = "Hello! I'm Continuum. Tell me about yourself - your name, job, hobbies - and I'll remember everything!"
        
        elif "thank" in lower_prompt:
            response = "You're very welcome! Happy to help anytime."
        
        elif "bye" in lower_prompt or "goodbye" in lower_prompt:
            if remembered_name:
                response = f"Goodbye **{remembered_name}**! I'll remember our conversation for next time. Take care!"
            else:
                response = "Goodbye! Come back anytime to continue our conversation."
        
        elif "help" in lower_prompt:
            response = """**Continuum Help**

I can remember and recall information about you:

**Tell me:**
• "My name is [name]"
• "I work as [job]"
• "I like [hobby]"
• "I live in [city]"
• "I have a pet [pet]"

**Ask me:**
• "What do you know about me?"
• "What's my name?"
• "What's my job?"
• "Where do I live?"

Try something now!"""
        
        else:
            if remembered_name:
                responses = [
                    f"I hear you, **{remembered_name}**. Tell me more about that.",
                    f"That's interesting, **{remembered_name}**. Can you elaborate?",
                    f"Thanks for sharing, **{remembered_name}**. What else would you like to talk about?"
                ]
                response = responses[hash(prompt) % len(responses)]
            else:
                response = "I'm listening. Tell me about yourself - your name, what you do, or what you enjoy!"
        
        st.markdown(response)
        
        st.session_state.memory.add(prompt, source="user")
        
        extracted_facts = extract_all_facts(prompt, st.session_state.memory)
        
        if extracted_facts:
            for category, value in extracted_facts:
                if category == "name":
                    st.success(f"📝 Nice to meet you, **{value}**! I'll remember your name.")
                elif category == "job":
                    st.success(f"📝 Got it! I'll remember you work as **{value}**.")
                elif category == "hobby":
                    st.success(f"📝 Awesome! I'll remember you enjoy **{value}**.")
                elif category == "location":
                    st.success(f"📝 Cool! I'll remember you live in **{value}**.")
                else:
                    st.success(f"📝 I'll remember that!")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
