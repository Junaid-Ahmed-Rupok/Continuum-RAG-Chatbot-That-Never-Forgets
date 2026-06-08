import streamlit as st
import time
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Continuum Teacher", page_icon="📚", layout="wide")

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
    <h1>📚 Continuum Teacher</h1>
    <p>Your Personal AI Teacher · Knowledgeable · No API Keys · Free Forever</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# COMPREHENSIVE KNOWLEDGE BASE
# ============================================================================

class KnowledgeBase:
    """Extensive subject knowledge across multiple domains"""
    
    @staticmethod
    def get_answer(topic, subtopic=None):
        kb = {
            # ========== BANGLADESH ==========
            "bangladesh": {
                "capital": "Dhaka is the capital of Bangladesh. It is one of the most densely populated cities in the world.",
                "population": "Bangladesh has approximately 170 million people, making it the 8th most populous country in the world.",
                "language": "Bengali (Bangla) is the official language of Bangladesh. It is spoken by over 98% of the population.",
                "currency": "Bangladeshi Taka (BDT) is the currency. 1 Taka = 100 poisha.",
                "independence": "Bangladesh gained independence from Pakistan on December 16, 1971 after a 9-month liberation war.",
                "river": "Bangladesh is called the 'Land of Rivers' with over 700 rivers, including the Ganges (Padma), Brahmaputra (Jamuna), and Meghna.",
                "food": "Popular foods include: Hilsa fish (national fish), Biryani, Panta Bhat, Roshogolla, and Chotpoti.",
                "sport": "Cricket is the most popular sport. The national cricket team is known as the Tigers.",
                "history": "Bangladesh was historically part of ancient Bengal, then British India, then East Pakistan, and became independent in 1971.",
                "economy": "Bangladesh has a growing economy driven by the ready-made garment (RMG) industry, remittances, and agriculture.",
                "sundarbans": "The Sundarbans is the largest mangrove forest in the world, home to the Royal Bengal Tiger.",
                "cox_bazar": "Cox's Bazar is the longest natural sea beach in the world, stretching 120 km.",
                "default": "Bangladesh is a South Asian country known for its rich culture, rivers, and the Sundarbans. What specific aspect would you like to learn about?"
            },
            
            # ========== MATHEMATICS ==========
            "math": {
                "pi": "Pi (π) is approximately 3.14159. It is the ratio of a circle's circumference to its diameter.",
                "pythagoras": "The Pythagorean theorem: a² + b² = c², where c is the hypotenuse of a right triangle.",
                "algebra": "Algebra uses letters to represent unknown numbers. Example: x + 5 = 10 → x = 5",
                "calculus": "Calculus is the study of change. It has two branches: Differential (rates of change) and Integral (accumulation).",
                "statistics": "Statistics is the science of collecting, analyzing, and interpreting data.",
                "default": "I can teach you about algebra, geometry, calculus, statistics, or specific formulas. What would you like to learn?"
            },
            
            # ========== SCIENCE ==========
            "science": {
                "physics": "Physics studies matter, energy, and their interactions. Branches include mechanics, thermodynamics, electromagnetism, and quantum physics.",
                "chemistry": "Chemistry studies matter, its properties, how substances combine and change. Key areas: organic, inorganic, physical, and analytical chemistry.",
                "biology": "Biology studies living organisms. Major branches: botany (plants), zoology (animals), microbiology (microorganisms), genetics, and ecology.",
                "gravity": "Gravity is a force that attracts objects with mass. On Earth, acceleration due to gravity is 9.8 m/s².",
                "photosynthesis": "Photosynthesis is how plants make food: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ (carbon dioxide + water → glucose + oxygen).",
                "dna": "DNA (deoxyribonucleic acid) contains genetic instructions. It has a double helix structure discovered by Watson and Crick.",
                "default": "I can teach physics, chemistry, biology, or specific topics like gravity, photosynthesis, or DNA. What interests you?"
            },
            
            # ========== HISTORY ==========
            "history": {
                "world_war_1": "World War I (1914-1918) was a global war centered in Europe. It started after the assassination of Archduke Franz Ferdinand.",
                "world_war_2": "World War II (1939-1945) involved most of the world. Key events: Holocaust, atomic bombs on Hiroshima and Nagasaki.",
                "french_revolution": "The French Revolution (1789-1799) overthrew the monarchy and established a republic. 'Liberty, Equality, Fraternity'.",
                "industrial_revolution": "The Industrial Revolution (1760-1840) transformed manufacturing with machines, steam power, and factories.",
                "default": "I can teach about world wars, revolutions, ancient civilizations, or specific historical periods. What would you like to learn?"
            },
            
            # ========== GEOGRAPHY ==========
            "geography": {
                "continents": "The seven continents are: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
                "oceans": "The five oceans are: Pacific (largest), Atlantic, Indian, Southern, and Arctic (smallest).",
                "mount_everest": "Mount Everest is the highest mountain at 8,848 meters (29,029 feet), located in Nepal/Tibet.",
                "amazon": "The Amazon River is the longest (7,062 km). The Amazon Rainforest is the largest tropical rainforest.",
                "default": "I can teach about continents, oceans, mountains, rivers, countries, or capitals. What would you like to learn?"
            },
            
            # ========== COMPUTER SCIENCE ==========
            "computers": {
                "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
                "ai": "Artificial Intelligence (AI) simulates human intelligence in machines. Subfields include machine learning, deep learning, and NLP.",
                "machine_learning": "Machine Learning enables computers to learn from data without explicit programming. Types: supervised, unsupervised, reinforcement.",
                "algorithm": "An algorithm is a step-by-step procedure for solving a problem. Examples: sorting, searching, pathfinding.",
                "default": "I can teach programming, AI, algorithms, data structures, or computer history. What interests you?"
            },
            
            # ========== LITERATURE ==========
            "literature": {
                "shakespeare": "William Shakespeare (1564-1616) wrote 37 plays and 154 sonnets. Famous works: Hamlet, Romeo and Juliet, Macbeth.",
                "poetry": "Poetry uses aesthetic and rhythmic qualities of language. Types: sonnet, haiku, free verse, epic, lyric.",
                "novel": "A novel is a long fictional narrative. The first novel is often considered 'Don Quixote' by Cervantes (1605).",
                "default": "I can teach about Shakespeare, poetry, novels, famous authors, or literary devices. What would you like to learn?"
            },
            
            # ========== PHILOSOPHY ==========
            "philosophy": {
                "socrates": "Socrates (470-399 BCE) was a Greek philosopher known for the Socratic method of questioning.",
                "plato": "Plato (428-348 BCE) founded the Academy in Athens. Wrote 'The Republic' about justice and ideal society.",
                "aristotle": "Aristotle (384-322 BCE) studied logic, ethics, politics, and biology. Tutored Alexander the Great.",
                "default": "I can teach about Socrates, Plato, Aristotle, Stoicism, Existentialism, or major philosophical ideas."
            }
        }
        
        for domain, content in kb.items():
            if domain in topic.lower():
                if subtopic and subtopic in content:
                    return content[subtopic]
                return content.get("default", f"I can teach you about {domain}. What specific aspect would you like to know?")
        return None

# ============================================================================
# TEACHER BOT
# ============================================================================

class TeacherBot:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.conversation_context = []
    
    def teach(self, user_query):
        query_lower = user_query.lower()
        
        # Subject detection
        subjects = {
            "bangladesh": ["bangladesh", "dhaka", "sundarbans", "cox", "bay of bengal", "padma", "meghna"],
            "math": ["math", "algebra", "geometry", "calculus", "statistics", "pi", "pythagoras"],
            "science": ["science", "physics", "chemistry", "biology", "gravity", "photosynthesis", "dna"],
            "history": ["history", "world war", "revolution", "industrial", "ancient", "medieval"],
            "geography": ["geography", "continent", "ocean", "mountain", "river", "country", "capital"],
            "computers": ["computer", "programming", "python", "ai", "algorithm", "machine learning"],
            "literature": ["literature", "shakespeare", "poetry", "novel", "author", "book"],
            "philosophy": ["philosophy", "socrates", "plato", "aristotle", "stoic", "existential"]
        }
        
        detected_subject = None
        detected_keyword = None
        
        for subject, keywords in subjects.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_subject = subject
                    detected_keyword = keyword
                    break
            if detected_subject:
                break
        
        # Special patterns
        if "who is" in query_lower or "what is" in query_lower or "tell me about" in query_lower:
            # Extract the topic
            patterns = [r"who is (\w+)", r"what is (\w+)", r"tell me about (\w+)"]
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    topic = match.group(1)
                    answer = self.kb.get_answer(topic, topic)
                    if answer:
                        return answer
        
        # Subject-based teaching
        if detected_subject:
            # Extract specific subtopic if any
            words = query_lower.split()
            for word in words:
                answer = self.kb.get_answer(detected_subject, word)
                if answer and word not in detected_keyword:
                    return answer
            return self.kb.get_answer(detected_subject, "default")
        
        # Question patterns
        if query_lower.startswith(("what", "why", "how", "when", "where", "who", "which")):
            return self._handle_general_question(query_lower)
        
        # Greetings
        if any(g in query_lower for g in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm your teacher. What subject would you like to learn today? I can teach Math, Science, History, Geography, Computer Science, Literature, Philosophy, and more!"
        
        # How are you
        if "how are you" in query_lower:
            return "I'm ready to teach! What would you like to learn today?"
        
        # Thanks
        if "thank" in query_lower:
            return "You're welcome! Keep learning. Is there anything else I can teach you?"
        
        # Help
        if "help" in query_lower or "what can you teach" in query_lower:
            return self._get_help()
        
        # Default - offer to teach
        return "I'm your teacher. You can ask me about: Bangladesh, Math, Science, History, Geography, Computer Science, Literature, or Philosophy. What would you like to learn?"
    
    def _handle_general_question(self, query):
        # Try to extract subject from question
        subjects = ["bangladesh", "math", "science", "history", "geography", "computer", "philosophy", "literature"]
        for subject in subjects:
            if subject in query:
                return self.kb.get_answer(subject, "default")
        
        # Try specific keywords
        keywords = ["capital", "population", "river", "ocean", "mountain", "gravity", "photosynthesis", "dna", "python", "ai", "algorithm"]
        for keyword in keywords:
            if keyword in query:
                for subject in subjects:
                    answer = self.kb.get_answer(subject, keyword)
                    if answer:
                        return answer
        
        return "That's a great question! Could you tell me which subject you're asking about? I can teach Math, Science, History, Geography, and more."
    
    def _get_help(self):
        return """**📚 Continuum Teacher - Subjects I Can Teach**

**🇧🇩 Bangladesh Studies**
• History, Independence, Capital, Population, Language, Currency, Rivers, Food, Sports, Sundarbans, Cox's Bazar

**🧮 Mathematics**
• Algebra, Geometry, Calculus, Statistics, Pi, Pythagorean Theorem

**🔬 Science**
• Physics, Chemistry, Biology, Gravity, Photosynthesis, DNA, Evolution

**📖 History**
• World War I & II, French Revolution, Industrial Revolution, Ancient Civilizations

**🌍 Geography**
• Continents, Oceans, Mountains, Rivers, Countries, Capitals

**💻 Computer Science**
• Programming, Python, AI, Machine Learning, Algorithms

**📚 Literature**
• Shakespeare, Poetry, Novels, Famous Authors

**🤔 Philosophy**
• Socrates, Plato, Aristotle, Stoicism, Existentialism

**Try asking:**
• "Tell me about Bangladesh"
• "What is the capital of Bangladesh?"
• "Teach me about gravity"
• "Who was Shakespeare?"
• "What is Python programming?" """

# ============================================================================
# MAIN APP
# ============================================================================

if "teacher" not in st.session_state:
    st.session_state.teacher = TeacherBot()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 📚 Teacher Info")
    st.markdown("I can teach you:")
    st.markdown("• 🇧🇩 Bangladesh Studies")
    st.markdown("• 🧮 Mathematics")
    st.markdown("• 🔬 Science")
    st.markdown("• 📖 History")
    st.markdown("• 🌍 Geography")
    st.markdown("• 💻 Computer Science")
    st.markdown("• 📚 Literature")
    st.markdown("• 🤔 Philosophy")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 Try asking:")
    st.caption("• 'Tell me about Bangladesh'")
    st.caption("• 'What is the capital of Dhaka?'")
    st.caption("• 'Teach me about gravity'")
    st.caption("• 'Who was Shakespeare?'")
    st.caption("• 'What is Python?'")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📚"):
        st.markdown("""Hello! I'm **Continuum Teacher**.

I can teach you about:
- **Bangladesh** (history, capital, culture, rivers, food)
- **Mathematics** (algebra, geometry, calculus)
- **Science** (physics, chemistry, biology)
- **History** (world wars, revolutions)
- **Geography** (continents, oceans, mountains)
- **Computer Science** (programming, AI)
- **Literature** (Shakespeare, poetry)
- **Philosophy** (Socrates, Plato)

**What would you like to learn today?** 📚""")

if prompt := st.chat_input("Ask me anything... I'm your teacher..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="📚"):
        response = st.session_state.teacher.teach(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
