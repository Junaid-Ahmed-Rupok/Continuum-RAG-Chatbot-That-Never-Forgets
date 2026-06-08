import streamlit as st
import time
import hashlib
import json
import re
import math
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
    
    .stButton button {
        background: linear-gradient(135deg, #6d28d9, #7c3aed);
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
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
    <h1>📚 Continuum Teacher</h1>
    <p>Your Personal AI Teacher · Answers Questions · No API Keys · Free Forever</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# COMPREHENSIVE KNOWLEDGE BASE
# ============================================================================

class TeacherBot:
    def __init__(self):
        pass
    
    def teach(self, query):
        lower = query.lower().strip()
        
        # ========== MATH PROBLEMS ==========
        
        # Basic arithmetic
        if re.search(r"\d+\s*\+\s*\d+", lower):
            numbers = re.findall(r"\d+", lower)
            if len(numbers) >= 2:
                result = sum(int(n) for n in numbers[:2])
                return f"{numbers[0]} + {numbers[1]} = {result}"
        
        if re.search(r"\d+\s*\-\s*\d+", lower):
            numbers = re.findall(r"\d+", lower)
            if len(numbers) >= 2:
                result = int(numbers[0]) - int(numbers[1])
                return f"{numbers[0]} - {numbers[1]} = {result}"
        
        if re.search(r"\d+\s*\*\s*\d+|\d+\s*x\s*\d+", lower):
            numbers = re.findall(r"\d+", lower)
            if len(numbers) >= 2:
                result = int(numbers[0]) * int(numbers[1])
                return f"{numbers[0]} × {numbers[1]} = {result}"
        
        if re.search(r"\d+\s*/\s*\d+", lower):
            numbers = re.findall(r"\d+", lower)
            if len(numbers) >= 2:
                result = int(numbers[0]) / int(numbers[1])
                return f"{numbers[0]} ÷ {numbers[1]} = {result:.2f}"
        
        # ========== SPECIFIC MATH QUESTIONS ==========
        
        if "square root" in lower or "sqrt" in lower:
            numbers = re.findall(r"\d+", lower)
            if numbers:
                result = math.sqrt(int(numbers[0]))
                return f"The square root of {numbers[0]} is approximately {result:.4f}"
            return "The square root of a number x is a value that when multiplied by itself gives x. For example, √16 = 4."
        
        if "power" in lower or "square" in lower or "cube" in lower:
            numbers = re.findall(r"\d+", lower)
            if "square" in lower and numbers:
                result = int(numbers[0]) ** 2
                return f"{numbers[0]} squared = {result}"
            if "cube" in lower and numbers:
                result = int(numbers[0]) ** 3
                return f"{numbers[0]} cubed = {result}"
            if len(numbers) >= 2:
                result = int(numbers[0]) ** int(numbers[1])
                return f"{numbers[0]} to the power of {numbers[1]} = {result}"
        
        if "percentage" in lower:
            numbers = re.findall(r"\d+", lower)
            if len(numbers) >= 2:
                result = (int(numbers[0]) / int(numbers[1])) * 100
                return f"{numbers[0]} is {result:.2f}% of {numbers[1]}"
            return "Percentage = (Part / Whole) × 100. Example: What is 20% of 50? (20/100) × 50 = 10"
        
        # ========== GENERAL KNOWLEDGE ==========
        
        # Bangladesh
        if "bangladesh" in lower:
            if "capital" in lower:
                return "Dhaka is the capital of Bangladesh."
            if "population" in lower:
                return "Bangladesh has approximately 170 million people."
            if "language" in lower:
                return "Bengali (Bangla) is the official language of Bangladesh."
            if "currency" in lower:
                return "Bangladeshi Taka (BDT) is the currency."
            if "river" in lower:
                return "Bangladesh has over 700 rivers including the Padma (Ganges), Jamuna (Brahmaputra), and Meghna."
            if "food" in lower:
                return "Popular Bangladeshi foods include Hilsa fish, Biryani, Panta Bhat, and Roshogolla."
            if "sundarbans" in lower:
                return "The Sundarbans is the largest mangrove forest in the world, home to the Royal Bengal Tiger."
            if "cox" in lower:
                return "Cox's Bazar is the longest natural sea beach in the world, stretching 120 km."
            if "independence" in lower:
                return "Bangladesh gained independence from Pakistan on December 16, 1971."
            return "Bangladesh is a South Asian country. Ask me about its capital, population, language, rivers, food, or the Sundarbans."
        
        # Science
        if "gravity" in lower:
            return "Gravity is a force that attracts objects with mass. On Earth, acceleration due to gravity is 9.8 m/s². Sir Isaac Newton discovered gravity."
        
        if "photosynthesis" in lower:
            return "Photosynthesis is how plants make food: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ (Carbon dioxide + Water → Glucose + Oxygen)."
        
        if "dna" in lower:
            return "DNA (Deoxyribonucleic Acid) contains genetic instructions. It has a double helix structure discovered by Watson and Crick in 1953."
        
        if "python" in lower and "programming" or "language" in lower:
            return "Python is a high-level programming language created by Guido van Rossum in 1991. It's known for readability and simplicity."
        
        if "ai" in lower or "artificial intelligence" in lower:
            return "Artificial Intelligence (AI) simulates human intelligence in machines. Subfields include machine learning, deep learning, and natural language processing."
        
        # History
        if "shakespeare" in lower:
            return "William Shakespeare (1564-1616) was an English playwright and poet. Famous works: Hamlet, Romeo and Juliet, Macbeth."
        
        if "world war" in lower:
            if "1" in lower or "one" in lower or "first" in lower:
                return "World War I (1914-1918) was a global war centered in Europe. It started after the assassination of Archduke Franz Ferdinand."
            if "2" in lower or "two" in lower or "second" in lower:
                return "World War II (1939-1945) involved most world nations. The Allies (US, UK, USSR) defeated the Axis (Germany, Italy, Japan)."
            return "World War I (1914-1918) and World War II (1939-1945) were major global conflicts. Which one would you like to learn about?"
        
        # Math concepts
        if "pythagoras" in lower or "pythagorean" in lower:
            return "The Pythagorean theorem: a² + b² = c², where c is the hypotenuse of a right triangle."
        
        if "pi" in lower:
            return "Pi (π) is approximately 3.14159. It is the ratio of a circle's circumference to its diameter."
        
        # ========== CALCULATIONS ==========
        
        if re.search(r"\d+", lower) and ("calculate" in lower or "what is" in lower):
            numbers = re.findall(r"\d+", lower)
            if numbers:
                if "plus" in lower or "+" in lower:
                    return f"{numbers[0]} + {numbers[1]} = {int(numbers[0]) + int(numbers[1])}" if len(numbers) >= 2 else f"The number is {numbers[0]}."
                if "minus" in lower or "-" in lower:
                    return f"{numbers[0]} - {numbers[1]} = {int(numbers[0]) - int(numbers[1])}" if len(numbers) >= 2 else f"The number is {numbers[0]}."
                if "times" in lower or "multiply" in lower or "*" in lower or "x" in lower:
                    return f"{numbers[0]} × {numbers[1]} = {int(numbers[0]) * int(numbers[1])}" if len(numbers) >= 2 else f"The number is {numbers[0]}."
                if "divide" in lower or "divided by" in lower or "/" in lower:
                    return f"{numbers[0]} ÷ {numbers[1]} = {int(numbers[0]) / int(numbers[1]):.2f}" if len(numbers) >= 2 else f"The number is {numbers[0]}."
        
        # ========== GREETINGS ==========
        
        if any(g in lower for g in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm your teacher. Ask me any question - math problems, science facts, history, or anything you want to learn!"
        
        if "how are you" in lower:
            return "I'm ready to teach! What would you like to learn today?"
        
        if "thank" in lower:
            return "You're welcome! Keep learning. Ask me anything else."
        
        # ========== HELP ==========
        
        if "help" in lower or "what can you do" in lower or "what can you teach" in lower:
            return """**📚 What I Can Teach You:**

**Math:**
• "2 + 2 = ?" → I'll solve it
• "What is the square root of 16?"
• "Calculate 20% of 50"
• "Explain Pythagoras theorem"

**Bangladesh:**
• "What is the capital of Bangladesh?"
• "Tell me about the Sundarbans"
• "What is the population of Bangladesh?"

**Science:**
• "What is gravity?"
• "Explain photosynthesis"
• "What is DNA?"

**History:**
• "Who was Shakespeare?"
• "Tell me about World War II"
• "When did Bangladesh gain independence?"

**Programming:**
• "What is Python?"
• "Explain AI"

**Just ask me anything! I'm here to teach.** """
        
        # ========== DEFAULT - ANSWER DIRECTLY ==========
        
        # If it's a simple question starting with question words
        if lower.startswith(("what", "why", "how", "when", "where", "who", "which", "can", "could", "would")):
            return "That's a good question. Could you be more specific? For example: 'What is the capital of Bangladesh?' or 'How does gravity work?'"
        
        # Final fallback
        return "I'm your teacher. Ask me a math problem, a question about Bangladesh, science, history, or anything you want to learn!"

# ============================================================================
# MAIN APP
# ============================================================================

if "teacher" not in st.session_state:
    st.session_state.teacher = TeacherBot()
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 📚 Teacher Info")
    st.markdown("**I can answer:**")
    st.markdown("• ✅ Math problems (2+2, √16, 20% of 50)")
    st.markdown("• ✅ Bangladesh questions")
    st.markdown("• ✅ Science facts")
    st.markdown("• ✅ History questions")
    st.markdown("• ✅ Programming concepts")
    
    st.markdown("---")
    st.markdown("**Try these:**")
    st.markdown("• `2 + 2 = ?`")
    st.markdown("• `What is the capital of Bangladesh?`")
    st.markdown("• `What is gravity?`")
    st.markdown("• `Who was Shakespeare?`")
    st.markdown("• `What is Python?`")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📚"):
        st.markdown("""Hello! I'm **Continuum Teacher**.

**I can solve math problems like:**
• 2 + 2 = ?
• √16 = ?
• 20% of 50 = ?

**I can answer questions about:**
• Bangladesh (capital, population, Sundarbans, history)
• Science (gravity, photosynthesis, DNA)
• History (Shakespeare, World Wars)
• Programming (Python, AI)

**Try asking me:**
• "What is 2 + 2?"
• "What is the capital of Bangladesh?"
• "What is gravity?"
• "Who was Shakespeare?"

**Ask me anything!** 📚""")

if prompt := st.chat_input("Ask me a question... I'm your teacher..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="📚"):
        response = st.session_state.teacher.teach(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
