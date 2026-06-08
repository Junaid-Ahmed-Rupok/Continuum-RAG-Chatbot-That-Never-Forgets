import streamlit as st
import time
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Continuum RAG - Persistent Memory Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Professional Dark Theme
# ============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 20, 0.95);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
    }
    
    /* User message specific */
    [data-testid="stChatMessage"][data-testid="user"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Input box styling */
    [data-testid="stChatInputTextArea"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 0.75rem;
        color: white;
    }
    
    [data-testid="stChatInputTextArea"]:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 0.75rem;
        padding: 0.75rem;
    }
    
    [data-testid="stMetric"] label {
        color: #a0a0a0;
    }
    
    [data-testid="stMetric"] value {
        color: #667eea;
        font-weight: bold;
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 0.5rem;
        color: #e0e0e0;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Info panel */
    .info-panel {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 0.75rem;
        padding: 0.75rem;
        margin-top: 1rem;
    }
    
    /* Typing indicator */
    .typing-indicator {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 1rem;
        padding: 0.5rem 1rem;
        display: inline-block;
        font-size: 0.85rem;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration"""
    max_tokens: int = 512
    temperature: float = 0.7
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-large"
    chroma_collection: str = "continuum_memory"
    top_k: int = 5
    retention_days: int = 30
    max_memory_size_mb: int = 100
    enable_fact_extraction: bool = True
    enable_decay: bool = True
    decay_rate: float = 0.1

# ============================================================================
# PATHS AND STORAGE
# ============================================================================

class StoragePaths:
    """Centralized path management"""
    
    def __init__(self):
        self.base_path = Path("./continuum_data")
        self.chroma_path = self.base_path / "chromadb"
        self.memories_path = self.base_path / "memories"
        self.export_path = self.base_path / "exports"
        self.logs_path = self.base_path / "logs"
        self.config_path = self.base_path / "config"
        
        for path in [self.base_path, self.chroma_path, self.memories_path, 
                     self.export_path, self.logs_path, self.config_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_memory_file(self, memory_id: str) -> Path:
        return self.memories_path / f"{memory_id}.json"
    
    def get_export_file(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.export_path / f"continuum_export_{timestamp}.json"

paths = StoragePaths()

# ============================================================================
# MEMORY MANAGEMENT SYSTEM
# ============================================================================

class MemoryMetadata:
    """Metadata for each memory entry"""
    
    def __init__(self, text: str, source: str = "conversation"):
        self.id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]
        self.text = text
        self.source = source
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.strength = 1.0
        self.importance_score = 0.5
        self.tags = self._extract_tags(text)
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract simple tags from text"""
        tags = []
        keywords = ["name", "work", "live", "like", "love", "from", "job", "home"]
        for keyword in keywords:
            if keyword in text.lower():
                tags.append(keyword)
        return tags
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "strength": self.strength,
            "importance_score": self.importance_score,
            "tags": self.tags
        }

class AdvancedMemorySystem:
    """Sophisticated memory management with decay and reinforcement"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embed_model, device="cpu")
        self.client = chromadb.PersistentClient(path=str(paths.chroma_path))
        
        try:
            self.collection = self.client.get_collection(config.chroma_collection)
        except:
            self.collection = self.client.create_collection(
                name=config.chroma_collection,
                metadata={"hnsw:space": "cosine", "construction_ef": 200, "M": 64}
            )
        
        self.memory_metadata: Dict[str, MemoryMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load memory metadata from disk"""
        metadata_file = paths.config_path / "memory_metadata.json"
        if metadata_file.exists():
            try:
                data = json.loads(metadata_file.read_text())
                for mem_id, mem_data in data.items():
                    meta = MemoryMetadata("", "")
                    meta.__dict__.update(mem_data)
                    self.memory_metadata[mem_id] = meta
            except Exception as e:
                st.warning(f"Could not load metadata: {e}")
    
    def _save_metadata(self):
        """Save memory metadata to disk"""
        metadata_file = paths.config_path / "memory_metadata.json"
        data = {mid: meta.to_dict() for mid, meta in self.memory_metadata.items()}
        metadata_file.write_text(json.dumps(data, indent=2))
    
    def add_memory(self, text: str, source: str = "conversation") -> str:
        """Add a new memory with full metadata"""
        metadata = MemoryMetadata(text, source)
        embedding = self.embedder.encode(text).tolist()
        
        try:
            self.collection.add(
                ids=[metadata.id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "created_at": metadata.created_at,
                    "source": source,
                    "strength": metadata.strength
                }]
            )
            self.memory_metadata[metadata.id] = metadata
            self._save_metadata()
            return metadata.id
        except Exception as e:
            st.error(f"Failed to add memory: {e}")
            return ""
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant memories with metadata"""
        if self.collection.count() == 0:
            return []
        
        top_k = top_k or self.config.top_k
        query_embedding = self.embedder.encode(query).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            return []
        
        memories = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1.0 - dist
            metadata = self.memory_metadata.get(doc_id)
            
            memories.append({
                "id": doc_id,
                "text": doc,
                "similarity": similarity,
                "strength": metadata.strength if metadata else 0.5,
                "access_count": metadata.access_count if metadata else 0,
                "created_at": metadata.created_at if metadata else time.time()
            })
            
            # Update access metrics
            if metadata:
                metadata.last_accessed = time.time()
                metadata.access_count += 1
                metadata.strength = min(1.0, metadata.strength + 0.05)
                self._save_metadata()
        
        # Sort by relevance and strength
        memories.sort(key=lambda x: (x["similarity"], x["strength"]), reverse=True)
        return memories[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "avg_strength": 0, "active": 0, "size_kb": 0}
        
        strengths = [meta.strength for meta in self.memory_metadata.values()]
        avg_strength = np.mean(strengths) if strengths else 0
        
        # Calculate storage size
        size_bytes = sum(f.stat().st_size for f in paths.chroma_path.rglob("*") if f.is_file())
        size_kb = size_bytes / 1024
        
        return {
            "total": total,
            "avg_strength": round(avg_strength, 3),
            "active": sum(1 for s in strengths if s > 0.3),
            "size_kb": round(size_kb, 2),
            "unique_tags": len(set(tag for meta in self.memory_metadata.values() for tag in meta.tags))
        }
    
    def get_top_memories(self, n: int = 10) -> List[Dict]:
        """Get top memories by strength and access count"""
        if not self.memory_metadata:
            return []
        
        sorted_memories = sorted(
            self.memory_metadata.values(),
            key=lambda x: (x.strength, x.access_count),
            reverse=True
        )
        
        return [{"text": m.text[:80], "strength": m.strength, "access_count": m.access_count} 
                for m in sorted_memories[:n]]
    
    def apply_decay(self):
        """Apply memory decay based on time and access frequency"""
        if not self.config.enable_decay:
            return
        
        current_time = time.time()
        decayed_count = 0
        
        for mem_id, metadata in list(self.memory_metadata.items()):
            days_since_access = (current_time - metadata.last_accessed) / 86400
            decay_factor = math.exp(-self.config.decay_rate * days_since_access)
            metadata.strength *= decay_factor
            
            if metadata.strength < 0.05:
                # Remove weak memories
                try:
                    self.collection.delete(ids=[mem_id])
                    del self.memory_metadata[mem_id]
                    decayed_count += 1
                except:
                    pass
        
        if decayed_count > 0:
            self._save_metadata()
    
    def export_all(self) -> Path:
        """Export all memories to JSON"""
        export_data = {
            "export_timestamp": time.time(),
            "export_date": datetime.now().isoformat(),
            "total_memories": self.collection.count(),
            "config": {
                "embed_model": self.config.embed_model,
                "collection": self.config.chroma_collection
            },
            "memories": []
        }
        
        all_mem = self.collection.get(include=["documents", "metadatas"])
        for mem_id, doc, meta in zip(all_mem["ids"], all_mem["documents"], all_mem["metadatas"]):
            metadata = self.memory_metadata.get(mem_id)
            export_data["memories"].append({
                "id": mem_id,
                "text": doc,
                "created_at": meta.get("created_at"),
                "metadata": metadata.to_dict() if metadata else {}
            })
        
        export_file = paths.get_export_file()
        export_file.write_text(json.dumps(export_data, indent=2))
        return export_file
    
    def reset_all(self):
        """Complete memory wipe"""
        try:
            self.client.delete_collection(self.config.chroma_collection)
            self.collection = self.client.create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )
            self.memory_metadata.clear()
            self._save_metadata()
        except Exception as e:
            st.error(f"Reset failed: {e}")

# ============================================================================
# INTELLIGENT RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """Handles LLM interactions with context management"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
        
        @st.cache_resource
        def load_model():
            return pipeline(
                "text2text-generation",
                model=config.llm_model,
                device=-1,
                model_kwargs={"torch_dtype": "float32"}
            )
        
        self.model = load_model()
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def build_context(self, query: str, memories: List[Dict]) -> str:
        """Build intelligent context from memories"""
        if not memories:
            return query
        
        context_parts = ["Relevant information from my memory:"]
        for i, mem in enumerate(memories[:3], 1):
            context_parts.append(f"{i}. {mem['text']}")
        
        context = "\n".join(context_parts)
        return f"{context}\n\nUser question: {query}"
    
    def generate_response(self, query: str, memories: List[Dict]) -> str:
        """Generate a thoughtful response using context"""
        try:
            context = self.build_context(query, memories)
            
            prompt = f"""You are Continuum, a warm, helpful AI assistant with persistent memory.
You remember past conversations and use that information naturally.
Be concise, friendly, and conversational.

{context}

Your response:"""
            
            result = self.model(
                prompt,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            response = result[0]['generated_text'].strip()
            
            if not response or len(response) < 5:
                response = "I understand. Could you tell me more about that?"
            
            return response
            
        except Exception as e:
            return f"I appreciate you sharing that. Let me think about what you said."

# ============================================================================
# SMART FACT EXTRACTION
# ============================================================================

class FactExtractor:
    """Intelligent fact extraction from conversations"""
    
    PATTERNS = {
        "name": [
            r"(?:my|our) name is (\w+)",
            r"i(?:'m| am) (\w+)",
            r"call me (\w+)",
            r"you can call me (\w+)"
        ],
        "location": [
            r"i (?:live|stay|reside) (?:in|at) (.+?)(?:\.|,|$)",
            r"i(?:'m| am) from (.+?)(?:\.|,|$)",
            r"from (.+?)(?:\.|,|$)"
        ],
        "work": [
            r"i (?:work|job|employed) (?:at|for|as) (.+?)(?:\.|,|$)",
            r"i(?:'m| am) (?:a|an) (.+?) (?:at|for|in|and|\.)",
            r"my job is (.+?)(?:\.|,|$)"
        ],
        "interest": [
            r"i (?:like|love|enjoy|prefer) (.+?)(?:\.|,|$)",
            r"(?:my|our) (?:hobby|passion|interest) is (.+?)(?:\.|,|$)"
        ],
        "preference": [
            r"i (?:prefer|favor|like) (.+?)(?:\.|,|$)",
            r"(?:my|our) favorite (.+?) is (.+?)(?:\.|,|$)"
        ]
    }
    
    @classmethod
    def extract_facts(cls, user_message: str, bot_response: str) -> List[Dict[str, Any]]:
        """Extract structured facts from conversation"""
        combined = f"{user_message} {bot_response}".lower()
        facts = []
        
        for category, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, combined, re.IGNORECASE)
                for match in matches:
                    fact_text = match if isinstance(match, str) else match[0]
                    fact_text = fact_text.strip().capitalize()
                    
                    if len(fact_text) > 3 and len(fact_text) < 100:
                        facts.append({
                            "category": category,
                            "value": fact_text,
                            "statement": cls._format_fact(category, fact_text)
                        })
        
        # Remove duplicates
        unique_facts = []
        seen = set()
        for fact in facts:
            if fact["statement"] not in seen:
                seen.add(fact["statement"])
                unique_facts.append(fact)
        
        return unique_facts[:5]
    
    @classmethod
    def _format_fact(cls, category: str, value: str) -> str:
        """Format fact into natural language"""
        templates = {
            "name": f"User's name is {value}",
            "location": f"User lives in {value}",
            "work": f"User works at/with {value}",
            "interest": f"User enjoys {value}",
            "preference": f"User prefers {value}"
        }
        return templates.get(category, f"User mentioned: {value}")

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Continuum RAG</h1>
            <p>Persistent Memory Chatbot with Intelligent Recall • No API Keys • Free Forever</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_stats(memory_system: AdvancedMemorySystem):
        with st.sidebar:
            st.markdown("### 📊 Memory Statistics")
            
            stats = memory_system.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Memories", stats["total"])
                st.metric("Active Memories", stats["active"])
            with col2:
                st.metric("Avg Strength", f"{stats['avg_strength']:.2f}")
                st.metric("Storage", f"{stats['size_kb']:.0f} KB")
            
            if stats["total"] > 0:
                st.markdown("---")
                st.markdown("### 🔥 Top Memories")
                top_memories = memory_system.get_top_memories(5)
                for mem in top_memories:
                    strength_color = "🟢" if mem["strength"] > 0.6 else "🟡" if mem["strength"] > 0.3 else "🔴"
                    st.caption(f"{strength_color} {mem['text'][:50]}...")
                    st.caption(f"   Strength: {mem['strength']:.2f} | Access: {mem['access_count']}")
            
            st.markdown("---")
            st.markdown("### 🛠️ Controls")
    
    @staticmethod
    def render_settings():
        with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
            top_k = st.slider("Memory Retrieval (Top-K)", 1, 10, 5, 
                             help="Number of memories to retrieve for context")
            temperature = st.slider("Response Creativity", 0.0, 1.5, 0.7, 0.05,
                                   help="Higher = more creative, Lower = more focused")
            enable_decay = st.checkbox("Enable Memory Decay", True,
                                      help="Memories fade naturally over time")
            
            return {
                "top_k": top_k,
                "temperature": temperature,
                "enable_decay": enable_decay
            }
    
    @staticmethod
    def render_action_buttons():
        col1, col2, col3 = st.columns(3)
        with col1:
            export_clicked = st.button("💾 Export Memories", use_container_width=True)
        with col2:
            reset_clicked = st.button("🗑️ Reset Memory", use_container_width=True)
        with col3:
            clear_clicked = st.button("💬 Clear Chat", use_container_width=True)
        
        return export_clicked, reset_clicked, clear_clicked
    
    @staticmethod
    def render_typing_indicator():
        return st.markdown('<div class="typing-indicator">🧠 Continuum is thinking...</div>', 
                          unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def initialize_session():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.config = AppConfig()
        st.session_state.memory_system = AdvancedMemorySystem(st.session_state.config)
        st.session_state.response_generator = ResponseGenerator(st.session_state.config)
        st.session_state.messages = []
        st.session_state.fact_extractor = FactExtractor()
        st.session_state.initialized = True
        st.session_state.last_decay_time = time.time()

def process_message(message: str, memory_system: AdvancedMemorySystem, 
                    response_gen: ResponseGenerator, settings: dict) -> tuple:
    """Process user message and generate response"""
    
    # Apply memory decay periodically
    if time.time() - st.session_state.last_decay_time > 3600:
        memory_system.apply_decay()
        st.session_state.last_decay_time = time.time()
    
    # Retrieve relevant memories
    memories = memory_system.retrieve(message, top_k=settings.get("top_k", 5))
    
    # Generate response
    response = response_gen.generate_response(message, memories)
    
    # Extract and store facts
    if st.session_state.config.enable_fact_extraction:
        facts = FactExtractor.extract_facts(message, response)
        for fact in facts:
            memory_system.add_memory(fact["statement"], source="extracted_fact")
    
    # Update conversation history
    response_gen.add_to_history("user", message)
    response_gen.add_to_history("assistant", response)
    
    return response, memories

# ============================================================================
# MAIN APP RENDER
# ============================================================================

def main():
    initialize_session()
    
    # Render UI
    UIComponents.render_header()
    
    # Sidebar
    UIComponents.render_sidebar_stats(st.session_state.memory_system)
    settings = UIComponents.render_settings()
    export_clicked, reset_clicked, clear_clicked = UIComponents.render_action_buttons()
    
    # Handle actions
    if export_clicked:
        with st.spinner("Exporting memories..."):
            export_file = st.session_state.memory_system.export_all()
            st.success(f"✅ Exported to {export_file}")
    
    if reset_clicked:
        with st.spinner("Resetting memory system..."):
            st.session_state.memory_system.reset_all()
            st.session_state.response_generator.clear_history()
            st.session_state.messages = []
            st.success("🧠 Memory system has been reset!")
            time.sleep(1)
            st.rerun()
    
    if clear_clicked:
        st.session_state.response_generator.clear_history()
        st.session_state.messages = []
        st.rerun()
    
    # Display chat interface
    st.markdown("### 💬 Conversation")
    
    # Show welcome message if no messages
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown("""Hello! I'm **Continuum**, your persistent memory assistant.

I remember our conversations and learn about you over time. Try telling me:

- *"My name is Sarah"*
- *"I work as a software engineer"*
- *"I love hiking and photography"*

Then ask me *"What do you know about me?"* and watch me remember! ✨""")
    
    # Display chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🧠"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("💭 Thinking..."):
                response, memories = process_message(
                    prompt,
                    st.session_state.memory_system,
                    st.session_state.response_generator,
                    settings
                )
                
                # Show retrieved memories in expander
                if memories:
                    with st.expander(f"🔍 Retrieved {len(memories)} memories", expanded=False):
                        for mem in memories:
                            strength_pct = int(mem['strength'] * 100)
                            st.caption(f"📝 {mem['text'][:100]}...")
                            st.progress(strength_pct / 100, text=f"Strength: {strength_pct}%")
                
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update UI
        st.rerun()

if __name__ == "__main__":
    main()
