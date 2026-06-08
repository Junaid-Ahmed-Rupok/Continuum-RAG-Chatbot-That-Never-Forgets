import streamlit as st
import time
import json
import math
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

class MemoryStrength(Enum):
    """Memory strength levels for UI visualization"""
    STRONG = (0.7, "#22c55e")      # Green
    MODERATE = (0.4, "#eab308")    # Yellow
    WEAK = (0.0, "#ef4444")        # Red

@dataclass
class AppConfig:
    """Centralized application configuration"""
    # Model settings
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-large"
    
    # Memory settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 5
    min_strength: float = 0.05
    decay_lambda: float = 0.1
    reinforcement_boost: float = 0.3
    
    # UI settings
    max_conversation_turns: int = 20
    max_display_facts: int = 5
    typing_indicator_delay: float = 0.05
    
    # Storage
    chroma_collection: str = "continuum_memory_v2"
    data_dir: str = "./continuum_data"
    
    @property
    def chroma_path(self) -> Path:
        return Path(self.data_dir) / "chromadb"
    
    @property
    def export_path(self) -> Path:
        return Path(self.data_dir) / "exports"
    
    @property
    def logs_path(self) -> Path:
        return Path(self.data_dir) / "logs"

# Global config instance
CONFIG = AppConfig()

# Create directories
for path in [CONFIG.chroma_path, CONFIG.export_path, CONFIG.logs_path]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================================================

CUSTOM_CSS = """
<style>
    /* ========== ROOT VARIABLES ========== */
    :root {
        --primary: #7c3aed;
        --primary-dark: #6d28d9;
        --primary-light: #a78bfa;
        --secondary: #ec4899;
        --background: #0f0f13;
        --surface: #1a1a24;
        --surface-hover: #22222e;
        --border: #2a2a35;
        --text: #e2e2e8;
        --text-muted: #a1a1b0;
        --success: #22c55e;
        --warning: #eab308;
        --error: #ef4444;
        --info: #3b82f6;
    }
    
    /* ========== GLOBAL ========== */
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, #0a0a0f 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: rgba(26, 26, 36, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    /* ========== HEADER ========== */
    .continuum-header {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        border-radius: 1rem;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeInDown 0.6s ease-out;
        box-shadow: 0 10px 30px -10px rgba(124, 58, 237, 0.3);
    }
    
    .continuum-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .continuum-header p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    /* ========== CHAT MESSAGES ========== */
    [data-testid="stChatMessage"] {
        border-radius: 1rem;
        margin-bottom: 0.75rem;
        animation: fadeInUp 0.4s ease-out;
        transition: all 0.2s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        transform: translateX(4px);
    }
    
    [data-testid="stChatMessage"][data-testid="user"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(236, 72, 153, 0.05));
        border: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* ========== INPUT AREA ========== */
    [data-testid="stChatInputTextArea"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        color: var(--text);
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stChatInputTextArea"]:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
    }
    
    /* ========== BUTTONS ========== */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }
    
    /* ========== METRICS ========== */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 0.5rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-muted);
    }
    
    [data-testid="stMetric"] value {
        color: var(--primary-light);
        font-weight: bold;
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .typing-indicator {
        display: flex;
        gap: 0.25rem;
        align-items: center;
        padding: 0.5rem 1rem;
        background: var(--surface);
        border-radius: 1rem;
        width: fit-content;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: var(--primary-light);
        border-radius: 50%;
        animation: pulse 1.4s ease-in-out infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--surface);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-dark);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* ========== HIDE BRANDING ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .continuum-header h1 { font-size: 1.5rem; }
        .continuum-header { padding: 1rem; }
    }
</style>
"""

# ============================================================================
# MEMORY METADATA
# ============================================================================

@dataclass
class MemoryEntry:
    """Rich memory entry with metadata"""
    id: str
    text: str
    strength: float
    created_at: float
    last_accessed: float
    access_count: int
    source: str
    tags: List[str] = field(default_factory=list)
    
    @property
    def strength_category(self) -> Tuple[str, str]:
        if self.strength >= 0.7:
            return "Strong", "#22c55e"
        elif self.strength >= 0.4:
            return "Moderate", "#eab308"
        return "Weak", "#ef4444"
    
    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source": self.source,
            "tags": self.tags
        }

# ============================================================================
# MEMORY SYSTEM - CORE
# ============================================================================

class ContinuumMemory:
    """Professional-grade memory system with vector-based semantic search and Ebbinghaus forgetting curve"""
    
    def __init__(self):
        self.config = CONFIG
        self._initialize_embedder()
        self._initialize_chromadb()
        self._load_metadata()
    
    def _initialize_embedder(self):
        """Initialize sentence transformer with error handling"""
        with st.spinner("🧠 Loading memory engine..."):
            try:
                self.embedder = SentenceTransformer(
                    self.config.embed_model,
                    device="cpu",
                    cache_folder=str(self.config.chroma_path / "models")
                )
                logger.info(f"Embedder initialized: {self.config.embed_model}")
            except Exception as e:
                logger.error(f"Failed to load embedder: {e}")
                st.error("Failed to initialize memory system. Please refresh.")
                raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with persistent storage"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.client.get_collection(self.config.chroma_collection)
                logger.info(f"Loaded existing collection: {self.collection.count()} memories")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.chroma_collection,
                    metadata={"hnsw:space": "cosine", "construction_ef": 200}
                )
                logger.info("Created new memory collection")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            st.error("Memory storage initialization failed.")
            raise
    
    def _load_metadata(self):
        """Load memory metadata from disk"""
        self.memories: Dict[str, MemoryEntry] = {}
        metadata_path = self.config.chroma_path / "metadata.json"
        
        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text())
                for mem_id, mem_data in data.items():
                    entry = MemoryEntry(
                        id=mem_id,
                        text=mem_data["text"],
                        strength=mem_data["strength"],
                        created_at=mem_data["created_at"],
                        last_accessed=mem_data["last_accessed"],
                        access_count=mem_data["access_count"],
                        source=mem_data.get("source", "conversation"),
                        tags=mem_data.get("tags", [])
                    )
                    self.memories[mem_id] = entry
                logger.info(f"Loaded {len(self.memories)} memory metadata entries")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
    
    def _save_metadata(self):
        """Save memory metadata to disk"""
        metadata_path = self.config.chroma_path / "metadata.json"
        data = {mem_id: entry.to_dict() for mem_id, entry in self.memories.items()}
        metadata_path.write_text(json.dumps(data, indent=2))
    
    def _generate_id(self, text: str) -> str:
        """Generate unique memory ID"""
        timestamp = int(time.time() * 1000)
        hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"mem_{timestamp}_{hash_val}"
    
    def add(self, text: str, source: str = "conversation", tags: List[str] = None) -> str:
        """Add a new memory to the system"""
        if not text or len(text.strip()) < 5:
            return ""
        
        memory_id = self._generate_id(text)
        embedding = self.embedder.encode(text).tolist()
        
        now = time.time()
        metadata = {
            "timestamp": now,
            "strength": 1.0,
            "source": source,
            "access_count": 0
        }
        
        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
            
            entry = MemoryEntry(
                id=memory_id,
                text=text,
                strength=1.0,
                created_at=now,
                last_accessed=now,
                access_count=0,
                source=source,
                tags=tags or []
            )
            self.memories[memory_id] = entry
            self._save_metadata()
            
            logger.info(f"Memory added: {text[:50]}...")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return ""
    
    def retrieve(self, query: str, top_k: int = None) -> List[MemoryEntry]:
        """Retrieve relevant memories with strength filtering"""
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
            logger.error(f"Retrieval failed: {e}")
            return []
        
        retrieved = []
        if not results["ids"] or not results["ids"][0]:
            return []
        
        for mem_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1.0 - dist
            strength = float(meta.get("strength", 0.0))
            
            if strength >= self.config.min_strength:
                if mem_id in self.memories:
                    self.memories[mem_id].last_accessed = time.time()
                    self.memories[mem_id].access_count += 1
                    self.memories[mem_id].strength = min(1.0, strength + self.config.reinforcement_boost)
                
                entry = self.memories.get(mem_id)
                if entry:
                    retrieved.append(entry)
                
                if len(retrieved) >= top_k:
                    break
        
        self._save_metadata()
        return retrieved
    
    def apply_decay(self):
        """Apply Ebbinghaus forgetting curve"""
        now = time.time()
        decayed_count = 0
        
        for mem_id, entry in list(self.memories.items()):
            days_since_access = (now - entry.last_accessed) / 86400
            decay_factor = math.exp(-self.config.decay_lambda * days_since_access)
            new_strength = entry.strength * decay_factor
            
            if new_strength < self.config.min_strength:
                try:
                    self.collection.delete(ids=[mem_id])
                    del self.memories[mem_id]
                    decayed_count += 1
                except Exception as e:
                    logger.error(f"Failed to prune {mem_id}: {e}")
            else:
                entry.strength = new_strength
        
        if decayed_count > 0:
            self._save_metadata()
            logger.info(f"Decay applied: {decayed_count} memories pruned")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        total = len(self.memories)
        if total == 0:
            return {
                "total": 0,
                "avg_strength": 0,
                "strong": 0,
                "moderate": 0,
                "weak": 0,
                "total_accesses": 0,
                "unique_tags": 0
            }
        
        strengths = [m.strength for m in self.memories.values()]
        tags = set()
        total_accesses = 0
        
        for m in self.memories.values():
            tags.update(m.tags)
            total_accesses += m.access_count
        
        strong = sum(1 for s in strengths if s >= 0.7)
        moderate = sum(1 for s in strengths if 0.4 <= s < 0.7)
        weak = sum(1 for s in strengths if s < 0.4)
        
        return {
            "total": total,
            "avg_strength": round(np.mean(strengths), 3),
            "strong": strong,
            "moderate": moderate,
            "weak": weak,
            "total_accesses": total_accesses,
            "unique_tags": len(tags)
        }
    
    def get_top_memories(self, limit: int = 5) -> List[MemoryEntry]:
        """Get strongest memories"""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda x: (x.strength, x.access_count),
            reverse=True
        )
        return sorted_memories[:limit]
    
    def export(self) -> Path:
        """Export all memories to JSON"""
        export_data = {
            "export_timestamp": time.time(),
            "export_date": datetime.now().isoformat(),
            "total_memories": len(self.memories),
            "config": {
                "embed_model": self.config.embed_model,
                "decay_lambda": self.config.decay_lambda,
                "min_strength": self.config.min_strength
            },
            "memories": [entry.to_dict() for entry in self.memories.values()]
        }
        
        filename = self.config.export_path / f"continuum_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(export_data, indent=2))
        return filename
    
    def reset(self):
        """Complete memory wipe"""
        try:
            self.client.delete_collection(self.config.chroma_collection)
            self.collection = self.client.create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )
            self.memories.clear()
            self._save_metadata()
            logger.warning("Memory system reset")
        except Exception as e:
            logger.error(f"Reset failed: {e}")

# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """Handles LLM interactions with context management"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        
        @st.cache_resource
        def _load_model():
            with st.spinner("🤖 Loading language model..."):
                return pipeline(
                    "text2text-generation",
                    model=CONFIG.llm_model,
                    device=-1,
                    model_kwargs={"torch_dtype": "float32"}
                )
        
        self.model = _load_model()
    
    def generate(self, query: str, memories: List[MemoryEntry]) -> str:
        """Generate response with context awareness"""
        try:
            context = ""
            if memories:
                context_parts = ["I recall:"]
                for mem in memories[:3]:
                    context_parts.append(f"• {mem.text}")
                context = "\n".join(context_parts) + "\n\n"
            
            prompt = f"""{context}User question: {query}
Answer concisely and helpfully, weaving in relevant memories naturally:"""
            
            result = self.model(
                prompt,
                max_length=CONFIG.max_tokens,
                temperature=CONFIG.temperature,
                do_sample=True,
                top_p=0.95
            )
            
            response = result[0]['generated_text'].strip()
            if not response:
                response = "I understand. Could you tell me more?"
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I appreciate you sharing that. Let me think about it."
    
    def clear_history(self):
        self.conversation_history = []

# ============================================================================
# FACT EXTRACTOR
# ============================================================================

class FactExtractor:
    """Intelligent fact extraction from conversation"""
    
    PATTERNS = {
        "name": [r"(?:my|our) name is (\w+)", r"i(?:'m| am) (\w+)", r"call me (\w+)"],
        "location": [r"i live (?:in|at) (.+?)(?:\.|,|$)", r"i(?:'m| am) from (.+?)(?:\.|,|$)"],
        "work": [r"i work (?:at|for|as) (.+?)(?:\.|,|$)", r"my job is (.+?)(?:\.|,|$)"],
        "interest": [r"i (?:like|love|enjoy|prefer) (.+?)(?:\.|,|$)"]
    }
    
    @classmethod
    def extract(cls, user_message: str, bot_response: str) -> List[str]:
        combined = f"{user_message} {bot_response}".lower()
        facts = []
        
        for category, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, combined, re.IGNORECASE)
                for match in matches:
                    fact_text = match if isinstance(match, str) else match[0]
                    fact_text = fact_text.strip().capitalize()
                    
                    if 3 <= len(fact_text) <= 50 and fact_text not in facts:
                        templates = {
                            "name": f"User's name is {fact_text}",
                            "location": f"User lives in {fact_text}",
                            "work": f"User works at {fact_text}",
                            "interest": f"User enjoys {fact_text}"
                        }
                        facts.append(templates.get(category, fact_text))
                        
                        if len(facts) >= 3:
                            return facts
        
        return facts[:3]

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Professional UI components"""
    
    @staticmethod
    def render_header():
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown("""
        <div class="continuum-header">
            <h1>🧠 Continuum RAG</h1>
            <p>Persistent Memory Chatbot • No API Keys • Free Forever • Enterprise Grade</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar(memory: ContinuumMemory) -> Dict[str, Any]:
        with st.sidebar:
            st.markdown("### 🧠 Memory Dashboard")
            
            stats = memory.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Memories", stats["total"], delta=None)
                st.metric("Strong Memories", stats["strong"])
            with col2:
                st.metric("Avg Strength", f"{stats['avg_strength']:.2f}")
                st.metric("Total Accesses", stats["total_accesses"])
            
            if stats["total"] > 0:
                st.markdown("---")
                st.markdown("#### Memory Strength")
                progress_data = [
                    (stats["strong"], "Strong", "#22c55e"),
                    (stats["moderate"], "Moderate", "#eab308"),
                    (stats["weak"], "Weak", "#ef4444")
                ]
                for count, label, color in progress_data:
                    if count > 0:
                        pct = (count / stats["total"]) * 100
                        st.markdown(f"<small>{label}: {count}</small>", unsafe_allow_html=True)
                        st.progress(pct / 100)
            
            st.markdown("---")
            st.markdown("#### 🔥 Strongest Memories")
            top_memories = memory.get_top_memories(5)
            if top_memories:
                for mem in top_memories:
                    strength_color = "#22c55e" if mem.strength >= 0.7 else "#eab308" if mem.strength >= 0.4 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03); border-radius:0.5rem; padding:0.5rem; margin-bottom:0.5rem">
                        <div style="font-size:0.8rem; color:#a1a1b0">{mem.text[:60]}...</div>
                        <div style="font-size:0.7rem; margin-top:0.25rem">
                            <span style="color:{strength_color}">●</span> Strength: {mem.strength:.2f}
                            <span style="margin-left:0.5rem">🔍 Accesses: {mem.access_count}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("*No memories yet. Start chatting!*")
            
            st.markdown("---")
            st.markdown("#### ⚡ Actions")
            export_clicked = st.button("💾 Export Memories", use_container_width=True)
            reset_clicked = st.button("🗑️ Reset All Memory", use_container_width=True, type="secondary")
            clear_clicked = st.button("💬 Clear Conversation", use_container_width=True)
            
            st.markdown("---")
            st.caption("🧠 Continuum RAG v2.0")
            st.caption(f"Model: {CONFIG.llm_model.split('/')[-1]}")
            st.caption("⚡ Local & Private")
            
            return {
                "export": export_clicked,
                "reset": reset_clicked,
                "clear": clear_clicked,
                "stats": stats
            }
    
    @staticmethod
    def render_typing_indicator():
        return st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="margin-left:0.5rem; color:#a1a1b0">Continuum is thinking...</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MESSAGE PROCESSOR
# ============================================================================

class MessageProcessor:
    """Handles message processing pipeline"""
    
    def __init__(self, memory: ContinuumMemory, generator: ResponseGenerator):
        self.memory = memory
        self.generator = generator
    
    def process(self, message: str) -> Tuple[str, List[MemoryEntry]]:
        """Process user message through the pipeline"""
        self.memory.apply_decay()
        memories = self.memory.retrieve(message)
        response = self.generator.generate(message, memories)
        facts = FactExtractor.extract(message, response)
        for fact in facts:
            self.memory.add(fact, source="extracted")
        return response, memories

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def initialize_session():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.memory = ContinuumMemory()
        st.session_state.generator = ResponseGenerator()
        st.session_state.processor = MessageProcessor(st.session_state.memory, st.session_state.generator)
        st.session_state.messages = []
        st.session_state.initialized = True

def main():
    """Main application entry point"""
    
    initialize_session()
    UIComponents.render_header()
    actions = UIComponents.render_sidebar(st.session_state.memory)
    
    if actions["reset"]:
        with st.spinner("Resetting memory system..."):
            st.session_state.memory.reset()
            st.session_state.messages = []
            st.success("✅ Memory system reset successfully!")
            time.sleep(1)
            st.rerun()
    
    if actions["clear"]:
        st.session_state.messages = []
        st.session_state.generator.clear_history()
        st.rerun()
    
    if actions["export"]:
        try:
            export_path = st.session_state.memory.export()
            with open(export_path, "r") as f:
                st.download_button(
                    label="📥 Download Export",
                    data=f.read(),
                    file_name=export_path.name,
                    mime="application/json"
                )
            st.success(f"✅ Exported {actions['stats']['total']} memories")
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    st.markdown("### 💬 Conversation")
    
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown("""
            Hello! I'm **Continuum**, your persistent memory assistant.
            
            I remember our conversations and learn about you over time. Try telling me:
            
            > *"My name is Sarah, and I love photography"*
            
            Then ask me *"What do you know about me?"* — I'll remember! ✨
            """)
    
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🧠"):
            placeholder = st.empty()
            with placeholder:
                UIComponents.render_typing_indicator()
            
            response, memories = st.session_state.processor.process(prompt)
            placeholder.empty()
            
            if memories:
                with st.expander(f"🔍 Retrieved {len(memories)} relevant memories", expanded=False):
                    for mem in memories:
                        strength_color = "#22c55e" if mem.strength >= 0.7 else "#eab308" if mem.strength >= 0.4 else "#ef4444"
                        st.markdown(f"""
                        <div style="margin-bottom:0.5rem; padding:0.5rem; background:rgba(255,255,255,0.03); border-radius:0.5rem">
                            <div style="font-size:0.85rem">{mem.text}</div>
                            <div style="font-size:0.7rem; margin-top:0.25rem">
                                <span style="color:{strength_color}">●</span> Strength: {mem.strength:.2f}
                                <span style="margin-left:0.5rem">🔄 Accessed: {mem.access_count} times</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown(response)
            
            facts = FactExtractor.extract(prompt, response)
            if facts:
                with st.status(f"📝 Learned {len(facts)} new fact(s)", expanded=False):
                    for fact in facts:
                        st.write(f"• {fact}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
