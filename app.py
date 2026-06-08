#!/usr/bin/env python3
"""
Continuum RAG Chatbot - Persistent Memory System
Fully Local - No API Keys Required
Deployment-ready for Hugging Face Spaces, Streamlit, or Local
"""

import os
import sys
import time
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from threading import Thread
from datetime import datetime

import torch
import numpy as np
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from loguru import logger

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

BASE_PATH = Path(os.getenv("CONTINUUM_DATA_PATH", "./data"))
CHROMA_PATH = BASE_PATH / "db"
EXPORT_PATH = BASE_PATH / "exports"
IMAGE_PATH = BASE_PATH / "images"
LOG_PATH = BASE_PATH / "logs"
CONFIG_PATH = BASE_PATH / "config"

for d in [CHROMA_PATH, EXPORT_PATH, IMAGE_PATH, LOG_PATH, CONFIG_PATH]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{message}</cyan>"
)
logger.add(
    LOG_PATH / "app.log",
    rotation="100 MB",
    retention="7 days",
    level="DEBUG"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ContinuumConfig:
    """Immutable configuration for Continuum RAG system."""
    max_tokens: int = 512
    temperature: float = 0.7
    embed_model: str = "all-MiniLM-L6-v2"
    local_model: str = "google/flan-t5-small"
    chroma_collection: str = "continuum_memory"
    top_k: int = 5
    min_strength: float = 0.05
    decay_lambda: float = 0.1
    ctx_window_turns: int = 8
    base_path: str = str(BASE_PATH)

# ============================================================================
# MEMORY RESULT
# ============================================================================

@dataclass
class MemoryResult:
    """Single result returned from memory retrieval."""
    id: str
    text: str
    score: float
    strength: float
    age_days: float
    metadata: Dict[str, Any]

# ============================================================================
# RAG MEMORY SYSTEM
# ============================================================================

class RAGMemory:
    """Persistent memory with ChromaDB + Ebbinghaus decay."""

    def __init__(self, config: ContinuumConfig):
        self.config = config
        self.last_decay_time = time.time()
        self.session_added = 0
        self.session_pruned = 0
        
        # Embedding model (CPU - stable on all platforms)
        logger.info(f"Loading embedding model: {config.embed_model}")
        self.embedder = SentenceTransformer(config.embed_model, device="cpu")
        dim = self.embedder.get_sentence_embedding_dimension()
        logger.success(f"Embedder ready — dimension: {dim}")
        
        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(config.chroma_collection)
            count = self.collection.count()
            logger.info(f"Loaded collection '{config.chroma_collection}' ({count} memories)")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{config.chroma_collection}'")
        
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load session stats from disk."""
        stats_path = CONFIG_PATH / "stats.json"
        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text())
                self.session_added = data.get("session_added", 0)
                self.session_pruned = data.get("session_pruned", 0)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load stats: {e}")
    
    def _save_stats(self) -> None:
        """Persist session stats to disk."""
        stats_path = CONFIG_PATH / "stats.json"
        try:
            stats_path.write_text(json.dumps({
                "session_added": self.session_added,
                "session_pruned": self.session_pruned,
                "last_updated": time.time()
            }, indent=2))
        except OSError as e:
            logger.error(f"Could not save stats: {e}")
    
    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Embed and store a new memory."""
        if metadata is None:
            metadata = {}
        
        now = time.time()
        memory_metadata = {
            "timestamp": now,
            "strength": 1.0,
            "last_reinforced": now,
            "source": metadata.get("source", "conversation"),
            **{k: v for k, v in metadata.items() if k != "source"}
        }
        
        embedding = self.embedder.encode(text).tolist()
        memory_id = f"mem_{int(now * 1000)}_{abs(hash(text)) % 10000}"
        
        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[memory_metadata],
                documents=[text]
            )
            self.session_added += 1
            self._save_stats()
            logger.info(f"Memory added [{memory_id}]: {text[:60]}")
            return memory_id
        except Exception as e:
            logger.error(f"add_memory failed: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None, min_strength: float = None) -> List[MemoryResult]:
        """Retrieve relevant memories for a query."""
        if self.collection.count() == 0:
            return []
        
        top_k = top_k or self.config.top_k
        min_strength = min_strength if min_strength is not None else self.config.min_strength
        
        query_embedding = self.embedder.encode(query).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"retrieve query failed: {e}")
            return []
        
        memories: List[MemoryResult] = []
        if not results["ids"] or not results["ids"][0]:
            return []
        
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)
            strength = float(meta.get("strength", 0.0))
            age_days = (time.time() - float(meta.get("timestamp", time.time()))) / 86400
            
            if strength >= min_strength:
                memories.append(MemoryResult(
                    id=doc_id,
                    text=doc,
                    score=score,
                    strength=strength,
                    age_days=age_days,
                    metadata=meta
                ))
                if len(memories) >= top_k:
                    break
        
        logger.info(f"Retrieved {len(memories)} memories for: '{query[:50]}'")
        return memories
    
    def reinforce(self, memory_id: str) -> None:
        """Boost a memory's strength on re-access (capped at 1.0)."""
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not result["ids"]:
                logger.warning(f"reinforce: memory {memory_id} not found")
                return
            
            meta = result["metadatas"][0]
            old = float(meta.get("strength", 0.5))
            meta["strength"] = round(min(1.0, old + 0.3), 6)
            meta["last_reinforced"] = time.time()
            
            self.collection.update(ids=[memory_id], metadatas=[meta])
            logger.info(f"Reinforced {memory_id}: {old:.3f} → {meta['strength']:.3f}")
        except Exception as e:
            logger.error(f"reinforce failed: {e}")
    
    def decay_all(self) -> int:
        """Apply Ebbinghaus decay to all memories. Prune below min_strength."""
        if time.time() - self.last_decay_time <= 3600:
            logger.debug("Decay skipped — less than 1 hour since last run")
            return 0
        
        if self.collection.count() == 0:
            return 0
        
        logger.info("Running memory decay pass...")
        
        try:
            all_mem = self.collection.get(include=["metadatas", "documents"])
        except Exception as e:
            logger.error(f"decay_all fetch failed: {e}")
            return 0
        
        if not all_mem["ids"]:
            return 0
        
        pruned = 0
        updated = 0
        
        for mem_id, meta in zip(all_mem["ids"], all_mem["metadatas"]):
            old_strength = float(meta.get("strength", 1.0))
            last_reinforced = float(meta.get("last_reinforced", meta.get("timestamp", time.time())))
            days_since = (time.time() - last_reinforced) / 86400
            new_strength = old_strength * math.exp(-self.config.decay_lambda * days_since)
            
            if new_strength < self.config.min_strength:
                try:
                    self.collection.delete(ids=[mem_id])
                    pruned += 1
                except Exception as e:
                    logger.error(f"Failed to prune {mem_id}: {e}")
            else:
                try:
                    meta["strength"] = round(new_strength, 6)
                    self.collection.update(ids=[mem_id], metadatas=[meta])
                    updated += 1
                except Exception as e:
                    logger.error(f"Failed to update {mem_id}: {e}")
        
        self.session_pruned += pruned
        self.last_decay_time = time.time()
        self._save_stats()
        logger.info(f"Decay done — updated: {updated}, pruned: {pruned}")
        return pruned
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current memory system statistics."""
        total = self.collection.count()
        avg_strength = 0.0
        
        if total > 0:
            try:
                all_mem = self.collection.get(include=["metadatas"])
                strengths = [float(m.get("strength", 0)) for m in all_mem["metadatas"]]
                avg_strength = float(np.mean(strengths)) if strengths else 0.0
            except Exception:
                avg_strength = 0.0
        
        size_kb = 0.0
        if CHROMA_PATH.exists():
            size_kb = sum(f.stat().st_size for f in CHROMA_PATH.rglob("*") if f.is_file()) / 1024
        
        return {
            "total": total,
            "avg_strength": round(avg_strength, 3),
            "session_added": self.session_added,
            "session_pruned": self.session_pruned,
            "size_kb": round(size_kb, 2),
            "seconds_since_decay": round(time.time() - self.last_decay_time)
        }
    
    def get_top_facts(self, n: int = 5) -> List[Dict]:
        """Get the strongest memories."""
        if self.collection.count() == 0:
            return []
        try:
            all_mem = self.collection.get(include=["documents", "metadatas"])
            pairs = sorted(
                zip(all_mem["documents"], all_mem["metadatas"]),
                key=lambda x: float(x[1].get("strength", 0)),
                reverse=True
            )
            return [
                {"text": doc, "strength": float(meta.get("strength", 0))}
                for doc, meta in pairs[:n]
            ]
        except Exception:
            return []
    
    def export_json(self) -> Path:
        """Serialize all memories to a timestamped JSON file."""
        filename = EXPORT_PATH / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            all_mem = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"export_json fetch failed: {e}")
            raise
        
        export_data = {
            "export_timestamp": time.time(),
            "total_memories": len(all_mem["ids"]),
            "config": {
                "embed_model": self.config.embed_model,
                "collection": self.config.chroma_collection,
                "decay_lambda": self.config.decay_lambda,
                "min_strength": self.config.min_strength,
            },
            "memories": [
                {"id": mid, "text": doc, "metadata": meta}
                for mid, doc, meta in zip(all_mem["ids"], all_mem["documents"], all_mem["metadatas"])
            ]
        }
        
        filename.write_text(json.dumps(export_data, indent=2))
        logger.success(f"Exported {len(export_data['memories'])} memories → {filename}")
        return filename
    
    def reset(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        try:
            self.chroma_client.delete_collection(self.config.chroma_collection)
            self.collection = self.chroma_client.create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )
            self.session_added = 0
            self.session_pruned = 0
            self.last_decay_time = time.time()
            self._save_stats()
            logger.warning("Memory system reset — all memories deleted")
        except Exception as e:
            logger.error(f"reset failed: {e}")
            raise

# ============================================================================
# CONVERSATION BUFFER
# ============================================================================

class ConversationBuffer:
    """Sliding window conversation history with persistence."""
    
    BUFFER_PATH = CONFIG_PATH / "conversation_buffer.json"
    
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.buffer: deque = deque(maxlen=max_turns * 2)
        self._load_from_drive()
    
    def add(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content, "timestamp": time.time()})
        logger.debug(f"Buffer add [{role}]: {content[:60]}")
    
    def format_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.buffer]
    
    def save_to_drive(self) -> None:
        try:
            CONFIG_PATH.mkdir(parents=True, exist_ok=True)
            self.BUFFER_PATH.write_text(json.dumps(list(self.buffer), indent=2))
            logger.debug(f"Buffer saved ({len(self.buffer)} messages)")
        except OSError as e:
            logger.error(f"Buffer save failed: {e}")
    
    def _load_from_drive(self) -> None:
        if not self.BUFFER_PATH.exists():
            return
        try:
            data = json.loads(self.BUFFER_PATH.read_text())
            for msg in data[-(self.max_turns * 2):]:
                self.buffer.append(msg)
            logger.info(f"Buffer loaded ({len(self.buffer)} messages)")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Buffer load failed: {e}")
    
    def clear(self) -> None:
        self.buffer.clear()
        self.save_to_drive()
        logger.info("Conversation buffer cleared")
    
    def __len__(self) -> int:
        return len(self.buffer)

# ============================================================================
# FACT EXTRACTION
# ============================================================================

def extract_facts(user_msg: str, bot_msg: str) -> List[str]:
    """Extract personal facts from conversation using regex patterns."""
    combined = f"{user_msg} {bot_msg}".lower().strip()
    
    patterns = [
        r"my name is (\w+)",
        r"i(?:'m| am) (?:called )?(\w+)",
        r"i (?:like|love|enjoy|prefer) (.+?)(?:\.|,|$)",
        r"i (?:work|worked) (?:at|for) (.+?)(?:\.|,|$)",
        r"i live (?:in|at) (.+?)(?:\.|,|$)",
        r"i(?:'m| am) (?:a |an )?(\w[\w\s]{2,20}?) (?:who|by|at|in|and|but)",
        r"i have (?:a |an )?(.+?)(?:\.|,|$)",
        r"i(?:'ve| have) been (.+?)(?:\.|,|$)",
        r"my (.+?) is (.+?)(?:\.|,|$)",
        r"i(?:'m| am) from (.+?)(?:\.|,|$)",
    ]
    
    seen: set = set()
    facts: List[str] = []
    
    for pattern in patterns:
        for match in re.findall(pattern, combined):
            fact = " ".join(match).strip() if isinstance(match, tuple) else match.strip()
            fact = re.sub(r"\s+", " ", fact)
            
            if len(fact) < 3 or fact in seen:
                continue
            seen.add(fact)
            
            if re.fullmatch(r"[a-z]+", fact) and len(fact) < 20:
                facts.append(f"User's name is {fact.capitalize()}")
            elif any(w in pattern for w in ["like", "love", "enjoy", "prefer"]):
                facts.append(f"User enjoys {fact}")
            elif "live" in pattern:
                facts.append(f"User lives in {fact}")
            elif "work" in pattern:
                facts.append(f"User works at {fact}")
            elif "from" in pattern:
                facts.append(f"User is from {fact}")
            else:
                facts.append(f"User mentioned: {fact}")
            
            if len(facts) >= 3:
                break
        if len(facts) >= 3:
            break
    
    logger.info(f"Extracted {len(facts)} facts")
    return facts

# ============================================================================
# LOCAL LLM CLIENT (NO API KEY)
# ============================================================================

class LocalLLMClient:
    """Runs local Hugging Face model - no API key needed."""
    
    def __init__(self, config: ContinuumConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local model: {config.local_model} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.local_model)
        self.model.eval()
        
        logger.success(f"Local model loaded on {self.device}")
    
    def build_prompt(self, user_msg: str, memories: List[MemoryResult], history: List[Dict[str, str]]) -> str:
        """Construct the prompt with memory context."""
        context = ""
        if memories:
            facts = "\n".join(f"- {m.text}" for m in memories[:3])
            context = f"\n\nRelevant information I remember:\n{facts}\n"
        
        # Build conversation history
        conv_history = ""
        for msg in history[-4:]:  # Last 4 messages
            conv_history += f"{msg['role']}: {msg['content']}\n"
        
        prompt = f"""You are Continuum, a helpful AI assistant with persistent memory.
You remember facts about the user across conversations.
Be warm, concise, and helpful.{context}

{conv_history}User: {user_msg}
Assistant:"""
        
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Generate a response token by token."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.95
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I'm having trouble generating a response. Please try again."

# ============================================================================
# RESPONSE FUNCTION
# ============================================================================

def respond(
    message: str,
    history: List[List[Optional[str]]],
    top_k: int,
    decay_rate: float,
    memory: RAGMemory,
    conv_buffer: ConversationBuffer,
    llm: LocalLLMClient,
    config: ContinuumConfig
):
    """Stream response. history format: [[user, bot], ...]"""
    if not message or not message.strip():
        yield history
        return
    
    # Decay (guarded — max once/hour)
    memory.decay_all()
    
    # Retrieve + reinforce
    memories = memory.retrieve(message, top_k=int(top_k), min_strength=config.min_strength)
    for m in memories:
        memory.reinforce(m.id)
    
    # Build LLM history from Gradio history format
    llm_history: List[Dict[str, str]] = []
    for user_turn, bot_turn in history:
        if user_turn and isinstance(user_turn, str):
            llm_history.append({"role": "user", "content": user_turn})
        if bot_turn and isinstance(bot_turn, str):
            llm_history.append({"role": "assistant", "content": bot_turn})
    
    # Also include buffer context
    conv_buffer.add("user", message)
    
    # Build prompt
    prompt = llm.build_prompt(message, memories, llm_history)
    
    # Generate response
    full_response = llm.generate(prompt)
    
    # Post-generation updates
    if full_response and not full_response.startswith("[Error"):
        conv_buffer.add("assistant", full_response)
        conv_buffer.save_to_drive()
        for fact in extract_facts(message, full_response):
            memory.add_memory(fact, {"source": "conversation"})
    
    # Update history
    history.append((message, full_response))
    yield history

def do_reset(memory: RAGMemory, conv_buffer: ConversationBuffer) -> List:
    memory.reset()
    conv_buffer.clear()
    return []

def do_export(memory: RAGMemory) -> str:
    try:
        return str(memory.export_json())
    except Exception as e:
        return f"Export failed: {e}"

# ============================================================================
# CSS
# ============================================================================

CSS = """
body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Inter', sans-serif !important;
}
.sidebar-card {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    backdrop-filter: blur(8px) !important;
}
.chatbot {
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 12px !important;
}
button.primary {
    background: linear-gradient(135deg,#6d28d9,#7c3aed) !important;
    border: none !important;
}
button.stop {
    background: rgba(220,38,38,0.8) !important;
}
textarea, input[type=text] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
"""

# ============================================================================
# SIDEBAR HELPERS
# ============================================================================

def get_stats_md(memory: RAGMemory) -> str:
    s = memory.get_stats()
    return (
        f"### 🧠 Memory Stats\n"
        f"- **Total:** `{s['total']}`\n"
        f"- **Avg Strength:** `{s['avg_strength']:.2f}`\n"
        f"- **Added:** `{s['session_added']}`\n"
        f"- **Pruned:** `{s['session_pruned']}`\n"
        f"- **Size:** `{s['size_kb']:.1f} KB`\n"
    )

def get_facts_md(memory: RAGMemory) -> str:
    facts = memory.get_top_facts(5)
    if not facts:
        return "### 📌 Top Facts\n*No memories yet — start chatting!*"
    lines = "\n".join(
        f"- `{f['strength']:.2f}` {f['text'][:55]}{'...' if len(f['text']) > 55 else ''}"
        for f in facts
    )
    return f"### 📌 Top Facts\n{lines}"

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("Starting Continuum RAG Chatbot...")
    
    # Initialize components
    config = ContinuumConfig()
    memory = RAGMemory(config)
    conv_buffer = ConversationBuffer(max_turns=config.ctx_window_turns)
    llm = LocalLLMClient(config)
    
    logger.success("All components initialized!")
    
    # Gradio theme
    theme = gr.themes.Base(
        primary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    
    # Build UI
    with gr.Blocks(css=CSS, theme=theme, title="Continuum") as demo:
        
        gr.Markdown(
            "<div style='text-align:center;padding:16px;"
            "background:linear-gradient(135deg,#6d28d9,#7c3aed);"
            "border-radius:12px;margin-bottom:16px'>"
            "<h1 style='color:white;margin:0'>🧠 Continuum</h1>"
            "<p style='color:rgba(255,255,255,0.85);margin:4px 0 0'>"
            "Persistent RAG Chatbot · Local LLM · ChromaDB · No API Keys</p></div>"
        )
        
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                stats_display = gr.Markdown(get_stats_md(memory))
                facts_display = gr.Markdown(get_facts_md(memory))
                
                with gr.Accordion("⚙️ Settings", open=False):
                    top_k_slider = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Top-K Retrieval",
                        info="Memories retrieved per message"
                    )
                    decay_slider = gr.Slider(
                        0.01, 0.5, value=0.1, step=0.01,
                        label="Decay Rate (λ)",
                        info="Higher = faster forgetting"
                    )
                
                with gr.Row():
                    reset_btn = gr.Button("🗑️ Reset Memory", variant="stop", size="sm")
                    export_btn = gr.Button("💾 Export", variant="secondary", size="sm")
                
                export_out = gr.Textbox(label="Export Path", interactive=False, visible=True)
            
            # Chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    elem_classes=["chatbot"],
                    bubble_full_width=False,
                )
                
                with gr.Row():
                    msg_box = gr.Textbox(
                        scale=4,
                        placeholder="Type a message… e.g. 'My name is Alex, I enjoy Python'",
                        lines=2,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    gr.Markdown(
                        "<div style='color:#64748b;font-size:0.78em;padding-top:6px'>"
                        "Flan-T5-small · ChromaDB · sentence-transformers · No API Keys</div>"
                    )
        
        # Event wiring - using closures to capture current instances
        def make_respond():
            return lambda msg, hist, top_k, decay: respond(
                msg, hist, top_k, decay, memory, conv_buffer, llm, config
            )
        
        def make_reset():
            return lambda: do_reset(memory, conv_buffer)
        
        def make_export():
            return lambda: do_export(memory)
        
        def update_stats_facts():
            return get_stats_md(memory), get_facts_md(memory)
        
        send_btn.click(
            make_respond(),
            inputs=[msg_box, chatbot, top_k_slider, decay_slider],
            outputs=[chatbot],
            queue=True,
        ).then(lambda: "", outputs=msg_box).then(update_stats_facts, outputs=[stats_display, facts_display])
        
        msg_box.submit(
            make_respond(),
            inputs=[msg_box, chatbot, top_k_slider, decay_slider],
            outputs=[chatbot],
            queue=True,
        ).then(lambda: "", outputs=msg_box).then(update_stats_facts, outputs=[stats_display, facts_display])
        
        clear_btn.click(
            lambda: (conv_buffer.clear(), []),
            outputs=[chatbot],
        ).then(update_stats_facts, outputs=[stats_display, facts_display])
        
        reset_btn.click(
            make_reset(),
            outputs=[chatbot],
        ).then(update_stats_facts, outputs=[stats_display, facts_display])
        
        export_btn.click(make_export(), outputs=export_out)
        
        demo.load(update_stats_facts, outputs=[stats_display, facts_display])
    
    # Launch
    demo.queue(default_concurrency_limit=1)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
