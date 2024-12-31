from typing import Any, Dict, List, Set, Union, Optional, Callable, Tuple, Iterator
import datetime
from uuid import UUID, uuid4
from enum import Enum
from collections import Counter, defaultdict
import json
import numpy as np
from pydantic import BaseModel, Field, HttpUrl, SecretStr, validator
import orjson
import logging
from scipy.spatial.distance import cosine
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptFormat(str, Enum):
    """Supported prompt formats"""
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    CHAIN_OF_THOUGHT = "chain-of-thought"
    Q_AND_A = "q-and-a"
    TREE_OF_THOUGHT = "tree-of-thought"
    XML_STRUCTURED = "xml-structured"

class SearchStrategy(str, Enum):
    """Search strategies for thought exploration"""
    BFS = "breadth-first"
    DFS = "depth-first"
    BEAM = "beam-search"
    BEST_FIRST = "best-first"

class ThoughtStatus(str, Enum):
    """Status tracking for thought branches"""
    ACTIVE = "active"
    ABANDONED = "abandoned"
    COMPLETED = "completed"
    BLOCKED = "blocked"

def orjson_dumps(v, *, default, **kwargs) -> str:
    """Enhanced JSON serialization"""
    return orjson.dumps(v, default=default, option=orjson.OPT_SERIALIZE_NUMPY, **kwargs).decode()

def now_tz() -> datetime.datetime:
    """Returns current UTC time with timezone"""
    return datetime.datetime.now(datetime.timezone.utc)

def calculate_embedding_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    """Calculate cosine similarity between embeddings"""
    if embed1 is None or embed2 is None:
        return 0.0
    return 1 - cosine(embed1, embed2)

class ThoughtBranch(BaseModel):
    """Comprehensive thought branch management"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    thoughts: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    status: ThoughtStatus = ThoughtStatus.ACTIVE
    evaluation_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=now_tz)
    updated_at: datetime.datetime = Field(default_factory=now_tz)
    
    def add_thought(self, thought: str, confidence: Optional[float] = None) -> None:
        """Add a new thought to the branch"""
        self.thoughts.append(thought)
        if confidence is not None:
            self.confidence = confidence
        self.updated_at = now_tz()
        
    def add_child(self, child_id: str) -> None:
        """Add a child branch"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.updated_at = now_tz()
            
    def update_status(self, status: ThoughtStatus, score: Optional[float] = None) -> None:
        """Update branch status and evaluation"""
        self.status = status
        if score is not None:
            self.evaluation_score = score
        self.updated_at = now_tz()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert branch to dictionary"""
        return {
            "id": self.id,
            "thoughts": self.thoughts,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "status": self.status,
            "evaluation_score": self.evaluation_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class DocumentChunk(BaseModel):
    """Manages document chunks for context handling"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    summary: Optional[str] = None
    tokens: Optional[int] = None
    chunk_index: int = 0
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.tokens is None and self.content:
            self.tokens = self.count_tokens(self.content)
            
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split())
            
    def update_embedding(self, embedding: np.ndarray) -> None:
        """Update chunk embedding"""
        self.embedding = embedding
        
    def update_summary(self, summary: str) -> None:
        """Update chunk summary"""
        self.summary = summary
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "summary": self.summary,
            "tokens": self.tokens,
            "chunk_index": self.chunk_index
        }
        
class Message(BaseModel):
    """Enhanced message class with comprehensive prompting capabilities"""
    # Base fields from original
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[str] = None
    received_at: datetime.datetime = Field(default_factory=now_tz)
    finish_reason: Optional[str] = None
    prompt_length: Optional[int] = None
    completion_length: Optional[int] = None
    total_length: Optional[int] = None

    # Format and structure handling
    format: PromptFormat = PromptFormat.ZERO_SHOT
    format_metadata: Dict[str, Any] = Field(default_factory=dict)
    structured_output: Optional[Dict[str, Any]] = None
    xml_tags: Optional[List[str]] = None

    # Reasoning and validation
    reasoning_path: List[str] = Field(default_factory=list)
    validation_status: Optional[bool] = None
    confidence_score: float = 1.0
    alternative_responses: List[str] = Field(default_factory=list)

    # Code integration
    code_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    code_outputs: List[Any] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def add_reasoning_step(self, step: str, confidence: Optional[float] = None) -> None:
        """Add a reasoning step with optional confidence score"""
        self.reasoning_path.append(step)
        if confidence is not None:
            self.confidence_score *= confidence

    def add_code_block(self, code: str, language: str = "python", metadata: Optional[Dict] = None) -> None:
        """Add a code block with metadata"""
        self.code_blocks.append({
            "code": code,
            "language": language,
            "metadata": metadata or {},
            "added_at": now_tz().isoformat()
        })

    def add_code_output(self, output: Any, metadata: Optional[Dict] = None) -> None:
        """Add output from code execution"""
        self.code_outputs.append({
            "output": output,
            "metadata": metadata or {},
            "added_at": now_tz().isoformat()
        })

    def set_structured_output(self, output: Dict[str, Any], format_type: str = "json") -> None:
        """Set structured output with format specification"""
        self.structured_output = output
        if format_type == "xml":
            self.xml_tags = list(output.keys())

    def add_alternative_response(self, response: str, confidence: Optional[float] = None) -> None:
        """Add alternative response for self-consistency"""
        self.alternative_responses.append(response)
        if confidence is not None:
            self.format_metadata[f"confidence_{len(self.alternative_responses)}"] = confidence

    def get_token_count(self) -> int:
        """Calculate total tokens in message"""
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = len(encoding.encode(self.content))
        
        # Add tokens from reasoning path
        for step in self.reasoning_path:
            total_tokens += len(encoding.encode(step))
            
        # Add tokens from code blocks
        for block in self.code_blocks:
            total_tokens += len(encoding.encode(block["code"]))
            
        return total_tokens

    def to_format(self, target_format: PromptFormat) -> 'Message':
        """Convert message to different format"""
        new_content = self.content
        if target_format == PromptFormat.Q_AND_A and self.format != PromptFormat.Q_AND_A:
            new_content = f"Q: {self.content}\nA: "
        elif target_format == PromptFormat.CHAIN_OF_THOUGHT:
            new_content = f"Let's solve this step by step:\n1. {self.content}"
            
        return Message(
            role=self.role,
            content=new_content,
            format=target_format,
            format_metadata=self.format_metadata,
            name=self.name,
            function_call=self.function_call
        )

    def validate(self) -> bool:
        """Validate message content and structure"""
        if not self.content.strip():
            return False
            
        if self.format == PromptFormat.XML_STRUCTURED and not self.xml_tags:
            return False
            
        if self.code_blocks and not self.code_outputs:
            logger.warning("Code blocks present without outputs")
            
        return True

    def __str__(self) -> str:
        """String representation with format consideration"""
        base_repr = self.model_dump(exclude_none=True)
        if self.format == PromptFormat.XML_STRUCTURED and self.structured_output:
            return json.dumps(self.structured_output, indent=2)
        return str(base_repr)

class Session(BaseModel):
    """Enhanced session class with comprehensive prompting capabilities"""
    # Base configuration from original
    id: Union[str, UUID] = Field(default_factory=uuid4)
    created_at: datetime.datetime = Field(default_factory=now_tz)
    auth: Dict[str, SecretStr]
    api_url: HttpUrl
    model: str
    
    # Core message handling
    messages: List[Message] = Field(default_factory=list)
    input_fields: Set[str] = {"role", "content", "name"}
    recent_messages: Optional[int] = None
    save_messages: Optional[bool] = True
    total_prompt_length: int = 0
    total_completion_length: int = 0
    total_length: int = 0
    title: Optional[str] = None

    # Advanced context and memory management
    system: str = "You are a helpful assistant."
    context_window: int = 4096
    context_overlap: int = 200
    chunks: List[DocumentChunk] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Thought and reasoning management
    thought_branches: Dict[str, ThoughtBranch] = Field(default_factory=dict)
    search_strategy: SearchStrategy = SearchStrategy.BFS
    max_branches: int = 5
    min_branch_confidence: float = 0.3
    
    # Format handling
    current_format: PromptFormat = PromptFormat.ZERO_SHOT
    format_handlers: Dict[Tuple[PromptFormat, PromptFormat], Callable] = Field(default_factory=dict)
    
    # Tool integration
    params: Dict[str, Any] = Field(default_factory=dict)
    tools: Dict[str, Callable] = Field(default_factory=dict)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def chunk_content(self, content: str) -> List[DocumentChunk]:
        """Split content into manageable chunks with overlap"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(content)
        chunks = []
        
        current_chunk = []
        current_tokens = 0
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = encoding.encode(sentence)
            sentence_token_count = len(sentence_tokens)
            
            if current_tokens + sentence_token_count > self.context_window:
                # Create new chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=len(chunks),
                    metadata={"original_position": i - len(current_chunk)}
                ))
                
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - self.context_overlap)
                current_chunk = current_chunk[overlap_point:]
                current_tokens = sum(len(encoding.encode(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_token_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(DocumentChunk(
                content=chunk_text,
                chunk_index=len(chunks),
                metadata={"original_position": len(sentences) - len(current_chunk)}
            ))
        
        return chunks

    def add_thought_branch(self, content: str, parent_id: Optional[str] = None) -> str:
        """Create new thought branch with optional parent"""
        branch = ThoughtBranch(
            thoughts=[content],
            parent_id=parent_id,
            metadata={"created_by": self.id}
        )
        
        self.thought_branches[branch.id] = branch
        if parent_id and parent_id in self.thought_branches:
            self.thought_branches[parent_id].add_child(branch.id)
            
        return branch.id

    def evaluate_branch(self, branch_id: str) -> float:
        """Evaluate a thought branch's potential"""
        branch = self.thought_branches.get(branch_id)
        if not branch:
            return 0.0
            
        # Implement scoring based on:
        # 1. Previous successful patterns
        # 2. Confidence scores
        # 3. Reasoning coherence
        base_score = branch.confidence
        
        # Pattern matching from memory
        if self.memory.get("successful_patterns"):
            for pattern in self.memory["successful_patterns"]:
                if any(pattern in thought for thought in branch.thoughts):
                    base_score *= 1.1

        # Reasoning coherence
        if len(branch.thoughts) > 1:
            coherence_score = sum(1 for i in range(len(branch.thoughts)-1)
                                if self._thoughts_connected(branch.thoughts[i], branch.thoughts[i+1]))
            base_score *= (1 + coherence_score / len(branch.thoughts))

        return min(1.0, base_score)

    def _thoughts_connected(self, thought1: str, thought2: str) -> bool:
        """Check if two thoughts are logically connected"""
        # Simple keyword continuity check
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())
        return len(words1.intersection(words2)) > 0

    def expand_branch(self, branch_id: str, num_expansions: int = 3) -> List[str]:
        """Expand a thought branch into multiple possibilities"""
        branch = self.thought_branches.get(branch_id)
        if not branch or branch.status != ThoughtStatus.ACTIVE:
            return []
            
        new_branch_ids = []
        for _ in range(num_expansions):
            new_branch_id = self.add_thought_branch(
                f"Alternative expansion of: {branch.thoughts[-1]}",
                parent_id=branch_id
            )
            new_branch_ids.append(new_branch_id)
            
        return new_branch_ids

    def format_input_messages(
        self, 
        system_message: Message, 
        user_message: Message
    ) -> list:
        """Format messages with all prompting enhancements"""
        messages = []
        
        # Add system message with potential DNA/Matrix prompting
        system_content = system_message.content
        if self.memory.get("successful_prompts"):
            # Enhance with learned patterns
            patterns = self.memory["successful_prompts"][:3]
            system_content = f"{system_content}\n\nPreviously successful approaches:\n"
            system_content += "\n".join(patterns)
            
        messages.append(Message(
            role="system",
            content=system_content
        ).model_dump(include=self.input_fields))

        # Handle format transitions if needed
        if user_message.format != self.current_format:
            transition_key = (self.current_format, user_message.format)
            if transition_key in self.format_handlers:
                user_message = self.format_handlers[transition_key](user_message)
            self.current_format = user_message.format

        # Add context from chunks if present
        if self.chunks:
            relevant_chunks = self._get_relevant_chunks(user_message.content)
            for chunk in relevant_chunks:
                messages.append(Message(
                    role="system",
                    content=f"Context: {chunk.content}",
                    metadata={"chunk_id": chunk.id}
                ).model_dump(include=self.input_fields))

        # Add conversation history
        recent_messages = (
            self.messages[-self.recent_messages:]
            if self.recent_messages else self.messages
        )
        
        # Add thought trails if present
        for msg in recent_messages:
            if msg.reasoning_path:
                content = f"{msg.content}\nReasoning:\n"
                content += "\n".join(f"- {step}" for step in msg.reasoning_path)
                msg.content = content
            messages.append(msg.model_dump(include=self.input_fields))

        # Add current message
        messages.append(user_message.model_dump(include=self.input_fields))
        
        return messages

    def _get_relevant_chunks(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """Retrieve most relevant context chunks for query"""
        if not self.chunks:
            return []
            
        # Score chunks based on term overlap and position
        chunk_scores = []
        query_terms = set(query.lower().split())
        
        for chunk in self.chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap_score = len(query_terms.intersection(chunk_terms)) / len(query_terms)
            position_score = 1 / (chunk.chunk_index + 1)  # Favor earlier chunks
            
            combined_score = (0.7 * overlap_score) + (0.3 * position_score)
            chunk_scores.append((chunk, combined_score))
            
        # Return top_k chunks
        return [c[0] for c in sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:top_k]]

    def add_messages(
        self,
        user_message: Message,
        assistant_message: Message,
        save_messages: bool = None,
    ) -> None:
        """Add messages with enhanced tracking and learning"""
        to_save = isinstance(save_messages, bool)
        
        # Update metrics
        self.total_prompt_length += user_message.get_token_count()
        self.total_completion_length += assistant_message.get_token_count()
        self.total_length = self.total_prompt_length + self.total_completion_length
        
        # Learn from successful interactions
        if assistant_message.confidence_score > 0.8:
            self._update_successful_patterns(assistant_message)
        
        # Save messages if configured
        if (to_save and save_messages) or (not to_save and self.save_messages):
            self.messages.append(user_message)
            self.messages.append(assistant_message)

    def _update_successful_patterns(self, message: Message) -> None:
        """Learn from successful interactions"""
        if not self.memory.get("successful_patterns"):
            self.memory["successful_patterns"] = []
            
        if message.reasoning_path:
            self.memory["successful_patterns"].extend(message.reasoning_path)
            
        # Keep only recent patterns
        self.memory["successful_patterns"] = self.memory["successful_patterns"][-50:]

    def __str__(self) -> str:
        """Enhanced string representation"""
        base_str = super().__str__()
        
        # Add thought branch statistics
        active_branches = sum(1 for b in self.thought_branches.values() 
                            if b.status == ThoughtStatus.ACTIVE)
        completed_branches = sum(1 for b in self.thought_branches.values() 
                               if b.status == ThoughtStatus.COMPLETED)
                               
        return f"{base_str}\nActive thought branches: {active_branches}\nCompleted branches: {completed_branches}"