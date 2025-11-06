"""Core cognitive memory system implementation."""

import os
import time
from collections import deque
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from strands.models import BedrockModel
from strands import Agent
from .models import MemoryItem, CognitiveState
from .vector_store import VectorStore
from config.settings import get_logger, load_config

logger = get_logger("core.memory_system")
config = load_config()


class CognitiveMemorySystem:
    """Multi-layered cognitive memory system with vector search and ReAct integration."""
    
    def __init__(self, 
                 embedding_model_id: str = "amazon.titan-embed-text-v1", 
                 synthesis_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
                 region: str = "us-east-1"):
        logger.info(f"Initializing CognitiveMemorySystem with embedding_model: {embedding_model_id}, synthesis_model: {synthesis_model_id}")

        # Create internal models
        self.embedding_model = BedrockModel(model_id=embedding_model_id, max_tokens=config.model.max_tokens)
        self.synthesis_model = BedrockModel(model_id=synthesis_model_id, max_tokens=config.model.max_tokens)
        
        # Create internal synthesis agent
        self._synthesis_agent = Agent(model=self.synthesis_model)

        # Layered memory buffers
        self.immediate_buffer = deque(maxlen=8)
        self.working_buffer = deque(maxlen=64)
        self.episodic_buffer = deque(maxlen=256)
        self.semantic_memory: Dict[str, MemoryItem] = {}
        
        # Vector storage with ChromaDB
        try:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            collection_name = os.getenv("CHROMA_COLLECTION", "cognitive_memory")
            
            self.vector_store = VectorStore(
                embedding_model=self.embedding_model,
                chroma_host=chroma_host,
                chroma_port=chroma_port,
                collection_name=collection_name
            )
            logger.info(f"ChromaDB vector store initialized: {chroma_host}:{chroma_port}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {e}")
            raise
        
        # Cognitive state tracking
        self.cognitive_state: Optional[CognitiveState] = None
        self.current_time = 0
        
        # Operation logging for reuse analysis
        self.operation_logs = []
        self.task_start_memory_size = 0  # Track memory size at task start
        
        # Memory management parameters
        self.attention_threshold = 0.5
        self.consolidation_threshold = 0.3
        self.similarity_threshold = 0.7
        
        logger.debug("CognitiveMemorySystem initialization complete")

    def process_task(self, task: str, documents: List[str] = None) -> Dict[str, Any]:
        """
        Main entry point for cognitive memory processing.
        
        Orchestrates the complete cognitive workspace workflow:
        1. Task understanding and planning
        2. Active information preparation  
        3. Progressive reasoning with memory reuse
        4. Synthesis and result generation
        """
        logger.info(f"Processing task: {task[:100]}...")
        start_time = time.time()
        
        # Check if this exact task has been processed before
        existing_task_knowledge = self._check_task_memory(task)
        if existing_task_knowledge:
            logger.info(f"Reusing existing knowledge for task: {task[:50]}...")
            return self._reuse_task_knowledge(task, existing_task_knowledge, start_time)
        # no need to continue if we don't have documents, nor memory
        if not existing_task_knowledge and not documents:
            return {}

        # Track memory size at task start for reuse analysis
        self.task_start_memory_size = len(self.working_buffer)
        
        # Phase 1: Task understanding and planning
        subtasks = self._decompose_task(task)
        
        # Initialize cognitive state
        self.cognitive_state = CognitiveState(
            current_task=task,
            subtasks=subtasks,
            completed_subtasks=[],
            information_gaps=[],
            working_hypothesis="",
            confidence_score=0.0,
        )
        
        # Phase 2: Active information preparation
        preparation_result = {}
        if documents:
            preparation_result = self._prepare_information_actively(task, documents)
        
        # Phase 3: Progressive reasoning
        insights = {}
        if preparation_result:
            insights = self._process_subtasks_progressively(subtasks)
        
        # Phase 4: Metacognitive assessment
        metacognitive_status = self.get_metacognitive_status()
        
        # Phase 5: Final synthesis
        final_synthesis = self._synthesize_final_result(task, insights)
        
        elapsed_time = time.time() - start_time
        
        result = {
            "task": task,
            "subtasks": subtasks,
            "insights": insights,
            "final_synthesis": final_synthesis,
            "preparation_result": preparation_result,
            "metacognitive_status": metacognitive_status,
            "processing_time": elapsed_time,
            "memory_state": {
                "immediate_buffer_size": len(self.immediate_buffer),
                "working_buffer_size": len(self.working_buffer),
                "episodic_buffer_size": len(self.episodic_buffer),
                "vector_store_size": len(self.vector_store.vectors)
            }
        }
        
        logger.info(f"Task processing complete in {elapsed_time:.2f}s")
        return result

    def get_metacognitive_status(self) -> Dict[str, Any]:
        """Get comprehensive metacognitive status."""
        if not self.cognitive_state:
            return {"error": "No active cognitive state"}
        
        information_gaps = self._assess_information_gaps(
            self.cognitive_state.current_task, 
            self.cognitive_state.completed_subtasks
        )
        
        confidence = self._evaluate_confidence()
        
        return {
            "current_task": self.cognitive_state.current_task,
            "progress": {
                "total_subtasks": len(self.cognitive_state.subtasks),
                "completed_subtasks": len(self.cognitive_state.completed_subtasks),
                "completion_ratio": len(self.cognitive_state.completed_subtasks) / max(1, len(self.cognitive_state.subtasks))
            },
            "information_gaps": information_gaps,
            "working_hypothesis": self.cognitive_state.working_hypothesis,
            "confidence_score": confidence,
            "memory_utilization": {
                "immediate_buffer": len(self.immediate_buffer),
                "working_buffer": len(self.working_buffer),
                "episodic_buffer": len(self.episodic_buffer),
                "vector_store": len(self.vector_store.vectors)
            }
        }

    def _decompose_task(self, task: str) -> List[str]:
        """Decompose task into subtasks using internal synthesis agent."""
        logger.debug(f"Decomposing task: {task[:50]}...")
        
        decomposition_prompt = f"""
        Decompose this task into subtasks. You must not produce more than 5 items:
        Task: {task}
        Output format: List of subtasks
        """
        try:
            subtasks_response = str(self._synthesis_agent(decomposition_prompt))
            return self._parse_subtasks(subtasks_response)
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return ["Analyze", "Process", "Synthesize"]

    def _parse_subtasks(self, response: str) -> List[str]:
        """Parse subtasks from LLM response."""
        lines = response.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                task = line.lstrip("0123456789.-) ").strip()
                if task:
                    subtasks.append(task)
        return subtasks if subtasks else ["Analyze", "Process", "Synthesize"]

    def _predict_information_needs(self, task: str) -> str:
        """Predict what information will be needed for a task."""
        logger.debug(f"Predicting information needs for task: {task[:50]}...")
        
        prediction_prompt = f"""
        Task: {task}
        Predict what information will be needed:
        """
        try:
            return str(self._synthesis_agent(prediction_prompt))
        except Exception as e:
            logger.error(f"Information needs prediction failed: {e}")
            return "Will need: definitions, examples, applications, limitations"

    def _assess_relevance(self, content: str, predicted_needs: str) -> float:
        """Assess content relevance against predicted needs."""
        if not predicted_needs:
            return 0.5

        keywords = predicted_needs.lower().split()
        content_lower = content.lower()

        matches = sum(1 for keyword in keywords if keyword in content_lower)
        relevance = min(1.0, matches / (len(keywords) + 1))

        return relevance

    def _process_subtasks_progressively(self, subtasks: List[str]) -> List[str]:
        """Process subtasks with progressive reasoning and memory reuse."""
        logger.debug(f"Processing {len(subtasks)} subtasks progressively")
        
        insights = []
        for subtask in subtasks:
            # Check if relevant information already exists in working memory
            existing_knowledge = self._check_working_memory(subtask)

            if existing_knowledge:
                logger.debug(f"Reusing existing knowledge: {len(existing_knowledge)} items")
                insight = self._synthesize_from_memory(existing_knowledge)
                # Log reuse operation
                self.operation_logs.append({"type": "memory_reuse", "subtask": subtask, "items_count": len(existing_knowledge)})
            else:
                logger.debug(f"Active retrieval for: {subtask}")
                new_info = self._active_retrieval(subtask)
                insight = self._process_new_information(new_info)
                # Log new information processing
                self.operation_logs.append({"type": "new_info_processing", "subtask": subtask, "items_count": len(new_info)})

            insights.append(insight)
            
            # Update cognitive state if available
            if self.cognitive_state:
                self.cognitive_state.completed_subtasks.append(subtask)
                self.cognitive_state.confidence_score = len(self.cognitive_state.completed_subtasks) / len(subtasks)

            # Add insights to working memory
            self._update_working_buffer(insight, subtask)

            # Consolidate memory after each subtask
            self._consolidate_memory()

        return insights

    def _check_working_memory(self, query: str) -> List[MemoryItem]:
        """Check relevant information in working memory with strict semantic matching."""
        relevant_items = []

        all_buffers = (
            list(self.working_buffer)
            + list(self.immediate_buffer)
            + list(self.episodic_buffer)
        )

        for item in all_buffers:
            query_words = set(query.lower().split())
            content_words = set(item.content.lower().split())

            # Very strict matching: require high word overlap (>50%) AND semantic relevance
            overlap = len(query_words & content_words)
            overlap_ratio = overlap / len(query_words) if query_words else 0
            
            # Only consider it relevant if:
            # 1. Very high word overlap (>50% of query words match), AND
            # 2. Content is substantial (not just generic phrases)
            if overlap_ratio > 0.5 and len(item.content) > 100:
                item.boost()
                relevant_items.append(item)

        return relevant_items

    def _synthesize_from_memory(self, memory_items: List[MemoryItem]) -> str:
        """Synthesize information from memory using internal synthesis agent."""
        if not memory_items:
            return "No relevant information found in memory"
        
        # Extract content from memory items
        contents = [item.content for item in memory_items[:5]]  # Use more items and full content
        combined_content = "\n\n".join(contents)
        
        synthesis_prompt = f"""Based on the following information from memory, provide a comprehensive synthesis:

{combined_content}

Please synthesize this information into a coherent and informative, yet succinct response."""
        
        try:
            return str(self._synthesis_agent(synthesis_prompt))
        except Exception as e:
            logger.error(f"Memory synthesis failed: {e}")
            return f"Memory synthesis: {combined_content[:200]}..."

    def _active_retrieval(self, subtask: str) -> List[str]:
        """Active retrieval (predictive rather than reactive)."""
        if self.vector_store:
            results = self.vector_store.search(subtask, top_k=3)
            return [document for _, _, document, _ in results]  # Extract document from (doc_id, similarity, document, metadata)
        return []

    def _process_new_information(self, info: List[str]) -> str:
        """Process new information using internal synthesis agent."""
        if not info:
            return "No new information found"
        
        # Combine multiple pieces of information
        combined_info = " | ".join(info[:3])  # Use top 3 results
        
        prompt = f"""
        Analyze this information and extract key insights:
        {combined_info}
        
        Provide a concise analysis focusing on the main points:
        """
        try:
            return str(self._synthesis_agent(prompt))
        except Exception as e:
            logger.error(f"New information processing failed: {e}")
            return f"Retrieved: {info[0][:200]}..." if info else "No information"

    def _update_working_buffer(self, content: str, source: str = "generation"):
        """Update working memory buffer."""
        # Avoid adding duplicate content
        for item in self.working_buffer:
            if item.content == content:
                item.boost()
                return

        memory_item = MemoryItem(
            content=content,
            embedding=self.vector_store.embed(content) if self.vector_store else np.zeros(384),
            creation_time=self.current_time,
            last_access_time=self.current_time,
            task_context=self.cognitive_state.current_task if self.cognitive_state else "",
            source=source,
        )

        self.working_buffer.append(memory_item)
        self.immediate_buffer.append(memory_item)

    def _intelligent_chunking(self, document: str) -> List[str]:
        """Intelligent chunking based on semantics rather than fixed size."""
        logger.debug(f"Chunking document of length: {len(document)}")
        
        # Split by sentences
        sentences = document.split(".")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < 200:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.debug(f"Document chunked into {len(chunks)} semantic chunks")
        return chunks

    def _prepare_information_actively(self, task: str, documents: List[str]) -> Dict[str, Any]:
        """Actively prepare information based on predicted needs."""
        logger.info(f"Actively preparing information for task: {task[:50]}...")
        
        # Predict needed information types
        predicted_needs = self._predict_information_needs(task)
        
        prepared_chunks = 0
        promoted_chunks = 0
        
        # Actively index and organize documents
        if self.vector_store and documents:
            for doc in documents:
                # Intelligent chunking
                chunks = self._intelligent_chunking(doc)
                for chunk in chunks:
                    # Add to vector store
                    idx = self.vector_store.add(
                        chunk, {"source": "document", "task": task}
                    )
                    prepared_chunks += 1

                    # Assess relevance and decide buffer level
                    relevance = self._assess_relevance(chunk, predicted_needs)
                    if relevance > self.consolidation_threshold:
                        self._promote_to_working_memory(chunk, relevance)
                        promoted_chunks += 1

        result = {
            "task": task,
            "predicted_needs": predicted_needs,
            "documents_processed": len(documents) if documents else 0,
            "chunks_prepared": prepared_chunks,
            "chunks_promoted": promoted_chunks,
            "promotion_rate": promoted_chunks / prepared_chunks if prepared_chunks > 0 else 0
        }
        
        logger.info(f"Information preparation complete: {prepared_chunks} chunks, {promoted_chunks} promoted")
        return result

    def _promote_to_working_memory(self, content: str, relevance: float):
        """Promote content to working memory."""
        memory_item = MemoryItem(
            content=content,
            embedding=self.vector_store.embed(content) if self.vector_store else np.zeros(384),
            relevance_score=relevance,
            creation_time=self.current_time,
            last_access_time=self.current_time,
            task_context=self.cognitive_state.current_task if self.cognitive_state else "",
            source="promotion",
        )

        self.working_buffer.append(memory_item)
        self.immediate_buffer.append(memory_item)

    def _assess_information_gaps(self, task: str, completed_subtasks: List[str]) -> List[str]:
        """Identify information gaps for metacognitive awareness."""
        logger.debug(f"Assessing information gaps for task: {task[:50]}...")
        
        information_gaps = []
        
        # Check if we have sufficient information for remaining subtasks
        if self.cognitive_state:
            remaining_subtasks = [
                subtask for subtask in self.cognitive_state.subtasks 
                if subtask not in completed_subtasks
            ]
            
            for subtask in remaining_subtasks:
                # Check working memory for relevant information
                relevant_items = self._check_working_memory(subtask)
                if len(relevant_items) < 2:  # Threshold for sufficient information
                    information_gaps.append(f"Insufficient information for: {subtask}")
        
        # Check overall task coverage
        if len(self.working_buffer) < 3:
            information_gaps.append("Limited working memory content")
        
        if len(self.vector_store.vectors) < 5:
            information_gaps.append("Insufficient knowledge base")
        
        logger.debug(f"Identified {len(information_gaps)} information gaps")
        return information_gaps

    def _update_working_hypothesis(self, new_evidence: str) -> str:
        """Update working hypothesis based on new evidence."""
        if not self.cognitive_state:
            return "No active cognitive state"
        
        current_hypothesis = self.cognitive_state.working_hypothesis
        
        if not current_hypothesis:
            # Initial hypothesis
            self.cognitive_state.working_hypothesis = f"Based on initial evidence: {new_evidence[:100]}..."
        else:
            # Update existing hypothesis
            self.cognitive_state.working_hypothesis = f"{current_hypothesis} | Updated with: {new_evidence[:50]}..."
        
        logger.debug("Working hypothesis updated")
        return self.cognitive_state.working_hypothesis

    def _evaluate_confidence(self) -> float:
        """Evaluate confidence based on available information and progress."""
        if not self.cognitive_state:
            return 0.0
        
        # Base confidence on task completion
        completion_ratio = len(self.cognitive_state.completed_subtasks) / max(1, len(self.cognitive_state.subtasks))
        
        # Adjust based on information availability
        info_factor = min(1.0, len(self.working_buffer) / 10)  # More info = higher confidence
        
        # Adjust based on information gaps
        gaps = self._assess_information_gaps(self.cognitive_state.current_task, self.cognitive_state.completed_subtasks)
        gap_penalty = len(gaps) * 0.1
        
        confidence = (completion_ratio * 0.6 + info_factor * 0.4) - gap_penalty
        confidence = max(0.0, min(1.0, confidence))
        
        self.cognitive_state.confidence_score = confidence
        logger.debug(f"Confidence evaluated: {confidence:.2f}")
        return confidence

    def _synthesize_final_result(self, task: str, insights: List[str]) -> str:
        """Synthesize final result from task and insights."""
        synthesis_prompt = f"""
        Task: {task}
        Key insights: {insights}
        Synthesize a comprehensive answer:
        """
        try:
            return str(self._synthesis_agent(synthesis_prompt))
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            return f"Task: {task}. Based on analysis: {'; '.join(insights[:3])}"

    def _find_similar_memories(self, content: str, threshold: float = 0.8) -> List[Tuple[float, str]]:
        """Find similar existing memories."""
        results = self.vector_store.search(content, top_k=5, threshold=threshold)
        return [(similarity, text) for _, similarity, text, _ in results]

    def _search_buffers(self, query: str) -> List[Tuple[str, float, str]]:
        """Search memory buffers for relevant content."""
        results = []
        query_words = set(query.lower().split())
        
        # Search all buffers
        all_buffers = [
            ("immediate", self.immediate_buffer),
            ("working", self.working_buffer),
            ("episodic", self.episodic_buffer)
        ]
        
        for buffer_name, buffer in all_buffers:
            for item in buffer:
                content_words = set(item.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    relevance = overlap / len(query_words)
                    results.append((buffer_name, relevance, item.content))
        
        return results

    def _combine_search_results(self, vector_results: List[Tuple], 
                               buffer_results: List[Tuple], top_k: int) -> List[Tuple]:
        """Combine and rank search results from different sources."""
        combined = []
        
        # Add vector results with metadata
        for idx, similarity, content, metadata in vector_results:
            combined.append((idx, similarity * 1.2, content, metadata))  # Boost vector results
        
        # Add buffer results
        for buffer_name, relevance, content in buffer_results:
            # Check if already in vector results
            if not any(content == c for _, _, c, _ in combined):
                combined.append((len(combined), relevance, content, {"source_buffer": buffer_name}))
        
        # Sort by similarity/relevance and return top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _update_access_pattern(self, content: str) -> None:
        """Update access patterns for retrieved content."""
        # Find and boost corresponding memory items
        for buffer in [self.immediate_buffer, self.working_buffer, self.episodic_buffer]:
            for item in buffer:
                if item.content == content:
                    item.boost()
                    break

    def _identify_source_buffer(self, content: str) -> str:
        """Identify which buffer contains the content."""
        for buffer_name, buffer in [
            ("immediate", self.immediate_buffer),
            ("working", self.working_buffer),
            ("episodic", self.episodic_buffer)
        ]:
            if any(item.content == content for item in buffer):
                return buffer_name
        return "vector_store"

    def _consolidate_memory(self):
        """
        Memory consolidation and organization.
        
        Implements controlled memory management through:
        1. Temporal decay (Ebbinghaus forgetting curve)
        2. Selective consolidation to episodic memory
        3. Attention threshold filtering
        4. Semantic clustering organization
        """
        logger.debug("Starting memory consolidation")
        self.current_time += 1

        # Apply forgetting curve to all buffers
        for buffer in [self.immediate_buffer, self.working_buffer, self.episodic_buffer]:
            for item in buffer:
                item.decay(self.current_time)

        # Remove low relevance items from working buffer
        original_size = len(self.working_buffer)
        self.working_buffer = deque(
            [item for item in self.working_buffer if item.relevance_score > self.attention_threshold],
            maxlen=self.working_buffer.maxlen,
        )
        removed_count = original_size - len(self.working_buffer)

        # Promote important items to episodic memory
        promoted_count = 0
        for item in self.working_buffer:
            if item.access_count > 2 and not any(id(item) == id(e) for e in self.episodic_buffer):
                self.episodic_buffer.append(item)
                promoted_count += 1

        # Organize semantic clusters
        cluster_count = self._organize_semantic_clusters()

        logger.debug(f"Memory consolidation complete: removed {removed_count}, promoted {promoted_count}, {cluster_count} clusters")

    def _organize_semantic_clusters(self) -> int:
        """Organize memories into semantic clusters."""
        clusters = {}
        cluster_count = 0
        
        for item in self.episodic_buffer:
            # Find cluster or create new one
            assigned = False
            for cluster_key, cluster_items in clusters.items():
                if len(cluster_items) > 0:
                    similarity = self.vector_store._cosine_similarity(
                        item.embedding, cluster_items[0].embedding
                    )
                    if similarity > 0.8:
                        cluster_items.append(item)
                        assigned = True
                        break
            
            if not assigned:
                clusters[f"cluster_{cluster_count}"] = [item]
                cluster_count += 1
        
        return len(clusters)

    def _check_task_memory(self, task: str) -> List[MemoryItem]:
        """Check if this exact task has been processed before."""
        # Search ChromaDB for content related to this specific task
        if self.vector_store:
            results = self.vector_store.search(task, top_k=5)
            # Convert search results to MemoryItem objects
            memory_items = []
            for doc_id, similarity, document, metadata in results:
                if similarity > 0.005:  # Higher threshold for task-level reuse
                    memory_item = MemoryItem(
                        content=document,
                        embedding=np.array([]),  # Empty embedding for reuse
                        relevance_score=similarity,
                        access_count=1,
                        task_context=task,
                        source="chromadb_reuse"
                    )
                    memory_items.append(memory_item)
            return memory_items
        return []

    def _reuse_task_knowledge(self, task: str, existing_knowledge: List[MemoryItem], start_time: float) -> Dict[str, Any]:
        """Reuse existing knowledge for a previously processed task."""
        # Synthesize the existing knowledge into a comprehensive response
        if existing_knowledge:
            synthesis = self._synthesize_from_memory(existing_knowledge)
            insights = [synthesis]
            
            # Log the reuse operation
            self.operation_logs.append({
                "type": "task_reuse", 
                "task": task, 
                "items_count": len(existing_knowledge)
            })
        else:
            synthesis = f"Previously processed task: {task}"
            insights = [synthesis]
        
        elapsed_time = time.time() - start_time
        
        return {
            "task": task,
            "subtasks": ["Task Reuse"],
            "insights": insights,
            "final_synthesis": synthesis,
            "preparation_result": {},
            "metacognitive_status": {
                "current_task": task,
                "progress": {
                    "total_subtasks": 1,
                    "completed_subtasks": 1,
                    "completion_ratio": 1.0
                },
                "information_gaps": [],
                "working_hypothesis": "Reusing existing knowledge",
                "confidence_score": 0.9,  # High confidence for reuse
                "memory_utilization": {
                    "immediate_buffer": len(self.immediate_buffer),
                    "working_buffer": len(self.working_buffer),
                    "episodic_buffer": len(self.episodic_buffer),
                    "vector_store": len(self.vector_store.vectors) if self.vector_store else 0
                }
            },
            "processing_time": elapsed_time,
            "memory_state": {
                "immediate_buffer_size": len(self.immediate_buffer),
                "working_buffer_size": len(self.working_buffer),
                "episodic_buffer_size": len(self.episodic_buffer),
                "vector_store_size": len(self.vector_store.vectors) if self.vector_store else 0
            }
        }
