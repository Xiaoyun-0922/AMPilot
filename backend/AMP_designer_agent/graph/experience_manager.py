"""
Experience Learning and Storage System

Implements experience learning, storage and retrieval system based on memory_agent,
for accumulating knowledge and experience from sequence analysis.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from configuration import EXPERIENCE_DB_PATH, MAX_EXPERIENCES, EXPERIENCE_SIMILARITY_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Experience data structure"""
    id: str
    content: str  # Main content of the experience
    context: str  # Context information of the experience
    sequence_pair: Tuple[str, str]  # Related sequence pair
    properties_analysis: Dict[str, Any]  # Physicochemical properties analysis
    mic_analysis: Dict[str, Any]  # MIC analysis
    bacteria_analysis: Dict[str, Any]  # Bacterial analysis
    insights: List[str]  # Insights and conclusions
    confidence: float  # Confidence level
    created_at: str
    updated_at: str
    usage_count: int = 0  # Usage count

class ExperienceManager:
    """Experience manager"""

    def __init__(self, db_path: str = EXPERIENCE_DB_PATH):
        """
        Initialize experience manager

        Args:
            db_path: Experience database file path
        """
        self.db_path = db_path
        self.experiences: Dict[str, Experience] = {}
        self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.experience_embeddings: Dict[str, np.ndarray] = {}
        
        # Load existing experiences
        self.load_experiences()

    def load_experiences(self):
        """Load experience data from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for exp_data in data.get('experiences', []):
                    experience = Experience(**exp_data)
                    self.experiences[experience.id] = experience

                # Load embedding vectors
                embeddings_data = data.get('embeddings', {})
                for exp_id, embedding_list in embeddings_data.items():
                    if exp_id in self.experiences:
                        self.experience_embeddings[exp_id] = np.array(embedding_list)

                logger.info(f"Loaded {len(self.experiences)} experiences")

            except Exception as e:
                logger.error(f"Failed to load experience data: {str(e)}")
                self.experiences = {}
                self.experience_embeddings = {}
        else:
            logger.info("Experience database file not found, will create new database")

    def save_experiences(self):
        """Save experience data to file"""
        try:
            # Prepare data
            experiences_data = []
            embeddings_data = {}
            
            for exp_id, experience in self.experiences.items():
                experiences_data.append(asdict(experience))
                if exp_id in self.experience_embeddings:
                    embeddings_data[exp_id] = self.experience_embeddings[exp_id].tolist()
            
            data = {
                'experiences': experiences_data,
                'embeddings': embeddings_data,
                'metadata': {
                    'total_experiences': len(self.experiences),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Save to file
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.experiences)} experiences to {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to save experience data: {str(e)}")

    def add_experience(self, content: str, context: str, sequence_pair: Tuple[str, str],
                      insights: List[str], confidence: float = 0.8,
                      properties_analysis: Dict[str, Any] = None,
                      mic_analysis: Dict[str, Any] = None,
                      bacteria_analysis: Dict[str, Any] = None) -> str:
        """
        Add new experience

        Args:
            content: Experience content
            context: Context
            sequence_pair: Sequence pair
            properties_analysis: Physicochemical properties analysis
            mic_analysis: MIC analysis
            bacteria_analysis: Bacterial analysis
            insights: Insights list
            confidence: Confidence level

        Returns:
            Experience ID
        """
        # Check if similar experience exists
        similar_exp_id = self.find_similar_experience(content, context)

        if similar_exp_id:
            # Update existing experience
            logger.info(f"Found similar experience, updating experience: {similar_exp_id}")
            return self.update_experience(similar_exp_id, content, context,
                                        sequence_pair, properties_analysis,
                                        mic_analysis, bacteria_analysis,
                                        insights, confidence)
        else:
            # Create new experience
            exp_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            experience = Experience(
                id=exp_id,
                content=content,
                context=context,
                sequence_pair=sequence_pair,
                properties_analysis=properties_analysis or {},
                mic_analysis=mic_analysis or {},
                bacteria_analysis=bacteria_analysis or {},
                insights=insights,
                confidence=confidence,
                created_at=now,
                updated_at=now
            )
            
            self.experiences[exp_id] = experience
            
            # Calculate and store embedding vector
            combined_text = f"{content} {context} {' '.join(insights)}"
            embedding = self.embedder.encode([combined_text])[0]
            self.experience_embeddings[exp_id] = embedding

            # If experience count exceeds limit, delete oldest experiences
            if len(self.experiences) > MAX_EXPERIENCES:
                self._cleanup_old_experiences()

            logger.info(f"Added new experience: {exp_id}")
            return exp_id

    def update_experience(self, exp_id: str, content: str, context: str,
                         sequence_pair: Tuple[str, str], properties_analysis: Dict[str, Any],
                         mic_analysis: Dict[str, Any], bacteria_analysis: Dict[str, Any],
                         insights: List[str], confidence: float) -> str:
        """
        Update existing experience

        Args:
            exp_id: Experience ID
            Other parameters same as add_experience

        Returns:
            Experience ID
        """
        if exp_id in self.experiences:
            experience = self.experiences[exp_id]

            # Update experience content
            experience.content = content
            experience.context = context
            experience.sequence_pair = sequence_pair
            experience.properties_analysis = properties_analysis
            experience.mic_analysis = mic_analysis
            experience.bacteria_analysis = bacteria_analysis
            experience.insights = insights
            experience.confidence = max(experience.confidence, confidence)  # Take higher confidence
            experience.updated_at = datetime.now().isoformat()
            experience.usage_count += 1

            # Recalculate embedding vector
            combined_text = f"{content} {context} {' '.join(insights)}"
            embedding = self.embedder.encode([combined_text])[0]
            self.experience_embeddings[exp_id] = embedding

            logger.info(f"Updated experience: {exp_id}")
            return exp_id
        else:
            logger.warning(f"Experience ID does not exist: {exp_id}")
            return ""

    def find_similar_experience(self, content: str, context: str) -> Optional[str]:
        """
        Find similar experience

        Args:
            content: Experience content
            context: Context

        Returns:
            ID of similar experience, None if not found
        """
        if not self.experiences:
            return None

        # Calculate embedding vector for query text
        query_text = f"{content} {context}"
        query_embedding = self.embedder.encode([query_text])[0]

        # Calculate similarity with all experiences
        max_similarity = 0
        most_similar_id = None

        for exp_id, exp_embedding in self.experience_embeddings.items():
            similarity = cosine_similarity([query_embedding], [exp_embedding])[0][0]

            if similarity > max_similarity and similarity >= EXPERIENCE_SIMILARITY_THRESHOLD:
                max_similarity = similarity
                most_similar_id = exp_id

        return most_similar_id

    def retrieve_relevant_experiences(self, query: str, top_k: int = 5) -> List[Tuple[Experience, float]]:
        """
        Retrieve relevant experiences

        Args:
            query: Query text
            top_k: Number of experiences to return

        Returns:
            List of relevant experiences, each element is (experience, similarity)
        """
        if not self.experiences:
            return []

        # Calculate query embedding vector
        query_embedding = self.embedder.encode([query])[0]

        # Calculate similarity
        similarities = []
        for exp_id, experience in self.experiences.items():
            if exp_id in self.experience_embeddings:
                exp_embedding = self.experience_embeddings[exp_id]
                similarity = cosine_similarity([query_embedding], [exp_embedding])[0][0]
                similarities.append((experience, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_experience_by_id(self, exp_id: str) -> Optional[Experience]:
        """Get experience by ID"""
        return self.experiences.get(exp_id)

    def get_all_experiences(self) -> List[Experience]:
        """Get all experiences"""
        return list(self.experiences.values())

    def get_relevant_experiences(self, query: str, top_k: int = 5) -> List[Experience]:
        """
        Get relevant experiences based on query

        Args:
            query: Query string
            top_k: Number of top experiences to return

        Returns:
            List of relevant experiences
        """
        if not self.experiences:
            return []

        # Use similarity search to find relevant experiences
        # For now, return all experiences sorted by creation time
        all_experiences = list(self.experiences.values())
        all_experiences.sort(key=lambda x: x.created_at, reverse=True)
        return all_experiences[:top_k]

    def _cleanup_old_experiences(self):
        """Clean up old experiences, keep count within limit"""
        if len(self.experiences) <= MAX_EXPERIENCES:
            return

        # Sort by creation time, delete oldest experiences
        experiences_by_time = sorted(
            self.experiences.items(),
            key=lambda x: (x[1].usage_count, x[1].created_at)  # Prioritize deleting less used ones
        )

        num_to_delete = len(self.experiences) - MAX_EXPERIENCES

        for i in range(num_to_delete):
            exp_id, _ = experiences_by_time[i]
            del self.experiences[exp_id]
            if exp_id in self.experience_embeddings:
                del self.experience_embeddings[exp_id]
            logger.info(f"Deleted old experience: {exp_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get experience database statistics"""
        if not self.experiences:
            return {"total_experiences": 0}

        total_usage = sum(exp.usage_count for exp in self.experiences.values())
        avg_confidence = sum(exp.confidence for exp in self.experiences.values()) / len(self.experiences)

        return {
            "total_experiences": len(self.experiences),
            "total_usage": total_usage,
            "average_confidence": avg_confidence,
            "most_used_experience": max(self.experiences.values(), key=lambda x: x.usage_count).id,
            "latest_experience": max(self.experiences.values(), key=lambda x: x.created_at).id
        }
