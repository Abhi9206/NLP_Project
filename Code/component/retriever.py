import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class StoryRetriever:
    """
    Retrieval system to find similar stories for guidance.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', k=3):
        self.encoder = SentenceTransformer(model_name)
        self.k = k
        self.index = None
        self.stories = None

    def build_index(self, stories):
        """
        Build retrieval index from training stories.

        Args:
            stories: List of story texts
        """
        print(f"Building retrieval index from {len(stories):,} stories...")

        # Encode all stories
        embeddings = self.encoder.encode(stories, show_progress_bar=True)

        # Build nearest neighbor index
        self.index = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.index.fit(embeddings)
        self.stories = stories

        print(f"✓ Index built with {len(stories):,} stories")

    def retrieve(self, query_text, keywords=None):
        """
        Retrieve top-K similar stories.

        Args:
            query_text: Partial sentence to query
            keywords: Optional keywords to bias retrieval

        Returns:
            List of retrieved story texts
        """
        # Create query (combine partial + keywords)
        if keywords:
            query = f"{query_text} Keywords: {', '.join(keywords)}"
        else:
            query = query_text

        # Encode query
        query_embedding = self.encoder.encode([query])

        # Find nearest neighbors
        distances, indices = self.index.kneighbors(query_embedding)

        # Return retrieved stories
        retrieved = [self.stories[idx] for idx in indices[0]]

        return retrieved