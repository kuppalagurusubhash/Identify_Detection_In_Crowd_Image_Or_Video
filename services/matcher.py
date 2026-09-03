import numpy as np
from utils.database import get_all_embeddings

class IdentityMatcher:
    def __init__(self, default_threshold=0.60):
        self.default_threshold = default_threshold

    def match_embedding(self, query_embedding, threshold=None):
        """
        Matches a query embedding vector against all stored registered embeddings in SQLite DB.
        
        Args:
            query_embedding (np.ndarray): 576-D or N-D normalized float vector
            threshold (float): Similarity threshold (default: 0.60)
            
        Returns:
            dict: {
                'person_id': 1 or None,
                'name': 'Rahul' or 'Unknown',
                'registration_id': 'REV001' or 'N/A',
                'department': 'CSE' or 'N/A',
                'similarity': 0.94,
                'is_match': True/False
            }
        """
        if query_embedding is None:
            return {
                'person_id': None,
                'name': 'Unknown',
                'registration_id': 'N/A',
                'department': 'N/A',
                'similarity': 0.0,
                'is_match': False
            }

        target_threshold = threshold if threshold is not None else self.default_threshold
        registered_records = get_all_embeddings()

        if not registered_records:
            return {
                'person_id': None,
                'name': 'Unknown',
                'registration_id': 'N/A',
                'department': 'N/A',
                'similarity': 0.0,
                'is_match': False
            }

        # Aggregate similarity scores per registered person
        person_scores = {}

        for rec in registered_records:
            stored_emb = np.array(rec['embedding'], dtype=np.float32)
            
            # Cosine similarity for L2 normalized vectors is dot product
            dot_product = np.dot(query_embedding, stored_emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_s = np.linalg.norm(stored_emb)
            
            if norm_q > 0 and norm_s > 0:
                sim = float(dot_product / (norm_q * norm_s))
            else:
                sim = 0.0

            pid = rec['person_id']
            if pid not in person_scores:
                person_scores[pid] = {
                    'person_id': pid,
                    'name': rec['name'],
                    'registration_id': rec['registration_id'],
                    'department': rec['department'],
                    'max_similarity': sim,
                    'scores': [sim]
                }
            else:
                person_scores[pid]['scores'].append(sim)
                if sim > person_scores[pid]['max_similarity']:
                    person_scores[pid]['max_similarity'] = sim

        # Find person with highest similarity score
        best_match = None
        best_sim = -1.0

        for pid, pdata in person_scores.items():
            if pdata['max_similarity'] > best_sim:
                best_sim = pdata['max_similarity']
                best_match = pdata

        if best_match and best_sim >= target_threshold:
            return {
                'person_id': best_match['person_id'],
                'name': best_match['name'],
                'registration_id': best_match['registration_id'],
                'department': best_match['department'],
                'similarity': round(float(best_sim), 3),
                'is_match': True
            }
        else:
            return {
                'person_id': None,
                'name': 'Unknown',
                'registration_id': 'N/A',
                'department': 'N/A',
                'similarity': round(float(max(best_sim, 0.0)), 3),
                'is_match': False
            }
