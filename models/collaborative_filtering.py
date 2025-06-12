"""
Κλασικοί Αλγόριθμοι Collaborative Filtering
==========================================

Αυτό το module περιλαμβάνει τις υλοποιήσεις των κλασικών αλγορίθμων
συνεργατικού φιλτραρίσματος για σύσταση μουσικής:

1. UserKNN: Αλγόριθμος βασισμένος στην ομοιότητα χρηστών
2. ItemKNN: Αλγόριθμος βασισμένος στην ομοιότητα αντικειμένων
3. SVD: Παραγοντοποίηση πίνακα με Singular Value Decomposition

Βιβλιογραφικές Αναφορές:
- Linden et al. (2003): "Amazon.com recommendations: item-to-item collaborative filtering"
  IEEE Internet Computing, Vol. 7, No. 1, pp. 76-80. DOI: 10.1109/MIC.2003.1167344
- Koren et al. (2009): "Matrix Factorization Techniques for Recommender Systems"
  Computer, Vol. 42, No. 8, pp. 30-37. DOI: 10.1109/MC.2009.263
- Su & Chang (2017): "Effective social content-based collaborative filtering for music recommendation"
  Intelligent Data Analysis, Vol. 21, pp. S195-S216. DOI: 10.3233/IDA-170878
- Bellogín et al. (2010): "A study of heterogeneity in recommendations for a social music service"
  RecSys '10: Fourth ACM Conference on Recommender Systems, pp. 1-8. DOI: 10.1145/1869446.1869447
- Majumdar (2013): "Music Recommendations based on Implicit Feedback and Social Circles: The Last FM Data Set"
  (Σχετικό με το dataset που χρησιμοποιούμε και collaborative filtering για Last.fm)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class BaseRecommender:
    """
    Βασική κλάση για όλους τους αλγόριθμους σύστασης
    
    Παρέχει κοινές λειτουργίες και interface για όλους τους αλγόριθμους
    συνεργατικού φιλτραρίσματος.
    """
    
    def __init__(self):
        """Αρχικοποίηση του βασικού συστήματος σύστασης"""
        self.is_fitted = False
        self.user_mean = None
        self.item_mean = None
        self.global_mean = None
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του μοντέλου
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων χρηστών-αντικειμένων
        """
        raise NotImplementedError("Η μέθοδος fit πρέπει να υλοποιηθεί από τις υποκλάσεις")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Πρόβλεψη βαθμολογιών για συγκεκριμένα ζεύγη χρήστη-αντικειμένου
        
        Args:
            user_ids (List[int]): Λίστα με IDs χρηστών
            item_ids (List[int]): Λίστα με IDs αντικειμένων
            
        Returns:
            np.ndarray: Προβλεπόμενες βαθμολογίες
        """
        raise NotImplementedError("Η μέθοδος predict πρέπει να υλοποιηθεί από τις υποκλάσεις")
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score) ταξινομημένη κατά φθίνουσα σειρά
        """
        raise NotImplementedError("Η μέθοδος recommend πρέπει να υλοποιηθεί από τις υποκλάσεις")


class UserKNN(BaseRecommender):
    """
    Αλγόριθμος User-based K-Nearest Neighbors
    
    Ο αλγόριθμος UserKNN βρίσκει τους πιο παρόμοιους χρήστες με τον στόχο
    και κάνει συστάσεις βασισμένες στις προτιμήσεις αυτών των χρηστών.
    Χρησιμοποιεί το cosine similarity για τον υπολογισμό της ομοιότητας.
    
    Βιβλιογραφικές Αναφορές:
    - Bellogín et al. (2010) "A study of heterogeneity in recommendations for a social music service"
      μελετά collaborative filtering σε music services όπως το Last.fm.
    - Majumdar (2013) "Music Recommendations based on Implicit Feedback and Social Circles"  
      περιγράφει τη χρήση του Last.fm dataset για collaborative filtering.
    """
    
    def __init__(self, k: int = 50, similarity_metric: str = 'cosine'):
        """
        Αρχικοποίηση του UserKNN
        
        Args:
            k (int): Αριθμός πλησιέστερων γειτόνων
            similarity_metric (str): Μετρική ομοιότητας ('cosine', 'pearson')
        """
        super().__init__()
        self.k = k
        self.similarity_metric = similarity_metric
        self.user_similarity_matrix = None
        self.interaction_matrix = None
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του UserKNN μοντέλου
        
        Υπολογίζει τη μήτρα ομοιότητας μεταξύ όλων των χρηστών
        χρησιμοποιώντας την επιλεγμένη μετρική.
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        """
        self.interaction_matrix = interaction_matrix
        
        # Υπολογισμός μέσων όρων
        self.global_mean = interaction_matrix.data.mean()
        self.user_mean = np.array(interaction_matrix.mean(axis=1)).flatten()
        
        # Υπολογισμός ομοιότητας χρηστών
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(interaction_matrix)
        elif self.similarity_metric == 'pearson':
            # Κεντράρισμα των δεδομένων για Pearson correlation
            user_mean_matrix = interaction_matrix.copy().astype(float)
            for i in range(interaction_matrix.shape[0]):
                user_interactions = interaction_matrix[i].nonzero()[1]
                if len(user_interactions) > 0:
                    user_mean_matrix[i, user_interactions] -= self.user_mean[i]
            
            self.user_similarity_matrix = cosine_similarity(user_mean_matrix)
        
        # Αφαίρεση αυτο-ομοιότητας (διαγώνιος = 0)
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        self.is_fitted = True
        print(f"UserKNN εκπαιδεύτηκε με k={self.k}, similarity={self.similarity_metric}")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Πρόβλεψη βαθμολογιών χρησιμοποιώντας τους k πλησιέστερους χρήστες
        
        Args:
            user_ids (List[int]): Λίστα με IDs χρηστών
            item_ids (List[int]): Λίστα με IDs αντικειμένων
            
        Returns:
            np.ndarray: Προβλεπόμενες βαθμολογίες
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Βρες τους k πλησιέστερους χρήστες
            user_similarities = self.user_similarity_matrix[user_id]
            top_k_users = np.argsort(user_similarities)[-self.k:]
            top_k_similarities = user_similarities[top_k_users]
            
            # Φιλτράρισμα χρηστών που έχουν αλληλεπιδράσει με το αντικείμενο
            valid_users = []
            valid_similarities = []
            
            for i, similar_user in enumerate(top_k_users):
                if self.interaction_matrix[similar_user, item_id] > 0:
                    valid_users.append(similar_user)
                    valid_similarities.append(top_k_similarities[i])
            
            if len(valid_users) == 0:
                # Αν δεν υπάρχουν παρόμοιοι χρήστες, χρησιμοποίησε τον μέσο όρο
                prediction = self.user_mean[user_id] if self.user_mean[user_id] > 0 else self.global_mean
            else:
                # Υπολογισμός σταθμισμένου μέσου όρου
                valid_similarities = np.array(valid_similarities)
                valid_ratings = []
                
                for similar_user in valid_users:
                    rating = self.interaction_matrix[similar_user, item_id]
                    valid_ratings.append(rating)
                
                valid_ratings = np.array(valid_ratings)
                
                if np.sum(np.abs(valid_similarities)) > 0:
                    prediction = np.sum(valid_similarities * valid_ratings) / np.sum(np.abs(valid_similarities))
                else:
                    prediction = self.user_mean[user_id] if self.user_mean[user_id] > 0 else self.global_mean
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        # Βρες αντικείμενα που δεν έχει δει ο χρήστης
        seen_items = set(self.interaction_matrix[user_id].nonzero()[1])
        all_items = set(range(self.interaction_matrix.shape[1]))
        
        if exclude_seen:
            candidate_items = list(all_items - seen_items)
        else:
            candidate_items = list(all_items)
        
        if len(candidate_items) == 0:
            return []
        
        # Υπολογισμός scores για όλα τα candidate items
        user_ids = [user_id] * len(candidate_items)
        scores = self.predict(user_ids, candidate_items)
        
        # Ταξινόμηση και επιστροφή top-N
        item_scores = list(zip(candidate_items, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n_recommendations]


class ItemKNN(BaseRecommender):
    """
    Αλγόριθμος Item-based K-Nearest Neighbors
    
    Ο αλγόριθμος ItemKNN βρίσκει τα πιο παρόμοια αντικείμενα με αυτά που
    έχει ακούσει ο χρήστης και κάνει συστάσεις βασισμένες σε αυτή την ομοιότητα.
    
    Βιβλιογραφική Αναφορά:
    Linden et al. (2003) "Amazon.com recommendations: item-to-item collaborative filtering"
    περιγράφει αναλυτικά τη μεθοδολογία item-to-item collaborative filtering.
    """
    
    def __init__(self, k: int = 50, similarity_metric: str = 'cosine'):
        """
        Αρχικοποίηση του ItemKNN
        
        Args:
            k (int): Αριθμός πλησιέστερων γειτόνων
            similarity_metric (str): Μετρική ομοιότητας
        """
        super().__init__()
        self.k = k
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
        self.interaction_matrix = None
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του ItemKNN μοντέλου
        
        Υπολογίζει τη μήτρα ομοιότητας μεταξύ όλων των αντικειμένων.
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        """
        self.interaction_matrix = interaction_matrix
        
        # Υπολογισμός μέσων όρων
        self.global_mean = interaction_matrix.data.mean()
        self.item_mean = np.array(interaction_matrix.mean(axis=0)).flatten()
        
        # Μετατροπή σε item-user matrix για υπολογισμό ομοιότητας αντικειμένων
        item_user_matrix = interaction_matrix.T
        
        # Υπολογισμός ομοιότητας αντικειμένων
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        elif self.similarity_metric == 'pearson':
            # Κεντράρισμα για Pearson correlation
            item_mean_matrix = item_user_matrix.copy().astype(float)
            for i in range(item_user_matrix.shape[0]):
                item_interactions = item_user_matrix[i].nonzero()[1]
                if len(item_interactions) > 0:
                    item_mean_matrix[i, item_interactions] -= self.item_mean[i]
            
            self.item_similarity_matrix = cosine_similarity(item_mean_matrix)
        
        # Αφαίρεση αυτο-ομοιότητας
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        self.is_fitted = True
        print(f"ItemKNN εκπαιδεύτηκε με k={self.k}, similarity={self.similarity_metric}")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Πρόβλεψη βαθμολογιών χρησιμοποιώντας τα k πλησιέστερα αντικείμενα
        
        Args:
            user_ids (List[int]): Λίστα με IDs χρηστών
            item_ids (List[int]): Λίστα με IDs αντικειμένων
            
        Returns:
            np.ndarray: Προβλεπόμενες βαθμολογίες
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Βρες τα k πλησιέστερα αντικείμενα
            item_similarities = self.item_similarity_matrix[item_id]
            top_k_items = np.argsort(item_similarities)[-self.k:]
            top_k_similarities = item_similarities[top_k_items]
            
            # Φιλτράρισμα αντικειμένων που έχει αξιολογήσει ο χρήστης
            valid_items = []
            valid_similarities = []
            
            for i, similar_item in enumerate(top_k_items):
                if self.interaction_matrix[user_id, similar_item] > 0:
                    valid_items.append(similar_item)
                    valid_similarities.append(top_k_similarities[i])
            
            if len(valid_items) == 0:
                # Αν δεν υπάρχουν παρόμοια αντικείμενα, χρησιμοποίησε μέσο όρο
                prediction = self.item_mean[item_id] if self.item_mean[item_id] > 0 else self.global_mean
            else:
                # Υπολογισμός σταθμισμένου μέσου όρου
                valid_similarities = np.array(valid_similarities)
                valid_ratings = []
                
                for similar_item in valid_items:
                    rating = self.interaction_matrix[user_id, similar_item]
                    valid_ratings.append(rating)
                
                valid_ratings = np.array(valid_ratings)
                
                if np.sum(np.abs(valid_similarities)) > 0:
                    prediction = np.sum(valid_similarities * valid_ratings) / np.sum(np.abs(valid_similarities))
                else:
                    prediction = self.item_mean[item_id] if self.item_mean[item_id] > 0 else self.global_mean
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        # Βρες αντικείμενα που δεν έχει δει ο χρήστης
        seen_items = set(self.interaction_matrix[user_id].nonzero()[1])
        all_items = set(range(self.interaction_matrix.shape[1]))
        
        if exclude_seen:
            candidate_items = list(all_items - seen_items)
        else:
            candidate_items = list(all_items)
        
        if len(candidate_items) == 0:
            return []
        
        # Υπολογισμός scores για όλα τα candidate items
        user_ids = [user_id] * len(candidate_items)
        scores = self.predict(user_ids, candidate_items)
        
        # Ταξινόμηση και επιστροφή top-N
        item_scores = list(zip(candidate_items, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n_recommendations]


class SVDRecommender(BaseRecommender):
    """
    Αλγόριθμος Matrix Factorization με Singular Value Decomposition
    
    Ο αλγόριθμος SVD παραγοντοποιεί τον πίνακα αλληλεπιδράσεων σε δύο
    χαμηλής διάστασης πίνακες που αντιπροσωπεύουν τα latent factors
    των χρηστών και των αντικειμένων.
    
    Βιβλιογραφική Αναφορά:
    Koren et al. (2009) "Matrix Factorization Techniques for Recommender Systems"
    περιγράφει τις τεχνικές παραγοντοποίησης πίνακα για συστήματα σύστασης.
    """
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        """
        Αρχικοποίηση του SVD Recommender
        
        Args:
            n_factors (int): Αριθμός latent factors
            random_state (int): Seed για αναπαραγωγιμότητα
        """
        super().__init__()
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.interaction_matrix = None
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του SVD μοντέλου
        
        Εκτελεί την παραγοντοποίηση του πίνακα αλληλεπιδράσεων.
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        """
        self.interaction_matrix = interaction_matrix
        
        # Υπολογισμός μέσων όρων
        self.global_mean = interaction_matrix.data.mean()
        
        # Εκτέλεση SVD
        self.user_factors = self.svd.fit_transform(interaction_matrix)
        self.item_factors = self.svd.components_.T
        
        self.is_fitted = True
        print(f"SVD εκπαιδεύτηκε με {self.n_factors} factors")
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Πρόβλεψη βαθμολογιών χρησιμοποιώντας τα latent factors
        
        Args:
            user_ids (List[int]): Λίστα με IDs χρηστών
            item_ids (List[int]): Λίστα με IDs αντικειμένων
            
        Returns:
            np.ndarray: Προβλεπόμενες βαθμολογίες
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Υπολογισμός εσωτερικού γινομένου των latent factors
            user_vector = self.user_factors[user_id]
            item_vector = self.item_factors[item_id]
            prediction = np.dot(user_vector, item_vector)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        # Υπολογισμός scores για όλα τα αντικείμενα
        user_vector = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vector)
        
        # Βρες αντικείμενα που δεν έχει δει ο χρήστης
        if exclude_seen:
            seen_items = set(self.interaction_matrix[user_id].nonzero()[1])
            for item_id in seen_items:
                scores[item_id] = -np.inf
        
        # Ταξινόμηση και επιστροφή top-N
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations 