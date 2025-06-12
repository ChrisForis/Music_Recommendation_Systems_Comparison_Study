"""
Μετρικές Αξιολόγησης για Συστήματα Σύστασης Μουσικής
===================================================

Αυτό το module περιλαμβάνει τις υλοποιήσεις των μετρικών αξιολόγησης
για την αξιολόγηση των αλγορίθμων σύστασης:

- Recall@K: Ποσοστό των επιθυμητών αντικειμένων στις K κορυφαίες προτάσεις
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: Ποσοστό χρηστών που έλαβαν τουλάχιστον μία σχετική σύσταση
- MRR: Mean Reciprocal Rank

Βιβλιογραφικές Αναφορές:
- Jarvelin, K. & Kekäläinen, J. (2002): "Cumulated gain-based evaluation of IR techniques"  
- Herlocker, J. L., Konstan, J. A., Borchers, A. & Riedl, J. (1999): "An algorithmic framework for performing collaborative filtering" 
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Εισαγωγή του logging συστήματος
from logger import get_logger


class RecommendationEvaluator:
    """
    Κλάση για την αξιολόγηση συστημάτων σύστασης
    
    Παρέχει όλες τις απαραίτητες μετρικές για την αξιολόγηση της απόδοσης
    των αλγορίθμων σύστασης μουσικής, συμπεριλαμβανομένων των Recall@K,
    NDCG@K, Hit Rate@K και MRR.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        """
        Αρχικοποίηση του αξιολογητή
        
        Args:
            k_values (List[int]): Λίστα με τιμές K για τις μετρικές @K
        """
        self.k_values = k_values
        self.results = {}
        self.logger = get_logger()
        self.logger.info(f"Αρχικοποίηση RecommendationEvaluator με K values: {k_values}")
    
    def evaluate_model(self, model, test_matrix: csr_matrix, train_matrix: csr_matrix,
                      model_name: str, **kwargs) -> Dict[str, float]:
        """
        Αξιολόγηση ενός μοντέλου σύστασης
        
        Args:
            model: Το μοντέλο σύστασης προς αξιολόγηση
            test_matrix (csr_matrix): Πίνακας δεδομένων αξιολόγησης
            train_matrix (csr_matrix): Πίνακας δεδομένων εκπαίδευσης
            model_name (str): Όνομα του μοντέλου
            **kwargs: Επιπλέον παράμετροι για το μοντέλο
            
        Returns:
            Dict[str, float]: Λεξικό με τις μετρικές αξιολόγησης
        """
        with self.logger.timer(f"Αξιολόγηση μοντέλου: {model_name}"):
            self.logger.info(f"Έναρξη αξιολόγησης μοντέλου: {model_name}")
            
            # Συλλογή συστάσεων για όλους τους χρήστες
            all_recommendations = {}
            all_ground_truth = {}
            
            n_users = test_matrix.shape[0]
            max_k = max(self.k_values)
            
            self.logger.info(f"Αξιολόγηση {n_users} χρηστών με max_k={max_k}")
            
            # Δημιουργία progress bar
            progress_bar = self.logger.create_progress_bar(
                f"eval_{model_name}", 
                n_users, 
                f"Αξιολόγηση {model_name}"
            )
            
            successful_evaluations = 0
            failed_evaluations = 0
            
            for user_id in range(n_users):
                # Βρες τα πραγματικά αντικείμενα στο test set
                true_items = set(test_matrix[user_id].nonzero()[1])
                
                if len(true_items) > 0:  # Μόνο αν ο χρήστης έχει αντικείμενα στο test set
                    try:
                        # Λήψη συστάσεων από το μοντέλο
                        if hasattr(model, 'recommend'):
                            # Έλεγχος αν το μοντέλο είναι LLM-based και χρειάζεται επιπλέον παραμέτρους
                            if model_name in ['BERT', 'ZeroShot'] and 'artists_df' in kwargs:
                                recommendations = model.recommend(
                                    user_id, n_recommendations=max_k, 
                                    exclude_seen=True, 
                                    interaction_matrix=kwargs.get('interaction_matrix', train_matrix),
                                    artists_df=kwargs.get('artists_df'),
                                    tags_df=kwargs.get('tags_df'),
                                    user_taggedartists_df=kwargs.get('user_taggedartists_df')
                                )
                            elif 'interaction_matrix' in kwargs:
                                recommendations = model.recommend(
                                    user_id, n_recommendations=max_k, 
                                    exclude_seen=True, 
                                    interaction_matrix=kwargs['interaction_matrix']
                                )
                            else:
                                recommendations = model.recommend(
                                    user_id, n_recommendations=max_k, 
                                    exclude_seen=True
                                )
                        else:
                            # Fallback για μοντέλα χωρίς recommend method
                            recommendations = self._get_recommendations_fallback(
                                model, user_id, max_k, train_matrix
                            )
                        
                        # Εξαγωγή μόνο των item IDs
                        recommended_items = [item_id for item_id, score in recommendations]
                        
                        all_recommendations[user_id] = recommended_items
                        all_ground_truth[user_id] = list(true_items)
                        successful_evaluations += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Σφάλμα στη σύσταση για χρήστη {user_id}: {e}")
                        failed_evaluations += 1
                
                # Ενημέρωση progress bar
                if progress_bar:
                    progress_bar.update(1)
                elif user_id % 100 == 0:
                    self.logger.info(f"Επεξεργασία χρήστη {user_id}/{n_users}")
            
            self.logger.close_progress_bar(f"eval_{model_name}")
            self.logger.info(f"Επιτυχείς αξιολογήσεις: {successful_evaluations}")
            self.logger.info(f"Αποτυχημένες αξιολογήσεις: {failed_evaluations}")
            
            # Υπολογισμός μετρικών
            self.logger.info("Υπολογισμός μετρικών αξιολόγησης...")
            metrics = self._compute_all_metrics(all_recommendations, all_ground_truth)
            
            # Αποθήκευση αποτελεσμάτων
            self.results[model_name] = metrics
            
            # Καταγραφή αποτελεσμάτων
            self.logger.log_evaluation_results(model_name, metrics)
            
            return metrics
    
    def _get_recommendations_fallback(self, model, user_id: int, n_recommendations: int,
                                    train_matrix: csr_matrix) -> List[Tuple[int, float]]:
        """
        Fallback μέθοδος για λήψη συστάσεων από μοντέλα χωρίς recommend method
        
        Args:
            model: Το μοντέλο σύστασης
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            train_matrix (csr_matrix): Πίνακας εκπαίδευσης
            
        Returns:
            List[Tuple[int, float]]: Λίστα συστάσεων
        """
        # Για μοντέλα που έχουν μόνο predict method
        if hasattr(model, 'predict'):
            seen_items = set(train_matrix[user_id].nonzero()[1])
            all_items = set(range(train_matrix.shape[1]))
            candidate_items = list(all_items - seen_items)
            
            if len(candidate_items) == 0:
                return []
            
            # Πρόβλεψη scores για όλα τα candidate items
            user_ids = [user_id] * len(candidate_items)
            scores = model.predict(user_ids, candidate_items)
            
            # Ταξινόμηση και επιστροφή top-N
            item_scores = list(zip(candidate_items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:n_recommendations]
        
        return []
    
    def _compute_all_metrics(self, recommendations: Dict[int, List[int]], 
                           ground_truth: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Υπολογισμός όλων των μετρικών αξιολόγησης
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            
        Returns:
            Dict[str, float]: Λεξικό με όλες τις μετρικές
        """
        metrics = {}
        
        # Υπολογισμός μετρικών για κάθε K
        for k in self.k_values:
            recall_k = self._compute_recall_at_k(recommendations, ground_truth, k)
            ndcg_k = self._compute_ndcg_at_k(recommendations, ground_truth, k)
            hit_rate_k = self._compute_hit_rate_at_k(recommendations, ground_truth, k)
            
            metrics[f'Recall@{k}'] = recall_k
            metrics[f'NDCG@{k}'] = ndcg_k
            metrics[f'Hit_Rate@{k}'] = hit_rate_k
        
        # Υπολογισμός MRR
        mrr = self._compute_mrr(recommendations, ground_truth)
        metrics['MRR'] = mrr
        
        # Υπολογισμός Coverage
        coverage = self._compute_coverage(recommendations, ground_truth)
        metrics['Coverage'] = coverage
        
        return metrics
    
    def _compute_recall_at_k(self, recommendations: Dict[int, List[int]], 
                           ground_truth: Dict[int, List[int]], k: int) -> float:
        """
        Υπολογισμός Recall@K
        
        Το Recall@K μετρά το ποσοστό των σχετικών αντικειμένων που βρίσκονται
        στις πρώτες K συστάσεις.
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            k (int): Αριθμός κορυφαίων συστάσεων
            
        Returns:
            float: Recall@K score
        """
        total_recall = 0.0
        valid_users = 0
        
        for user_id in recommendations:
            if user_id in ground_truth:
                recommended_k = recommendations[user_id][:k]
                relevant_items = set(ground_truth[user_id])
                
                if len(relevant_items) > 0:
                    hits = len(set(recommended_k) & relevant_items)
                    recall = hits / len(relevant_items)
                    total_recall += recall
                    valid_users += 1
        
        return total_recall / valid_users if valid_users > 0 else 0.0
    
    def _compute_ndcg_at_k(self, recommendations: Dict[int, List[int]], 
                         ground_truth: Dict[int, List[int]], k: int) -> float:
        """
        Υπολογισμός NDCG@K (Normalized Discounted Cumulative Gain)
        
        Το NDCG@K λαμβάνει υπόψη τη θέση των σχετικών αντικειμένων στη λίστα
        συστάσεων, δίνοντας μεγαλύτερο βάρος στα αντικείμενα που εμφανίζονται
        νωρίτερα στη λίστα.
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            k (int): Αριθμός κορυφαίων συστάσεων
            
        Returns:
            float: NDCG@K score
        """
        total_ndcg = 0.0
        valid_users = 0
        
        for user_id in recommendations:
            if user_id in ground_truth:
                recommended_k = recommendations[user_id][:k]
                relevant_items = set(ground_truth[user_id])
                
                if len(relevant_items) > 0:
                    # Υπολογισμός DCG
                    dcg = 0.0
                    for i, item_id in enumerate(recommended_k):
                        if item_id in relevant_items:
                            dcg += 1.0 / np.log2(i + 2)  # i+2 γιατί το log2(1) = 0
                    
                    # Υπολογισμός IDCG (Ideal DCG)
                    idcg = 0.0
                    for i in range(min(len(relevant_items), k)):
                        idcg += 1.0 / np.log2(i + 2)
                    
                    # Υπολογισμός NDCG
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    total_ndcg += ndcg
                    valid_users += 1
        
        return total_ndcg / valid_users if valid_users > 0 else 0.0
    
    def _compute_hit_rate_at_k(self, recommendations: Dict[int, List[int]], 
                             ground_truth: Dict[int, List[int]], k: int) -> float:
        """
        Υπολογισμός Hit Rate@K
        
        Το Hit Rate@K μετρά το ποσοστό των χρηστών που έλαβαν τουλάχιστον
        μία σχετική σύσταση στις πρώτες K προτάσεις.
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            k (int): Αριθμός κορυφαίων συστάσεων
            
        Returns:
            float: Hit Rate@K score
        """
        hits = 0
        total_users = 0
        
        for user_id in recommendations:
            if user_id in ground_truth:
                recommended_k = recommendations[user_id][:k]
                relevant_items = set(ground_truth[user_id])
                
                if len(relevant_items) > 0:
                    if len(set(recommended_k) & relevant_items) > 0:
                        hits += 1
                    total_users += 1
        
        return hits / total_users if total_users > 0 else 0.0
    
    def _compute_mrr(self, recommendations: Dict[int, List[int]], 
                   ground_truth: Dict[int, List[int]]) -> float:
        """
        Υπολογισμός MRR (Mean Reciprocal Rank)
        
        Το MRR μετρά τον μέσο όρο του αντίστροφου της θέσης της πρώτης
        σχετικής σύστασης για κάθε χρήστη.
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            
        Returns:
            float: MRR score
        """
        total_rr = 0.0
        valid_users = 0
        
        for user_id in recommendations:
            if user_id in ground_truth:
                recommended_items = recommendations[user_id]
                relevant_items = set(ground_truth[user_id])
                
                if len(relevant_items) > 0:
                    # Βρες τη θέση της πρώτης σχετικής σύστασης
                    for i, item_id in enumerate(recommended_items):
                        if item_id in relevant_items:
                            total_rr += 1.0 / (i + 1)  # Reciprocal rank
                            break
                    valid_users += 1
        
        return total_rr / valid_users if valid_users > 0 else 0.0
    
    def _compute_coverage(self, recommendations: Dict[int, List[int]], 
                        ground_truth: Dict[int, List[int]]) -> float:
        """
        Υπολογισμός Coverage (κάλυψη καταλόγου)
        
        Το Coverage μετρά το ποσοστό των μοναδικών αντικειμένων που
        συστήνονται σε σχέση με το συνολικό αριθμό διαθέσιμων αντικειμένων.
        
        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            
        Returns:
            float: Coverage score
        """
        # Συλλογή όλων των συστημένων αντικειμένων
        all_recommended = set()
        for user_recommendations in recommendations.values():
            all_recommended.update(user_recommendations)
        
        # Συλλογή όλων των διαθέσιμων αντικειμένων
        all_items = set()
        for user_items in ground_truth.values():
            all_items.update(user_items)
        for user_recommendations in recommendations.values():
            all_items.update(user_recommendations)
        
        return len(all_recommended) / len(all_items) if len(all_items) > 0 else 0.0
    
    def _print_results(self, model_name: str, metrics: Dict[str, float]):
        """
        Εκτύπωση αποτελεσμάτων αξιολόγησης
        
        Args:
            model_name (str): Όνομα του μοντέλου
            metrics (Dict[str, float]): Μετρικές αξιολόγησης
        """
        print(f"\n{'='*50}")
        print(f"Αποτελέσματα για {model_name}")
        print(f"{'='*50}")
        
        # Ομαδοποίηση μετρικών
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith('Recall')}
        ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('NDCG')}
        hit_rate_metrics = {k: v for k, v in metrics.items() if k.startswith('Hit_Rate')}
        other_metrics = {k: v for k, v in metrics.items() 
                        if not any(k.startswith(prefix) for prefix in ['Recall', 'NDCG', 'Hit_Rate'])}
        
        # Εκτύπωση Recall
        print("\nRecall@K:")
        for metric, value in recall_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Εκτύπωση NDCG
        print("\nNDCG@K:")
        for metric, value in ndcg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Εκτύπωση Hit Rate
        print("\nHit Rate@K:")
        for metric, value in hit_rate_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Εκτύπωση άλλων μετρικών
        print("\nΆλλες Μετρικές:")
        for metric, value in other_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Σύγκριση όλων των αξιολογημένων μοντέλων
        
        Returns:
            pd.DataFrame: DataFrame με τα αποτελέσματα όλων των μοντέλων
        """
        if not self.results:
            print("Δεν υπάρχουν αποτελέσματα για σύγκριση.")
            return pd.DataFrame()
        
        # Δημιουργία DataFrame με τα αποτελέσματα
        comparison_df = pd.DataFrame(self.results).T
        
        print("\n" + "="*80)
        print("ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ")
        print("="*80)
        print(comparison_df.round(4))
        
        # Εύρεση καλύτερου μοντέλου για κάθε μετρική
        print("\n" + "="*80)
        print("ΚΑΛΥΤΕΡΑ ΜΟΝΤΕΛΑ ΑΝΑ ΜΕΤΡΙΚΗ")
        print("="*80)
        
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric}: {best_model} ({best_score:.4f})")
        
        return comparison_df
    
    def save_results(self, filename: str):
        """
        Αποθήκευση αποτελεσμάτων σε αρχείο
        
        Args:
            filename (str): Όνομα αρχείου για αποθήκευση
        """
        if not self.results:
            print("Δεν υπάρχουν αποτελέσματα για αποθήκευση.")
            return
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df.to_csv(filename)
        print(f"Αποτελέσματα αποθηκεύτηκαν στο αρχείο: {filename}")


def evaluate_recommendation_diversity(recommendations: Dict[int, List[int]], 
                                    interaction_matrix: csr_matrix) -> Dict[str, float]:
    """
    Αξιολόγηση της ποικιλομορφίας των συστάσεων
    
    Υπολογίζει μετρικές που σχετίζονται με την ποικιλομορφία και την
    καινοτομία των συστάσεων.
    
    Args:
        recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
        interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        
    Returns:
        Dict[str, float]: Μετρικές ποικιλομορφίας
    """
    # Υπολογισμός δημοφιλίας αντικειμένων
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    total_interactions = interaction_matrix.sum()
    item_popularity_normalized = item_popularity / total_interactions
    
    # Υπολογισμός μέσης δημοφιλίας συστάσεων (χαμηλότερη = πιο διαφοροποιημένη)
    total_novelty = 0.0
    total_recommendations = 0
    
    for user_recommendations in recommendations.values():
        for item_id in user_recommendations:
            if item_id < len(item_popularity_normalized):
                # Novelty = -log(popularity)
                novelty = -np.log(item_popularity_normalized[item_id] + 1e-10)
                total_novelty += novelty
                total_recommendations += 1
    
    avg_novelty = total_novelty / total_recommendations if total_recommendations > 0 else 0.0
    
    # Υπολογισμός intra-list diversity (μέση ποικιλομορφία εντός λίστας)
    # Για απλότητα, χρησιμοποιούμε τον αριθμό μοναδικών αντικειμένων ανά χρήστη
    total_diversity = 0.0
    valid_users = 0
    
    for user_recommendations in recommendations.values():
        if len(user_recommendations) > 0:
            unique_items = len(set(user_recommendations))
            diversity = unique_items / len(user_recommendations)
            total_diversity += diversity
            valid_users += 1
    
    avg_diversity = total_diversity / valid_users if valid_users > 0 else 0.0
    
    return {
        'Novelty': avg_novelty,
        'Intra_List_Diversity': avg_diversity
    } 