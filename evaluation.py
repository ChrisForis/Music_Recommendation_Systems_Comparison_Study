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
        
        # Παράμετροι γρήγορης αξιολόγησης
        self.fast_evaluation = False
        self.evaluation_sample_size = 1000
        self.adaptive_evaluation = False
        self.evaluation_patience = 50
    
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
            precision_k = self._compute_precision_at_k(recommendations, ground_truth, k)
            recall_k = self._compute_recall_at_k(recommendations, ground_truth, k)
            ndcg_k = self._compute_ndcg_at_k(recommendations, ground_truth, k)
            hit_rate_k = self._compute_hit_rate_at_k(recommendations, ground_truth, k)

            metrics[f'Precision@{k}'] = precision_k
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
    
    def smart_user_sampling(self, test_matrix: csr_matrix, sample_size: int = 1000) -> np.ndarray:
        """
        Έξυπνο sampling χρηστών για γρήγορη αξιολόγηση
        
        Args:
            test_matrix (csr_matrix): Πίνακας test
            sample_size (int): Μέγεθος δείγματος
            
        Returns:
            np.ndarray: Επιλεγμένοι χρήστες
        """
        n_users = test_matrix.shape[0]
        
        if sample_size >= n_users:
            return np.arange(n_users)
        
        self.logger.info(f"Έναρξη intelligent sampling για {sample_size} χρήστες από {n_users}")
        
        # Υπολογισμός αλληλεπιδράσεων ανά χρήστη
        user_interactions = np.array(test_matrix.sum(axis=1)).flatten()
        
        # Διαχώριση σε κατηγορίες με βελτιωμένες οριακές τιμές
        low_interaction = user_interactions <= 2
        high_interaction = user_interactions > 15
        medium_interaction = ~(low_interaction | high_interaction)
        
        # Sampling διατηρώντας αναλογίες
        low_users = np.where(low_interaction)[0]
        high_users = np.where(high_interaction)[0]
        medium_users = np.where(medium_interaction)[0]
        
        # Υπολογισμός μεγεθών δείγματος με βελτιωμένη διαχείριση
        total_low = len(low_users)
        total_high = len(high_users)
        total_medium = len(medium_users)
        total_users = total_low + total_medium + total_high
        
        if total_users == 0:
            return np.random.choice(n_users, min(sample_size, n_users), replace=False)
        
        # Βελτιωμένη προσαρμογή μεγεθών δείγματος
        # Εγγυάται ελάχιστο δείγμα από κάθε κατηγορία
        min_sample_per_category = max(50, sample_size // 10)
        
        sample_low = min(max(min_sample_per_category, int(sample_size * total_low / total_users)), total_low)
        sample_high = min(max(min_sample_per_category, int(sample_size * total_high / total_users)), total_high)
        sample_medium = sample_size - sample_low - sample_high
        
        # Διόρθωση αν υπάρχει υπερβολή ή έλλειψη
        if sample_medium < 0:
            # Αν δεν έχουμε αρκετό χώρο για medium, μείωσε τα άλλα
            excess = abs(sample_medium)
            sample_low = max(min_sample_per_category, sample_low - excess // 2)
            sample_high = max(min_sample_per_category, sample_high - excess // 2)
            sample_medium = sample_size - sample_low - sample_high
        elif sample_medium > total_medium:
            # Αν έχουμε περισσότερο χώρο για medium, μείωσε το
            sample_medium = total_medium
            # Αναδιανόμησε το υπόλοιπο
            remaining = sample_size - sample_low - sample_high - sample_medium
            if remaining > 0:
                if total_low > sample_low:
                    sample_low += min(remaining // 2, total_low - sample_low)
                    remaining -= min(remaining // 2, total_low - sample_low)
                if remaining > 0 and total_high > sample_high:
                    sample_high += remaining
        
        # Δημιουργία progress bar για sampling
        total_sampling_steps = 3  # low + medium + high
        progress_bar = self.logger.create_progress_bar(
            "smart_sampling", 
            total_sampling_steps, 
            "Intelligent User Sampling"
        )
        
        # Sampling με βελτιωμένη διαχείριση σφαλμάτων
        selected_users = []
        
        # Sampling low interaction users
        if sample_low > 0 and len(low_users) > 0:
            selected_users.extend(np.random.choice(low_users, sample_low, replace=False))
            if progress_bar:
                progress_bar.update(1)
            self.logger.info(f"Sampling low interaction: {sample_low} χρήστες")
        
        # Sampling medium interaction users
        if sample_medium > 0 and len(medium_users) > 0:
            selected_users.extend(np.random.choice(medium_users, sample_medium, replace=False))
            if progress_bar:
                progress_bar.update(1)
            self.logger.info(f"Sampling medium interaction: {sample_medium} χρήστες")
        
        # Sampling high interaction users
        if sample_high > 0 and len(high_users) > 0:
            selected_users.extend(np.random.choice(high_users, sample_high, replace=False))
            if progress_bar:
                progress_bar.update(1)
            self.logger.info(f"Sampling high interaction: {sample_high} χρήστες")
        
        # Συμπλήρωση με τυχαίους χρήστες αν χρειάζεται
        if len(selected_users) < sample_size:
            remaining_users = np.setdiff1d(np.arange(n_users), selected_users)
            additional_needed = sample_size - len(selected_users)
            if len(remaining_users) > 0:
                additional_users = np.random.choice(remaining_users, 
                                                 min(additional_needed, len(remaining_users)), 
                                                 replace=False)
                selected_users.extend(additional_users)
                self.logger.info(f"Συμπλήρωση με τυχαίους χρήστες: {len(additional_users)}")
        
        # Κλείσιμο progress bar
        self.logger.close_progress_bar("smart_sampling")
        
        # Εγγύηση ότι έχουμε τουλάχιστον sample_size χρήστες
        if len(selected_users) < sample_size:
            # Αν δεν έχουμε αρκετούς, επέστρεψε όλους τους διαθέσιμους
            self.logger.warning(f"Δεν ήταν δυνατό να επιλεγούν {sample_size} χρήστες. Επιστρέφονται {len(selected_users)}")
            return np.array(selected_users)
        
        self.logger.info(f"Intelligent sampling ολοκληρώθηκε: {len(selected_users)} χρήστες")
        return np.array(selected_users[:sample_size])
    
    def print_sampling_statistics(self, test_matrix: csr_matrix, selected_users: np.ndarray):
        """
        Εκτύπωση στατιστικών για την ποιότητα του sampling
        
        Args:
            test_matrix (csr_matrix): Πίνακας test
            selected_users (np.ndarray): Επιλεγμένοι χρήστες
        """
        n_total_users = test_matrix.shape[0]
        n_selected_users = len(selected_users)
        
        # Στατιστικά για όλους τους χρήστες
        all_user_interactions = np.array(test_matrix.sum(axis=1)).flatten()
        all_mean_interactions = all_user_interactions.mean()
        all_std_interactions = all_user_interactions.std()
        
        # Στατιστικά για επιλεγμένους χρήστες
        selected_interactions = all_user_interactions[selected_users]
        selected_mean_interactions = selected_interactions.mean()
        selected_std_interactions = selected_interactions.std()
        
        # Υπολογισμός coverage
        total_interactions = test_matrix.nnz
        selected_interactions_sum = selected_interactions.sum()
        coverage_ratio = selected_interactions_sum / total_interactions
        
        print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ SAMPLING:")
        print(f"   • Συνολικοί χρήστες: {n_total_users}")
        print(f"   • Επιλεγμένοι χρήστες: {n_selected_users} ({n_selected_users/n_total_users*100:.1f}%)")
        print(f"   • Μέσες αλληλεπιδράσεις (όλοι): {all_mean_interactions:.2f} ± {all_std_interactions:.2f}")
        print(f"   • Μέσες αλληλεπιδράσεις (δείγμα): {selected_mean_interactions:.2f} ± {selected_std_interactions:.2f}")
        print(f"   • Coverage ratio: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
        
        # Έλεγχος αν το sampling είναι αντιπροσωπευτικό
        interaction_diff = abs(selected_mean_interactions - all_mean_interactions)
        if interaction_diff < all_std_interactions * 0.5:
            print(f"   ✅ Το sampling είναι αντιπροσωπευτικό (διαφορά: {interaction_diff:.2f})")
        else:
            print(f"   ⚠️ Το sampling μπορεί να μην είναι αντιπροσωπευτικό (διαφορά: {interaction_diff:.2f})")
    
    def adaptive_evaluation(self, model, test_matrix: csr_matrix, train_matrix: csr_matrix,
                          model_name: str, min_evaluations: int = 100, 
                          patience: int = 50, **kwargs) -> Dict[str, float]:
        """
        Προσαρμοστική αξιολόγηση με early stopping
        
        Args:
            model: Το μοντέλο προς αξιολόγηση
            test_matrix (csr_matrix): Πίνακας test
            train_matrix (csr_matrix): Πίνακας train
            model_name (str): Όνομα μοντέλου
            min_evaluations (int): Ελάχιστος αριθμός αξιολογήσεων
            patience (int): Αριθμός χρηστών χωρίς βελτίωση
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Dict[str, float]: Μετρικές αξιολόγησης
        """
        n_users = test_matrix.shape[0]
        user_ids = np.random.permutation(n_users)
        
        all_recommendations = {}
        all_ground_truth = {}
        
        best_recall = 0
        best_ndcg = 0
        patience_counter = 0
        evaluation_count = 0
        
        # Βελτιωμένα thresholds για early stopping
        recall_threshold = 0.001  # Βελτίωση τουλάχιστον 0.1%
        ndcg_threshold = 0.001
        
        self.logger.info(f"Προσαρμοστική αξιολόγηση για {model_name}")
        
        # Δημιουργία progress bar για adaptive evaluation
        progress_bar = self.logger.create_progress_bar(
            f"adaptive_eval_{model_name}", 
            n_users, 
            f"Προσαρμοστική αξιολόγηση {model_name}"
        )
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, user_id in enumerate(user_ids):
            true_items = set(test_matrix[user_id].nonzero()[1])
            
            if len(true_items) > 0:
                try:
                    # Λήψη συστάσεων
                    recommendations = self._get_recommendations_fallback(
                        model, user_id, max(self.k_values), train_matrix
                    )
                    
                    if recommendations:
                        rec_items = [item for item, _ in recommendations]
                        all_recommendations[user_id] = rec_items
                        all_ground_truth[user_id] = list(true_items)
                        
                        evaluation_count += 1
                        successful_evaluations += 1
                        
                        # Έλεγχος early stopping με βελτιωμένη λογική
                        if evaluation_count >= min_evaluations:
                            current_metrics = self._compute_all_metrics(
                                all_recommendations, all_ground_truth
                            )
                            current_recall = current_metrics.get('Recall@5', 0)
                            current_ndcg = current_metrics.get('NDCG@5', 0)
                            
                            # Έλεγχος βελτίωσης σε πολλαπλές μετρικές
                            recall_improved = current_recall > best_recall + recall_threshold
                            ndcg_improved = current_ndcg > best_ndcg + ndcg_threshold
                            
                            if recall_improved or ndcg_improved:
                                best_recall = max(best_recall, current_recall)
                                best_ndcg = max(best_ndcg, current_ndcg)
                                patience_counter = 0
                            else:
                                patience_counter += 1
                            
                            # Early stopping με βελτιωμένη συνθήκη
                            if patience_counter >= patience:
                                self.logger.info(f"Early stopping at {evaluation_count} χρήστες")
                                self.logger.info(f"Τελικές μετρικές: Recall@5={current_recall:.4f}, NDCG@5={current_ndcg:.4f}")
                                break
                        
                        # Έλεγχος αν έχουμε αρκετά δεδομένα για σταθερή εκτίμηση
                        if evaluation_count % 50 == 0 and evaluation_count >= min_evaluations:
                            # Υπολογισμός confidence interval (απλοποιημένος)
                            recent_metrics = self._compute_all_metrics(
                                all_recommendations, all_ground_truth
                            )
                            recent_recall = recent_metrics.get('Recall@5', 0)
                            
                            # Αν η βελτίωση είναι πολύ μικρή για πολλούς χρήστες, σταμάτα
                            if abs(recent_recall - best_recall) < recall_threshold * 2:
                                patience_counter += 2  # Επιτάχυνση early stopping
                    else:
                        failed_evaluations += 1
                
                except Exception as e:
                    self.logger.warning(f"Σφάλμα στην αξιολόγηση χρήστη {user_id}: {e}")
                    failed_evaluations += 1
                    continue
            
            # Ενημέρωση progress bar
            if progress_bar:
                progress_bar.update(1)
            elif i % 100 == 0:
                self.logger.info(f"Προσαρμοστική αξιολόγηση {model_name}: {i+1}/{n_users} χρήστες")
        
        # Κλείσιμο progress bar
        self.logger.close_progress_bar(f"adaptive_eval_{model_name}")
        
        self.logger.info(f"Προσαρμοστική αξιολόγηση {model_name} ολοκληρώθηκε:")
        self.logger.info(f"  • Επιτυχείς αξιολογήσεις: {successful_evaluations}")
        self.logger.info(f"  • Αποτυχημένες αξιολογήσεις: {failed_evaluations}")
        self.logger.info(f"  • Συνολικοί χρήστες που επεξεργάστηκαν: {i+1}")
        
        # Υπολογισμός τελικών μετρικών
        if all_recommendations:
            final_metrics = self._compute_all_metrics(all_recommendations, all_ground_truth)
            self.logger.info(f"Αξιολόγηση ολοκληρώθηκε με {evaluation_count} χρήστες")
            return final_metrics
        else:
            return {f'{metric}@{k}': 0.0 for metric in ['Recall', 'NDCG', 'Hit_Rate'] 
                   for k in self.k_values}
    
    def fast_evaluate_model(self, model, test_matrix: csr_matrix, train_matrix: csr_matrix,
                           model_name: str, **kwargs) -> Dict[str, float]:
        """
        Γρήγορη αξιολόγηση με sampling
        
        Args:
            model: Το μοντέλο προς αξιολόγηση
            test_matrix (csr_matrix): Πίνακας test
            train_matrix (csr_matrix): Πίνακας train
            model_name (str): Όνομα μοντέλου
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Dict[str, float]: Μετρικές αξιολόγησης
        """
        # Sampling χρηστών
        selected_users = self.smart_user_sampling(test_matrix, self.evaluation_sample_size)
        
        # Εμφάνιση στατιστικών sampling
        self.print_sampling_statistics(test_matrix, selected_users)
        
        self.logger.info(f"Γρήγορη αξιολόγηση {model_name} με {len(selected_users)} χρήστες")
        
        # Δημιουργία progress bar για γρήγορη αξιολόγηση
        progress_bar = self.logger.create_progress_bar(
            f"fast_eval_{model_name}", 
            len(selected_users), 
            f"Γρήγορη αξιολόγηση {model_name}"
        )
        
        all_recommendations = {}
        all_ground_truth = {}
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, user_id in enumerate(selected_users):
            true_items = set(test_matrix[user_id].nonzero()[1])
            
            if len(true_items) > 0:
                try:
                    recommendations = self._get_recommendations_fallback(
                        model, user_id, max(self.k_values), train_matrix
                    )
                    
                    if recommendations:
                        rec_items = [item for item, _ in recommendations]
                        all_recommendations[user_id] = rec_items
                        all_ground_truth[user_id] = list(true_items)
                        successful_evaluations += 1
                    else:
                        failed_evaluations += 1
                
                except Exception as e:
                    self.logger.warning(f"Σφάλμα στην αξιολόγηση χρήστη {user_id}: {e}")
                    failed_evaluations += 1
                    continue
            
            # Ενημέρωση progress bar
            if progress_bar:
                progress_bar.update(1)
            elif i % 50 == 0:
                self.logger.info(f"Γρήγορη αξιολόγηση {model_name}: {i+1}/{len(selected_users)} χρήστες")
        
        # Κλείσιμο progress bar
        self.logger.close_progress_bar(f"fast_eval_{model_name}")
        
        self.logger.info(f"Γρήγορη αξιολόγηση {model_name} ολοκληρώθηκε:")
        self.logger.info(f"  • Επιτυχείς αξιολογήσεις: {successful_evaluations}")
        self.logger.info(f"  • Αποτυχημένες αξιολογήσεις: {failed_evaluations}")
        
        # Υπολογισμός μετρικών
        if all_recommendations:
            return self._compute_all_metrics(all_recommendations, all_ground_truth)
        else:
            return {f'{metric}@{k}': 0.0 for metric in ['Recall', 'NDCG', 'Hit_Rate'] 
                   for k in self.k_values}

    def _compute_precision_at_k(self, recommendations: Dict[int, List[int]], 
                                 ground_truth: Dict[int, List[int]], k: int) -> float:
        """
        Υπολογισμός Precision@K

        Το Precision@K μετρά το ποσοστό των σχετικών αντικειμένων που βρίσκονται
        στις πρώτες K συστάσεις.

        Args:
            recommendations (Dict[int, List[int]]): Συστάσεις ανά χρήστη
            ground_truth (Dict[int, List[int]]): Πραγματικά αντικείμενα ανά χρήστη
            k (int): Αριθμός κορυφαίων συστάσεων

        Returns:
            float: Precision@K score
        """
        total_precision = 0.0
        valid_users = 0

        for user_id in recommendations:
            if user_id in ground_truth:
                recommended_k = recommendations[user_id][:k]
                relevant_items = set(ground_truth[user_id])

                if len(relevant_items) > 0:
                    hits = len(set(recommended_k) & relevant_items)
                    precision = hits / len(recommended_k)
                    total_precision += precision
                    valid_users += 1

        return total_precision / valid_users if valid_users > 0 else 0.0

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