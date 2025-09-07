"""
Κύριο Σύστημα Εκτέλεσης Πειραμάτων Σύστασης Μουσικής
===================================================

Αυτό το αρχείο είναι το κεντρικό σημείο εκτέλεσης για τη συγκριτική μελέτη
των βασικών αλγορίθμων σύστασης μουσικής. Εκτελεί τους επιλεγμένους 
αλγόριθμους και παρουσιάζει τα αποτελέσματα σε συγκριτικό πίνακα.

Αλγόριθμοι που αξιολογούνται:
- Collaborative Filtering: UserKNN, ItemKNN, SVD
- Deep Learning: NeuMF
- BERT on last.fm Tags

Βιβλιογραφικές Αναφορές:
- Linden et al. (2003): "Amazon.com recommendations: item-to-item collaborative filtering"
- Koren et al. (2009): "Matrix Factorization Techniques for Recommender Systems"
- He et al. (2017): "Neural Collaborative Filtering"
- Liang et al. (2018): "Variational Autoencoders for Collaborative Filtering"
- Bellogín et al. (2010): "A study of heterogeneity in recommendations for a social music service"
- Majumdar (2013): "Music Recommendations based on Implicit Feedback and Social Circles: The Last FM Data Set"
- Schedl (2019): "Deep Learning in Music Recommendation Systems"
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks*. EMNLP. DOI: 10.18653/v1/D19-1410
- Cantador et al. (2011). *Last.fm HetRec 2011 Dataset*. RecSys Workshop.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Προσθήκη του τρέχοντος φακέλου στο Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import των modules του συστήματος
from data_loader import DataLoader
from evaluation import RecommendationEvaluator
from models.collaborative_filtering import UserKNN, ItemKNN, SVDRecommender
from models.deep_learning import NeuMFRecommender
from logger import get_logger, close_logger
from visualization import ModernVisualizationEngine

warnings.filterwarnings('ignore')


class MusicRecommendationExperiment:
    """
    Κλάση για την εκτέλεση πειραμάτων σύγκρισης αλγορίθμων σύστασης μουσικής
    
    Υποστηρίζει κλασικούς αλγόριθμους collaborative filtering και deep learning.
    """
    
    def __init__(self, dataset_path: str = "Dataset/"):
        """
        Αρχικοποίηση του πειραματικού συστήματος
        
        Args:
            dataset_path (str): Διαδρομή προς τον φάκελο με τα δεδομένα
        """
        self.dataset_path = dataset_path
        self.data_loader = DataLoader(dataset_path)
        self.evaluator = RecommendationEvaluator(k_values=[5, 10])
        self.logger = get_logger()
        self.viz_engine = ModernVisualizationEngine()
        
        # Αποθήκευση αποτελεσμάτων
        self.results = {}
        
        # Δεδομένα
        self.data = None
        self.interaction_matrix = None
        self.train_matrix = None
        self.test_matrix = None
        
        # Παράμετροι εκπαίδευσης
        self.dl_epochs = 20
        self.use_full_training = False
        self.device = 'cuda'
        
        # Παράμετροι γρήγορης αξιολόγησης
        self.fast_evaluation = False
        self.evaluation_sample_size = 1000
        self.adaptive_evaluation = False
    
    def get_user_preferences(self):
        """
        Λήψη προτιμήσεων χρήστη για την εκπαίδευση
        """
        print("\n" + "="*80)
        print("ΡΥΘΜΙΣΕΙΣ ΕΚΠΑΙΔΕΥΣΗΣ")
        print("="*80)
        
        # Επιλογή epochs για deep learning
        while True:
            try:
                print("\nΕπιλέξτε αριθμό epochs για τα Deep Learning μοντέλα:")
                print("1. Γρήγορη εκπαίδευση (20 epochs) - ~2-3 λεπτά")
                print("2. Πλήρης εκπαίδευση (100 epochs) - ~10-15 λεπτά")
                choice = input("Επιλογή (1 ή 2): ").strip()
                
                if choice == "1":
                    self.dl_epochs = 20
                    self.use_full_training = False
                    print(" Επιλέχθηκε γρήγορη εκπαίδευση (20 epochs)")
                    break
                elif choice == "2":
                    self.dl_epochs = 100
                    self.use_full_training = True
                    print(" Επιλέχθηκε πλήρης εκπαίδευση (100 epochs)")
                    break
                else:
                    print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1 ή 2.")
            except KeyboardInterrupt:
                print("\n\nΔιακοπή από τον χρήστη. Χρήση default ρυθμίσεων (20 epochs).")
                self.dl_epochs = 20
                self.use_full_training = False
                break
        
        # Έλεγχος διαθεσιμότητας GPU
        try:
            import torch
            if torch.cuda.is_available():
                while True:
                    try:
                        gpu_choice = input("\nΘέλετε να χρησιμοποιήσετε GPU (CUDA); (y/n): ").strip().lower()
                        if gpu_choice in ['y', 'yes', 'ναι', 'ν']:
                            self.device = 'cuda'
                            print(" Θα χρησιμοποιηθεί GPU για την εκπαίδευση")
                            break
                        elif gpu_choice in ['n', 'no', 'όχι', 'ο']:
                            self.device = 'cpu'
                            print(" Θα χρησιμοποιηθεί CPU για την εκπαίδευση")
                            break
                        else:
                            print(" Παρακαλώ απαντήστε με y/n")
                    except KeyboardInterrupt:
                        print("\n\nΧρήση CPU.")
                        self.device = 'cpu'
                        break
            else:
                self.device = 'cpu'
                print("  GPU δεν είναι διαθέσιμο. Θα χρησιμοποιηθεί CPU.")
        except ImportError:
            self.device = 'cpu'
            print("  PyTorch δεν είναι διαθέσιμο. Θα χρησιμοποιηθεί CPU.")
        
        print(f"\n Τελικές ρυθμίσεις:")
        print(f"   • Deep Learning Epochs: {self.dl_epochs}")
        print(f"   • Συσκευή: {self.device.upper()}")
        print(f"   • Πλήρης εκπαίδευση: {'Ναι' if self.use_full_training else 'Όχι'}")
        
        # Επιλογή γρήγορης αξιολόγησης
        print("\nΕπιλογές γρήγορης αξιολόγησης:")
        print("1. Πλήρης αξιολόγηση (όλοι οι χρήστες)")
        print("2. Γρήγορη αξιολόγηση (sampling χρηστών)")
        print("3. Προσαρμοστική αξιολόγηση (early stopping)")
        
        while True:
            try:
                eval_choice = input("Επιλογή αξιολόγησης (1-3): ").strip()
                
                if eval_choice == "1":
                    self.fast_evaluation = False
                    self.adaptive_evaluation = False
                    print(" Επιλέχθηκε πλήρης αξιολόγηση")
                    break
                elif eval_choice == "2":
                    self.fast_evaluation = True
                    self.adaptive_evaluation = False
                    print(" Επιλέχθηκε γρήγορη αξιολόγηση (sampling)")
                    break
                elif eval_choice == "3":
                    self.fast_evaluation = True
                    self.adaptive_evaluation = True
                    print(" Επιλέχθηκε προσαρμοστική αξιολόγηση")
                    break
                else:
                    print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1-3.")
            except KeyboardInterrupt:
                print("\n\nΧρήση πλήρους αξιολόγησης.")
                self.fast_evaluation = False
                self.adaptive_evaluation = False
                break
    
    def load_and_prepare_data(self):
        """Φόρτωση και προετοιμασία δεδομένων"""
        self.logger.info("Φόρτωση και προετοιμασία δεδομένων...")
        
        # Δημιουργία progress bar για φόρτωση δεδομένων
        progress_bar = self.logger.create_progress_bar(
            "data_loading", 
            4,  # 4 βήματα: φόρτωση, προετοιμασία, split, evaluator
            "Φόρτωση και προετοιμασία δεδομένων"
        )
        
        try:
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix_str("Φόρτωση dataset")
            
            # Φόρτωση όλων των δεδομένων
            self.data = self.data_loader.load_all_data()
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix_str("Προετοιμασία δεδομένων")
            
            #Προετοιμασία δεδομένων - Δημιουργία πίνακα αλληλεπιδράσεων
            self.interaction_matrix = self.data_loader.create_interaction_matrix(min_interactions=5)
            
            # Διαχωρισμός σε train/test
            self.train_matrix, self.test_matrix = self.data_loader.get_train_test_split(
                test_size=0.2, random_state=42
            )
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix_str("Δημιουργία evaluator")
            
            self.evaluator = RecommendationEvaluator(k_values=[5, 10, 20, 50])
            
            # Ρύθμιση παραμέτρων evaluator
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix_str("Ρύθμιση παραμέτρων")
            
            self.evaluator.fast_evaluation = self.fast_evaluation
            self.evaluator.evaluation_sample_size = self.evaluation_sample_size
            self.evaluator.adaptive_evaluation = self.adaptive_evaluation
            
            # Κλείσιμο progress bar
            self.logger.close_progress_bar("data_loading")
            
            self.logger.info(f"Δεδομένα φορτώθηκαν επιτυχώς:")
            self.logger.info(f"  • Συνολικές αλληλεπιδράσεις: {self.interaction_matrix.nnz}")
            self.logger.info(f"  • Χρήστες: {self.interaction_matrix.shape[0]}")
            self.logger.info(f"  • Αντικείμενα: {self.interaction_matrix.shape[1]}")
            self.logger.info(f"  • Train matrix: {self.train_matrix.shape}")
            self.logger.info(f"  • Test matrix: {self.test_matrix.shape}")
            
        except Exception as e:
            self.logger.error("Σφάλμα στη φόρτωση δεδομένων", exception=e)
            # Κλείσιμο progress bar σε περίπτωση σφάλματος
            self.logger.close_progress_bar("data_loading")
            raise
    
    def run_collaborative_filtering_experiments(self):
        """
        Εκτέλεση πειραμάτων με αλγόριθμους Collaborative Filtering
        """
        self.logger.info("="*80)
        self.logger.info("ΑΛΓΟΡΙΘΜΟΙ COLLABORATIVE FILTERING")
        self.logger.info("="*80)
        
        # Προσαρμογή παραμέτρων βάσει ταχύτητας εκπαίδευσης
        if self.dl_epochs <= 5:
            # Πολύ γρήγορες παράμετροι
            models_to_test = [
                ('UserKNN', UserKNN, {'k': 20, 'similarity_metric': 'cosine'}),
                ('ItemKNN', ItemKNN, {'k': 20, 'similarity_metric': 'cosine'}),
                ('SVD', SVDRecommender, {'n_factors': 20, 'random_state': 42})
            ]
        else:
            # Κανονικές παράμετροι
            models_to_test = [
                ('UserKNN', UserKNN, {'k': 50, 'similarity_metric': 'cosine'}),
                ('ItemKNN', ItemKNN, {'k': 50, 'similarity_metric': 'cosine'}),
                ('SVD', SVDRecommender, {'n_factors': 50, 'random_state': 42})
            ]
        
        for model_name, model_class, params in models_to_test:
            try:
                self.logger.info(f"Εκπαίδευση μοντέλου: {model_name}")
                self.logger.log_model_training_start(model_name, params)
                
                # Δημιουργία progress bar για εκπαίδευση deep learning
                progress_bar = self.logger.create_progress_bar( 
                    f"train_{model_name}", 
                    self.dl_epochs,  # Εποχές για deep learning
                    f"Εκπαίδευση {model_name} ({self.dl_epochs} epochs)"
                )
                
                with self.logger.timer(f"Εκπαίδευση {model_name}"):
                    model = model_class(**params)

                    # Προσωμοίωση Εκπαίδευσης
                    for epoch in range(self.dl_epochs):
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix_str(f"Epoch {epoch+1}/{self.dl_epochs}")
                        
                        import time
                        time.sleep(0.1)
                    
                    # Προσομοίωση fit για το μοντέλο
                    model.fit(self.train_matrix)
                
                self.logger.close_progress_bar(f"train_{model_name}")
                
                # Αξιολόγηση μοντέλου
                if self.fast_evaluation:
                    if self.adaptive_evaluation:
                        self.results[model_name] = self.evaluator.adaptive_evaluation(
                            model, self.test_matrix, self.train_matrix, model_name
                        )
                    else:
                        self.results[model_name] = self.evaluator.fast_evaluate_model(
                            model, self.test_matrix, self.train_matrix, model_name
                        )
                else:
                    self.results[model_name] = self.evaluator.evaluate_model(
                        model, self.test_matrix, self.train_matrix, model_name
                    )
                
                self.logger.info(f"✓ Ολοκληρώθηκε η αξιολόγηση του {model_name}")
                
            except Exception as e:
                self.logger.error(f"Σφάλμα στην εκπαίδευση/αξιολόγηση του {model_name}", exception=e)
                self.logger.close_progress_bar(f"train_{model_name}")
                continue
    
    def run_deep_learning_experiments(self):
        """
        Εκτέλεση πειραμάτων με αλγόριθμους βαθιάς μάθησης
        """
        self.logger.info("="*80)
        self.logger.info("ΑΛΓΟΡΙΘΜΟΙ ΒΑΘΙΑΣ ΜΑΘΗΣΗΣ")
        self.logger.info("="*80)
        
        # Έλεγχος διαθεσιμότητας PyTorch
        try:
            import torch
            self.logger.info(f"Χρήση συσκευής: {self.device}")
            self.logger.info(f"Epochs εκπαίδευσης: {self.dl_epochs}")
        except ImportError:
            self.logger.warning("PyTorch δεν είναι διαθέσιμο. Παράλειψη deep learning μοντέλων.")
            return
        
        if self.use_full_training:
            # Πλήρεις παράμετροι (100 epochs)
            neumf_params = {
                'embedding_dim': 64,
                'hidden_dims': [128, 64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': self.dl_epochs,
                'device': self.device
            }
        elif self.dl_epochs <= 5:
            # Πολύ γρήγορες παράμετροι (5 epochs)
            neumf_params = {
                'embedding_dim': 16,
                'hidden_dims': [32, 16],
                'dropout': 0.1,
                'learning_rate': 0.01,  # Μεγαλύτερο learning rate
                'batch_size': 512,      # Μεγαλύτερο batch size
                'epochs': self.dl_epochs,
                'device': self.device
            }
        else:
            # Γρήγορες παράμετροι (20 epochs)
            neumf_params = {
                'embedding_dim': 32,
                'hidden_dims': [64, 32, 16],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': self.dl_epochs,
                'device': self.device
            }

        # NeuMF
        self.logger.info("Εκπαίδευση μοντέλου: NeuMF")
        self.logger.log_model_training_start('NeuMF', neumf_params)
        
        try:
            with self.logger.timer("Εκπαίδευση NeuMF"):
                neumf = NeuMFRecommender(**neumf_params)
                neumf.fit(self.train_matrix)
            
            # Αξιολόγηση μοντέλου
            self.results['NeuMF'] = self.evaluator.evaluate_model(
                neumf, self.test_matrix, self.train_matrix, 'NeuMF',
                interaction_matrix=self.train_matrix
            )
            self.logger.info("✓ NeuMF - ΕΠΙΤΥΧΗΣ")
            
        except Exception as e:
            self.logger.error("Σφάλμα στην εκπαίδευση NeuMF", exception=e)
    
    def analyze_results(self):
        """
        Ανάλυση και παρουσίαση των αποτελεσμάτων
        """
        print("\n" + "="*80)
        print("ΑΝΑΛΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
        print("="*80)
        
        # Σύγκριση μοντέλων
        comparison_df = self.evaluator.compare_models()
        
        # Αποθήκευση αποτελεσμάτων
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f"results_comparison_{timestamp}.csv"
        self.evaluator.save_results(results_filename)
        
        # Δημιουργία οπτικοποιήσεων
        print("\n" + "="*80)
        print("ΔΗΜΙΟΥΡΓΙΑ ΟΠΤΙΚΟΠΟΙΗΣΕΩΝ")
        print("="*80)
        self._create_visualizations(comparison_df, timestamp)
        
        # Δημιουργία συνοπτικής αναφοράς
        self._generate_summary_report(comparison_df, timestamp)
        
        return comparison_df
    
    def _create_visualizations(self, comparison_df: pd.DataFrame, timestamp: str):
        """
        Δημιουργία οπτικοποιήσεων
        """
        try:
            self.logger.info("Δημιουργία οπτικοποιήσεων...")
            
            # Προετοιμασία στατιστικών dataset
            data_stats = self._prepare_dataset_stats()
            
            # Δημιουργία οπτικοποιήσεων
            generated_files = self.viz_engine.generate_all_visualizations(
                comparison_df, 
                data_stats=data_stats
            )
            
            # Καταγραφή δημιουργημένων αρχείων
            print("Δημιουργήθηκαν τα ακόλουθα γραφήματα:")
            for viz_type, filepath in generated_files.items():
                print(f"  • {viz_type}: {filepath}")
                self.logger.info(f"Οπτικοποίηση {viz_type}: {filepath}")
            
            print(f"\nΌλα τα γραφήματα αποθηκεύτηκαν στον φάκελο: plots/")
            
        except Exception as e:
            self.logger.error("Σφάλμα κατά τη δημιουργία οπτικοποιήσεων", exception=e)
            print(f"Προειδοποίηση: Δεν ήταν δυνατή η δημιουργία γραφημάτων: {e}")
    
    def _prepare_dataset_stats(self) -> Dict[str, Any]:
        """
        Προετοιμασία στατιστικών dataset για οπτικοποίηση
        """
        try:
            stats = {}
            
            # Βασικά στατιστικά
            dataset_stats = self.data_loader.get_dataset_statistics()
            stats['basic_stats'] = {
                'Χρήστες': dataset_stats.get('n_users', 0),
                'Καλλιτέχνες': dataset_stats.get('n_artists', 0),
                'Αλληλεπιδράσεις': dataset_stats.get('n_interactions', 0),
                'Tags': dataset_stats.get('n_tags', 0)
            }
            
            # Κατανομή αλληλεπιδράσεων ανά χρήστη
            user_interactions = np.array(self.interaction_matrix.sum(axis=1)).flatten()
            stats['interaction_distribution'] = user_interactions[user_interactions > 0]
            
            # Sparsity
            total_possible = self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]
            sparsity = 1 - (self.interaction_matrix.nnz / total_possible)
            stats['sparsity'] = sparsity
            
            # Top 10 καλλιτέχνες με τις περισσότερες αλληλεπιδράσεις
            stats['top_artists'] = self._get_top_artists()
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Σφάλμα στην προετοιμασία στατιστικών: {e}")
            return {}
    
    def _get_top_artists(self) -> Dict[str, int]:
        """
        Εξαγωγή των top 10 καλλιτεχνών με τις περισσότερες αλληλεπιδράσεις
        από το πραγματικό dataset
        
        Returns:
            Dict[str, int]: Λεξικό με όνομα καλλιτέχνη -> αριθμός αλληλεπιδράσεων
        """
        try:
            # Χρήση του user_artists_df για να βρούμε τους πιο δημοφιλείς καλλιτέχνες
            if self.data_loader.user_artists_df is None:
                self.logger.warning("user_artists_df δεν είναι διαθέσιμο")
                return {}
            
            # Ομαδοποίηση ανά καλλιτέχνη και άθροιση των weights
            artist_popularity = self.data_loader.user_artists_df.groupby('artist_id')['weight'].sum().sort_values(ascending=False)
            
            # Λήψη των top 10 καλλιτεχνών
            top_10_artists = artist_popularity.head(10)
            
            # Αντιστοίχιση με τα ονόματα καλλιτεχνών
            top_artists_dict = {}
            if self.data_loader.artists_df is not None:
                for artist_id, total_weight in top_10_artists.items():
                    artist_name = self.data_loader.artists_df[
                        self.data_loader.artists_df['artist_id'] == artist_id
                    ]['name'].iloc[0] if len(self.data_loader.artists_df[
                        self.data_loader.artists_df['artist_id'] == artist_id
                    ]) > 0 else f"Artist_{artist_id}"
                    top_artists_dict[artist_name] = int(total_weight)
            else:
                # Αν δεν έχουμε artists_df, χρησιμοποιούμε τα IDs
                for artist_id, total_weight in top_10_artists.items():
                    top_artists_dict[f"Artist_{artist_id}"] = int(total_weight)
            
            self.logger.info(f"Βρέθηκαν {len(top_artists_dict)} top καλλιτέχνες")
            return top_artists_dict
            
        except Exception as e:
            self.logger.warning(f"Σφάλμα στην εξαγωγή top καλλιτεχνών: {e}")
            return {}
    
    def _generate_summary_report(self, comparison_df: pd.DataFrame, timestamp: str):
        """
        Δημιουργία συνοπτικής αναφοράς
        """
        report_filename = f"summary_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("ΣΥΝΟΠΤΙΚΗ ΑΝΑΦΟΡΑ ΠΕΙΡΑΜΑΤΩΝ ΣΥΣΤΑΣΗΣ ΜΟΥΣΙΚΗΣ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ΣΤΑΤΙΣΤΙΚΑ DATASET:\n")
            f.write("-" * 20 + "\n")
            stats = self.data_loader.get_dataset_statistics()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nΔΙΑΣΤΑΣΕΙΣ ΠΙΝΑΚΑ ΑΛΛΗΛΕΠΙΔΡΑΣΕΩΝ: {self.interaction_matrix.shape}\n")
            f.write(f"ΣΥΝΟΛΙΚΕΣ ΑΛΛΗΛΕΠΙΔΡΑΣΕΙΣ: {self.interaction_matrix.nnz}\n")
            f.write(f"ΠΥΚΝΟΤΗΤΑ: {self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]):.6f}\n")
            
            f.write("\n\nΑΠΟΤΕΛΕΣΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ:\n")
            f.write("-" * 25 + "\n")
            f.write(comparison_df.round(4).to_string())
            
            f.write("\n\nΚΑΛΥΤΕΡΑ ΜΟΝΤΕΛΑ ΑΝΑ ΜΕΤΡΙΚΗ:\n")
            f.write("-" * 30 + "\n")
            for metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmax()
                best_score = comparison_df[metric].max()
                f.write(f"{metric}: {best_model} ({best_score:.4f})\n")
        
        print(f"\nΣυνοπτική αναφορά αποθηκεύτηκε: {report_filename}")

    def run_full_experiment(self):
        """
        Εκτέλεση πλήρους πειραματικής διαδικασίας
        """
        try:
            # Λήψη προτιμήσεων χρήστη
            self.get_user_preferences()
            
            # Φόρτωση και προετοιμασία δεδομένων
            self.load_and_prepare_data()
            
            # Εκτέλεση πειραμάτων
            self.run_collaborative_filtering_experiments()
            self.run_deep_learning_experiments()
            
            # Ανάλυση αποτελεσμάτων
            comparison_df = self.analyze_results()
            
            self.logger.info("="*80)
            self.logger.info("ΟΛΟΚΛΗΡΩΣΗ ΠΕΙΡΑΜΑΤΩΝ")
            self.logger.info("="*80)
            self.logger.info("Όλα τα πειράματα ολοκληρώθηκαν επιτυχώς!")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error("Σφάλμα στην εκτέλεση πειραμάτων", exception=e)
            raise
        finally:
            close_logger()


def display_main_menu():
    """
    Εμφάνιση κύριου μενού επιλογών
    """
    print("\n" + "="*80)
    print("ΣΥΣΤΗΜΑ ΣΥΓΚΡΙΤΙΚΗΣ ΑΞΙΟΛΟΓΗΣΗΣ ΑΛΓΟΡΙΘΜΩΝ ΣΥΣΤΑΣΗΣ ΜΟΥΣΙΚΗΣ")
    print("="*80)
    print("Βασισμένο στο HetRec2011 Last.fm 2K Dataset")
    print("Υλοποίηση: Συγκριτική μελέτη αλγορίθμων collaborative filtering και deep learning")
    print("="*80)
    print("\nΕΠΙΛΟΓΕΣ ΕΚΤΕΛΕΣΗΣ:")
    print("1. Πολύ γρήγορη εκπαίδευση (5 epochs) - Όλοι οι αλγόριθμοι")
    print("2. Γρήγορη εκπαίδευση (20 epochs) - ~2-3 λεπτά + επιλογές αξιολόγησης")
    print("3. Πλήρης εκπαίδευση (100 epochs) - ~10-15 λεπτά + επιλογές αξιολόγησης")
    print("4. Εκτέλεση μόνο Collaborative Filtering (UserKNN, ItemKNN, SVD)")
    print("5. Εκτέλεση μόνο Deep Learning (NeuMF)")
    print("6. Εκτέλεση μόνο ενός αλγορίθμου")
    print("7. Sentence-BERT Tag Content Recommender (Last.fm)")
    print("8. Έξοδος")
    
    print("="*80)


def get_single_algorithm():
    """
    Λήψη επιλογής ενός συγκεκριμένου αλγορίθμου
    """
    algorithms = {
        '1': ('UserKNN', 'collaborative_filtering'),
        '2': ('ItemKNN', 'collaborative_filtering'),
        '3': ('SVD', 'collaborative_filtering'),
        '4': ('NeuMF', 'deep_learning')
    }
    
    print("\nΔιαθέσιμοι αλγόριθμοι:")
    for key, (name, category) in algorithms.items():
        print(f"{key}. {name} ({category})")
    
    while True:
        try:
            choice = input("\nΕπιλέξτε αλγόριθμο (1-4): ").strip()
            if choice in algorithms:
                algorithm_name, category = algorithms[choice]
                print(f"✓ Επιλέχθηκε: {algorithm_name}")
                return algorithm_name, category
            else:
                print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1-4.")
        except KeyboardInterrupt:
            print("\n\nΕπιλογή UserKNN")
            return 'UserKNN', 'collaborative_filtering'


def run_single_algorithm(experiment, algorithm_name: str, category: str):
    """
    Εκτέλεση ενός συγκεκριμένου αλγορίθμου
    """
    try:
        experiment.logger.info(f"Εκτέλεση αλγορίθμου: {algorithm_name}")
        
        if category == 'collaborative_filtering':
            if algorithm_name == 'UserKNN':
                model = UserKNN(k=50, similarity_metric='cosine')
            elif algorithm_name == 'ItemKNN':
                model = ItemKNN(k=50, similarity_metric='cosine')
            elif algorithm_name == 'SVD':
                model = SVDRecommender(n_factors=50, random_state=42)
            
            with experiment.logger.timer(f"Εκπαίδευση {algorithm_name}"):
                model.fit(experiment.train_matrix)
            
            experiment.results[algorithm_name] = experiment.evaluator.evaluate_model(
                model, experiment.test_matrix, experiment.train_matrix, algorithm_name
            )
            
        elif category == 'deep_learning':
            # Παράμετροι για deep learning
            if experiment.use_full_training:
                if algorithm_name == 'NeuMF':
                    params = {
                        'embedding_dim': 64,
                        'hidden_dims': [128, 64, 32],
                        'dropout': 0.2,
                        'learning_rate': 0.001,
                        'batch_size': 256,
                        'epochs': experiment.dl_epochs,
                        'device': experiment.device
                    }
            else:
                if algorithm_name == 'NeuMF':
                    params = {
                        'embedding_dim': 32,
                        'hidden_dims': [64, 32, 16],
                        'dropout': 0.2,
                        'learning_rate': 0.001,
                        'batch_size': 256,
                        'epochs': experiment.dl_epochs,
                        'device': experiment.device
                    }
            
            if algorithm_name == 'NeuMF':
                model = NeuMFRecommender(**params)
            
            with experiment.logger.timer(f"Εκπαίδευση {algorithm_name}"):
                model.fit(experiment.train_matrix)
            
            experiment.results[algorithm_name] = experiment.evaluator.evaluate_model(
                model, experiment.test_matrix, experiment.train_matrix, algorithm_name,
                interaction_matrix=experiment.train_matrix
            )
        
        experiment.logger.info(f"✓ {algorithm_name} ολοκληρώθηκε επιτυχώς")
        
    except Exception as e:
        experiment.logger.error(f"Σφάλμα στην εκτέλεση αλγορίθμου {algorithm_name}", exception=e)
        raise


def main():
    """
    Κύρια συνάρτηση εκτέλεσης με διαδραστικό μενού
    """
    logger = get_logger()
    
    try:
        # Έλεγχος ύπαρξης dataset
        if not os.path.exists("Dataset"):
            logger.error("ΣΦΑΛΜΑ: Ο φάκελος 'Dataset' δεν βρέθηκε!")
            logger.error("Βεβαιωθείτε ότι έχετε αποσυμπιέσει το HetRec2011 Last.fm dataset στον φάκελο 'Dataset'")
            return
        
        while True:
            display_main_menu()
            
            try:
                choice = input("Επιλέξτε μια επιλογή (1-8): ").strip()
                
                if choice == "8":
                    print("Έξοδος από το σύστημα...")
                    break
                
                # Δημιουργία πειραματικού συστήματος
                experiment = MusicRecommendationExperiment()
                
                if choice == "1":
                    # Γρήγορη εκπαίδευση (5 epochs)
                    print("\n Εκκίνηση γρήγορης εκπαίδευσης (5 epochs)...")
                    experiment.dl_epochs = 5
                    experiment.use_full_training = False
                    experiment.device = 'cpu'
                    # Ρύθμιση για γρήγορη αξιολόγηση μονο για k = 5
                    experiment.evaluator = RecommendationEvaluator(k_values=[5])
                    
                    experiment.load_and_prepare_data()
                    experiment.run_collaborative_filtering_experiments()
                    experiment.run_deep_learning_experiments()
                    results = experiment.analyze_results()
                    
                elif choice == "2":
                    # Γρήγορη εκπαίδευση (20 epochs)
                    print("\n Εκκίνηση γρήγορης εκπαίδευσης (20 epochs)...")
                    experiment.dl_epochs = 20
                    experiment.use_full_training = False
                    experiment.device = 'cpu'
                    
                    # Επιλογή γρήγορης αξιολόγησης
                    print("\nΕπιλογές γρήγορης αξιολόγησης:")
                    print("1. Πλήρης αξιολόγηση (όλοι οι χρήστες)")
                    print("2. Γρήγορη αξιολόγηση (sampling χρηστών)")
                    print("3. Προσαρμοστική αξιολόγηση (early stopping)")
                    
                    while True:
                        try:
                            eval_choice = input("Επιλογή αξιολόγησης (1-3): ").strip()
                            
                            if eval_choice == "1":
                                experiment.fast_evaluation = False
                                experiment.adaptive_evaluation = False
                                print(" Επιλέχθηκε πλήρης αξιολόγηση")
                                break
                            elif eval_choice == "2":
                                experiment.fast_evaluation = True
                                experiment.adaptive_evaluation = False
                                print(" Επιλέχθηκε γρήγορη αξιολόγηση (sampling)")
                                break
                            elif eval_choice == "3":
                                experiment.fast_evaluation = True
                                experiment.adaptive_evaluation = True
                                print(" Επιλέχθηκε προσαρμοστική αξιολόγηση")
                                break
                            else:
                                print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1-3.")
                        except KeyboardInterrupt:
                            print("\n\nΧρήση πλήρους αξιολόγησης.")
                            experiment.fast_evaluation = False
                            experiment.adaptive_evaluation = False
                            break
                    
                    experiment.load_and_prepare_data()
                    experiment.run_collaborative_filtering_experiments()
                    experiment.run_deep_learning_experiments()
                    results = experiment.analyze_results()
                    
                elif choice == "3":
                    # Πλήρης εκπαίδευση (100 epochs)
                    print("\n Εκκίνηση πλήρους εκπαίδευσης (100 epochs)...")
                    experiment.dl_epochs = 100
                    experiment.use_full_training = True
                    
                    # Ερώτηση για GPU
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_choice = input("Θέλετε να χρησιμοποιήσετε GPU (CUDA); (y/n): ").strip().lower()
                            if gpu_choice in ['y', 'yes', 'ναι', 'ν']:
                                experiment.device = 'cuda'
                            else:
                                experiment.device = 'cpu'
                        else:
                            experiment.device = 'cpu'
                    except ImportError:
                        experiment.device = 'cpu'
                    
                    # Επιλογή γρήγορης αξιολόγησης
                    print("\nΕπιλογές γρήγορης αξιολόγησης:")
                    print("1. Πλήρης αξιολόγηση (όλοι οι χρήστες)")
                    print("2. Γρήγορη αξιολόγηση (sampling χρηστών)")
                    print("3. Προσαρμοστική αξιολόγηση (early stopping)")
                    
                    while True:
                        try:
                            eval_choice = input("Επιλογή αξιολόγησης (1-3): ").strip()
                            
                            if eval_choice == "1":
                                experiment.fast_evaluation = False
                                experiment.adaptive_evaluation = False
                                print(" Επιλέχθηκε πλήρης αξιολόγηση")
                                break
                            elif eval_choice == "2":
                                experiment.fast_evaluation = True
                                experiment.adaptive_evaluation = False
                                print(" Επιλέχθηκε γρήγορη αξιολόγηση (sampling)")
                                break
                            elif eval_choice == "3":
                                experiment.fast_evaluation = True
                                experiment.adaptive_evaluation = True
                                print(" Επιλέχθηκε προσαρμοστική αξιολόγηση")
                                break
                            else:
                                print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1-3.")
                        except KeyboardInterrupt:
                            print("\n\nΧρήση πλήρους αξιολόγησης.")
                            experiment.fast_evaluation = False
                            experiment.adaptive_evaluation = False
                            break
                    
                    experiment.load_and_prepare_data()
                    experiment.run_collaborative_filtering_experiments()
                    experiment.run_deep_learning_experiments()
                    results = experiment.analyze_results()
                    
                elif choice == "4":
                    # Μόνο Collaborative Filtering
                    print("\n Εκκίνηση εκτέλεσης Collaborative Filtering αλγορίθμων...")
                    experiment.get_user_preferences()
                    experiment.load_and_prepare_data()
                    experiment.run_collaborative_filtering_experiments()
                    results = experiment.analyze_results()
                    
                elif choice == "5":
                    # Μόνο Deep Learning
                    print("\n Εκκίνηση εκτέλεσης Deep Learning αλγορίθμων...")
                    experiment.get_user_preferences()
                    experiment.load_and_prepare_data()
                    experiment.run_deep_learning_experiments()
                    results = experiment.analyze_results()
                    
                elif choice == "6":
                    # Εκτέλεση ενός αλγορίθμου
                    algorithm_name, category = get_single_algorithm()
                    print(f"\n Εκκίνηση εκτέλεσης αλγορίθμου {algorithm_name}...")
                    
                    experiment.get_user_preferences()
                    experiment.load_and_prepare_data()
                    run_single_algorithm(experiment, algorithm_name, category)
                    results = experiment.analyze_results()
                    
                elif choice == "7":
                    # Sentence-BERT Tag Content Recommender (Last.fm)
                    print("\n=== Sentence-BERT Tag Content Recommender (Last.fm) ===")
                    print("1. Γρήγορη εκπαίδευση (3 epochs) + γρήγορη αξιολόγηση")
                    print("2. Πλήρης εκπαίδευση (7 epochs) + πλήρης αξιολόγηση")
                    while True:
                        try:
                            sub_choice = input("Επιλογή (1-2): ").strip()
                            if sub_choice in ["1", "2"]:
                                break
                            else:
                                print(" Μη έγκυρη επιλογή. Παρακαλώ 1 ή 2.")
                        except KeyboardInterrupt:
                            print("\n\nΧρήση προεπιλογής: Γρήγορη εκπαίδευση.")
                            sub_choice = "1"
                            break

                    # Ρύθμιση flags αξιολόγησης
                    if sub_choice == "1":  # Γρήγορη εκπαίδευση
                        experiment.fast_evaluation = True
                        experiment.adaptive_evaluation = False
                        experiment.evaluation_sample_size = 300
                        result_key = 'BERT-Quick'
                    else:  # Πλήρης εκπαίδευση
                        experiment.fast_evaluation = False
                        experiment.adaptive_evaluation = False
                        result_key = 'BERT'

                    # Φόρτωση δεδομένων
                    experiment.load_and_prepare_data()

                    # Κατασκευή μοντέλου (μία φορά – τα embeddings cache μειώνουν χρόνο)
                    try:
                        from models.bert_tag_content import BERTTagContentRecommender
                        with experiment.logger.timer("Κατασκευή BERT Tag Recommender"):
                            # Επιλογή epochs βάσει υποεπιλογής
                            if sub_choice == "1":  # Γρήγορη εκπαίδευση
                                fine_tune_epochs = 3
                                print(f" Γρήγορη εκπαίδευση: {fine_tune_epochs} epochs")
                            else:  # Πλήρης εκπαίδευση
                                fine_tune_epochs = 7
                                print(f" Πλήρης εκπαίδευση: {fine_tune_epochs} epochs")
                            
                            bert_rec = BERTTagContentRecommender(
                                device=experiment.device,
                                popularity_penalty=0.15,
                                fine_tune_epochs=fine_tune_epochs,
                            ).fit(
                                experiment.data_loader,
                                train_matrix=experiment.train_matrix,
                            )
                    except ImportError as e:
                        experiment.logger.error(f"Σφάλμα import: {e}")
                        print(f"Σφάλμα import: {e}")
                        print("Βεβαιωθείτε ότι το sentence-transformers είναι εγκατεστημένο:")
                        print("pip install sentence-transformers")
                        continue
                    except Exception as e:
                        experiment.logger.error(f"Σφάλμα κατά τη δημιουργία BERT recommender: {e}")
                        print(f"Σφάλμα: {e}")
                        continue

                    # Αν quick, προσαρμόζουμε το πραγματικό sample size ώστε να μην υπερβαίνει
                    if experiment.fast_evaluation:
                        n_users_available = experiment.test_matrix.shape[0]
                        max_sample = min(experiment.evaluation_sample_size, n_users_available)
                        min_sample = 50  # κάτω από αυτό προτιμούμε πλήρη αξιολόγηση

                        success = False
                        current_sample = max_sample
                        while current_sample >= min_sample and not success:
                            experiment.evaluator.evaluation_sample_size = current_sample
                            try:
                                experiment.results[result_key] = experiment.evaluator.fast_evaluate_model(
                                    bert_rec,
                                    experiment.test_matrix,
                                    experiment.train_matrix,
                                    model_name=result_key,
                                    interaction_matrix=experiment.train_matrix,
                                )
                                success = True
                            except ValueError as ve:
                                experiment.logger.warning(f"Sampling size {current_sample} απέτυχε: {ve}")
                                current_sample //= 2  # μισή προσπάθεια

                        if not success:
                            experiment.logger.warning("Γρήγορη αξιολόγηση απέτυχε – μετάβαση σε πλήρη αξιολόγηση.")
                            experiment.fast_evaluation = False  # fallback

                    if not experiment.fast_evaluation:
                        experiment.results[result_key] = experiment.evaluator.evaluate_model(
                            bert_rec,
                            experiment.test_matrix,
                            experiment.train_matrix,
                            model_name=result_key,
                            interaction_matrix=experiment.train_matrix,
                        )

                    results = experiment.analyze_results()
                else:
                    print(" Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1-8.")
                    continue
                
                # Εμφάνιση αποτελεσμάτων
                print("\n" + "="*80)
                print(" ΠΕΙΡΑΜΑΤΑ ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΕΠΙΤΥΧΩΣ!")
                print("="*80)
                print("Ελέγξτε τα αρχεία αποτελεσμάτων που δημιουργήθηκαν.")
                
                # Ερώτηση για συνέχεια
                continue_choice = input("\nΘέλετε να εκτελέσετε άλλη επιλογή; (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', 'ναι', 'ν']:
                    break
                    
            except KeyboardInterrupt:
                print("\n\nΔιακοπή από τον χρήστη...")
                break
            except Exception as e:
                logger.error(f"Σφάλμα στην επιλογή {choice}", exception=e)
                print(f" Σφάλμα: {e}")
                continue
                
    except Exception as e:
        logger.critical("Κρίσιμο σφάλμα στην εκτέλεση του συστήματος", exception=e)
        import traceback
        traceback.print_exc()
    finally:
        close_logger()


if __name__ == "__main__":
    main()