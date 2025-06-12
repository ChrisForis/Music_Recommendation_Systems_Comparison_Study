"""
Φορτωτής Δεδομένων για το Σύστημα Σύστασης Μουσικής
==================================================

Αυτό το module είναι υπεύθυνο για τη φόρτωση και προεπεξεργασία των δεδομένων
από το HetRec2011 Last.fm 2K dataset. Περιλαμβάνει λειτουργίες για:
- Φόρτωση όλων των αρχείων δεδομένων σε pandas DataFrames
- Προεπεξεργασία και καθαρισμό των δεδομένων
- Δημιουργία πινάκων αλληλεπιδράσεων χρηστών-καλλιτεχνών
- Διαχωρισμό δεδομένων σε σύνολα εκπαίδευσης και αξιολόγησης

Βιβλιογραφικές Αναφορές:
- Cantador et al. (2011): HetRec 2011 Workshop
- Last.fm dataset documentation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Εισαγωγή του logging συστήματος
from logger import get_logger


class DataLoader:
    """
    Κλάση για τη φόρτωση και προεπεξεργασία των δεδομένων Last.fm
    
    Αυτή η κλάση παρέχει όλες τις απαραίτητες λειτουργίες για τη διαχείριση
    του HetRec2011 Last.fm dataset, συμπεριλαμβανομένης της φόρτωσης,
    προεπεξεργασίας και προετοιμασίας των δεδομένων για τους αλγόριθμους σύστασης.
    """
    
    def __init__(self, dataset_path: str = "Dataset/"):
        """
        Αρχικοποίηση του φορτωτή δεδομένων
        
        Args:
            dataset_path (str): Διαδρομή προς τον φάκελο με τα δεδομένα
        """
        self.dataset_path = dataset_path
        self.artists_df = None
        self.tags_df = None
        self.user_artists_df = None
        self.user_taggedartists_df = None
        self.user_friends_df = None
        self.user_taggedartists_timestamps_df = None
        
        # Πίνακες αλληλεπιδράσεων
        self.interaction_matrix = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.artist_to_idx = None
        self.idx_to_artist = None
        
        # Ρυθμίσεις dataset
        self.dataset_ratio = 1.0  # Ποσοστό του dataset που θα χρησιμοποιηθεί (1.0 = 100%)
        
        # Αρχικοποίηση logger
        self.logger = get_logger()
        self.logger.info(f"Αρχικοποίηση DataLoader με dataset path: {dataset_path}")
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Φόρτωση όλων των αρχείων δεδομένων
        
        Φορτώνει όλα τα αρχεία του dataset σε pandas DataFrames με τους
        κατάλληλους τύπους δεδομένων και επιστρέφει ένα λεξικό με όλα τα DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Λεξικό με όλα τα φορτωμένα DataFrames
        """
        with self.logger.timer("Φόρτωση δεδομένων Last.fm"):
            try:
                # Φόρτωση καλλιτεχνών
                self.logger.info("Φόρτωση αρχείου καλλιτεχνών...")
                self.artists_df = self._load_artists()
                self.logger.info(f"Φορτώθηκαν {len(self.artists_df)} καλλιτέχνες")
                
                # Φόρτωση tags
                self.logger.info("Φόρτωση αρχείου tags...")
                self.tags_df = self._load_tags()
                self.logger.info(f"Φορτώθηκαν {len(self.tags_df)} tags")
                
                # Φόρτωση αλληλεπιδράσεων χρηστών-καλλιτεχνών
                self.logger.info("Φόρτωση αλληλεπιδράσεων χρηστών-καλλιτεχνών...")
                self.user_artists_df = self._load_user_artists()
                self.logger.info(f"Φορτώθηκαν {len(self.user_artists_df)} αλληλεπιδράσεις χρηστών-καλλιτεχνών")
                
                # Φόρτωση tagged artists
                self.logger.info("Φόρτωση tag assignments...")
                self.user_taggedartists_df = self._load_user_taggedartists()
                self.logger.info(f"Φορτώθηκαν {len(self.user_taggedartists_df)} tag assignments")
                
                # Φόρτωση φιλιών
                self.logger.info("Φόρτωση σχέσεων φιλίας...")
                self.user_friends_df = self._load_user_friends()
                self.logger.info(f"Φορτώθηκαν {len(self.user_friends_df)} σχέσεις φιλίας")
                
                # Φόρτωση timestamps
                self.logger.info("Φόρτωση timestamps...")
                self.user_taggedartists_timestamps_df = self._load_user_taggedartists_timestamps()
                self.logger.info(f"Φορτώθηκαν {len(self.user_taggedartists_timestamps_df)} timestamps")
                
                # Υπολογισμός και καταγραφή στατιστικών
                stats = self.get_dataset_statistics()
                self.logger.log_dataset_stats(stats)
                
                return {
                    'artists': self.artists_df,
                    'tags': self.tags_df,
                    'user_artists': self.user_artists_df,
                    'user_taggedartists': self.user_taggedartists_df,
                    'user_friends': self.user_friends_df,
                    'user_taggedartists_timestamps': self.user_taggedartists_timestamps_df
                }
                
            except Exception as e:
                self.logger.error("Σφάλμα κατά τη φόρτωση δεδομένων", exception=e)
                raise
    
    def _load_artists(self) -> pd.DataFrame:
        """Φόρτωση αρχείου artists.dat"""
        filepath = os.path.join(self.dataset_path, "artists.dat")
        
        try:
            self.logger.debug(f"Φόρτωση αρχείου: {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Το αρχείο {filepath} δεν βρέθηκε")
            
            try:
                df = pd.read_csv(filepath, sep='\t', encoding='utf-8', 
                                names=['artist_id', 'name', 'url', 'picture_url'], skiprows=1)
                self.logger.debug("Επιτυχής φόρτωση με UTF-8 encoding")
            except UnicodeDecodeError:
                self.logger.warning("Αποτυχία UTF-8 encoding, δοκιμή με latin-1")
                df = pd.read_csv(filepath, sep='\t', encoding='latin-1', 
                                names=['artist_id', 'name', 'url', 'picture_url'], skiprows=1)
                self.logger.debug("Επιτυχής φόρτωση με latin-1 encoding")
            
            # Έλεγχος και μετατροπή τύπων δεδομένων
            if 'artist_id' not in df.columns or df['artist_id'].isnull().any():
                raise ValueError("Μη έγκυρα δεδομένα artist_id")
            
            df['artist_id'] = df['artist_id'].astype(int)
            self.logger.debug(f"Φορτώθηκαν {len(df)} καλλιτέχνες με επιτυχία")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά τη φόρτωση αρχείου καλλιτεχνών: {filepath}", exception=e)
            raise
    
    def _load_tags(self) -> pd.DataFrame:
        """Φόρτωση αρχείου tags.dat"""
        filepath = os.path.join(self.dataset_path, "tags.dat")
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8',
                            names=['tag_id', 'tag_value'], skiprows=1)
        except UnicodeDecodeError:
            # Δοκιμή με latin-1 encoding
            df = pd.read_csv(filepath, sep='\t', encoding='latin-1',
                            names=['tag_id', 'tag_value'], skiprows=1)
        df['tag_id'] = df['tag_id'].astype(int)
        return df
    
    def _load_user_artists(self) -> pd.DataFrame:
        """Φόρτωση αρχείου user_artists.dat"""
        filepath = os.path.join(self.dataset_path, "user_artists.dat")
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8',
                        names=['user_id', 'artist_id', 'weight'], skiprows=1)
        df['user_id'] = df['user_id'].astype(int)
        df['artist_id'] = df['artist_id'].astype(int)
        df['weight'] = df['weight'].astype(int)
        
        # Εφαρμογή μειωμένου dataset αν χρειάζεται
        if self.dataset_ratio < 1.0:
            original_size = len(df)
            # Επιλογή τυχαίου δείγματος χρηστών
            unique_users = df['user_id'].unique()
            n_users_to_keep = int(len(unique_users) * self.dataset_ratio)
            selected_users = np.random.choice(unique_users, n_users_to_keep, replace=False)
            df = df[df['user_id'].isin(selected_users)]
            self.logger.info(f"Μείωση dataset από {original_size} σε {len(df)} αλληλεπιδράσεις ({self.dataset_ratio*100:.1f}%)")
        
        return df
    
    def _load_user_taggedartists(self) -> pd.DataFrame:
        """Φόρτωση αρχείου user_taggedartists.dat"""
        filepath = os.path.join(self.dataset_path, "user_taggedartists.dat")
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8',
                        names=['user_id', 'artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        for col in ['user_id', 'artist_id', 'tag_id', 'day', 'month', 'year']:
            df[col] = df[col].astype(int)
        return df
    
    def _load_user_friends(self) -> pd.DataFrame:
        """Φόρτωση αρχείου user_friends.dat"""
        filepath = os.path.join(self.dataset_path, "user_friends.dat")
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8',
                        names=['user_id', 'friend_id'], skiprows=1)
        df['user_id'] = df['user_id'].astype(int)
        df['friend_id'] = df['friend_id'].astype(int)
        return df
    
    def _load_user_taggedartists_timestamps(self) -> pd.DataFrame:
        """Φόρτωση αρχείου user_taggedartists-timestamps.dat"""
        filepath = os.path.join(self.dataset_path, "user_taggedartists-timestamps.dat")
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8',
                        names=['user_id', 'artist_id', 'tag_id', 'timestamp'], skiprows=1)
        df['user_id'] = df['user_id'].astype(int)
        df['artist_id'] = df['artist_id'].astype(int)
        df['tag_id'] = df['tag_id'].astype(int)
        df['timestamp'] = df['timestamp'].astype(int)
        return df
    
    def create_interaction_matrix(self, min_interactions: int = 5) -> csr_matrix:
        """
        Δημιουργία πίνακα αλληλεπιδράσεων χρηστών-καλλιτεχνών
        
        Δημιουργεί έναν αραιό πίνακα (sparse matrix) που περιέχει τις αλληλεπιδράσεις
        μεταξύ χρηστών και καλλιτεχνών. Φιλτράρει χρήστες και καλλιτέχνες με λίγες
        αλληλεπιδράσεις για να βελτιώσει την ποιότητα των συστάσεων.
        
        Args:
            min_interactions (int): Ελάχιστος αριθμός αλληλεπιδράσεων για χρήστες/καλλιτέχνες
            
        Returns:
            csr_matrix: Αραιός πίνακας αλληλεπιδράσεων
        """
        if self.user_artists_df is None:
            raise ValueError("Πρέπει πρώτα να φορτώσετε τα δεδομένα με load_all_data()")
        
        # Φιλτράρισμα χρηστών και καλλιτεχνών με λίγες αλληλεπιδράσεις
        user_counts = self.user_artists_df['user_id'].value_counts()
        artist_counts = self.user_artists_df['artist_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_artists = artist_counts[artist_counts >= min_interactions].index
        
        filtered_df = self.user_artists_df[
            (self.user_artists_df['user_id'].isin(valid_users)) &
            (self.user_artists_df['artist_id'].isin(valid_artists))
        ].copy()
        
        # Δημιουργία mappings
        unique_users = sorted(filtered_df['user_id'].unique())
        unique_artists = sorted(filtered_df['artist_id'].unique())
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.artist_to_idx = {artist_id: idx for idx, artist_id in enumerate(unique_artists)}
        self.idx_to_artist = {idx: artist_id for artist_id, idx in self.artist_to_idx.items()}
        
        # Μετατροπή σε indices
        filtered_df['user_idx'] = filtered_df['user_id'].map(self.user_to_idx)
        filtered_df['artist_idx'] = filtered_df['artist_id'].map(self.artist_to_idx)
        
        # Δημιουργία sparse matrix
        n_users = len(unique_users)
        n_artists = len(unique_artists)
        
        with self.logger.timer("Δημιουργία sparse interaction matrix"):
            self.interaction_matrix = csr_matrix(
                (filtered_df['weight'].values, 
                 (filtered_df['user_idx'].values, filtered_df['artist_idx'].values)),
                shape=(n_users, n_artists)
            )
        
        density = self.interaction_matrix.nnz / (n_users * n_artists)
        
        self.logger.info(f"Δημιουργήθηκε πίνακας αλληλεπιδράσεων: {n_users} χρήστες x {n_artists} καλλιτέχνες")
        self.logger.info(f"Πυκνότητα πίνακα: {density:.4f}")
        self.logger.info(f"Συνολικές αλληλεπιδράσεις: {self.interaction_matrix.nnz}")
        
        return self.interaction_matrix
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[csr_matrix, csr_matrix]:
        """
        Διαχωρισμός δεδομένων σε σύνολα εκπαίδευσης και αξιολόγησης
        
        Χρησιμοποιεί τη μέθοδο leave-one-out για κάθε χρήστη, κρατώντας
        ένα μέρος των αλληλεπιδράσεων για αξιολόγηση.
        
        Args:
            test_size (float): Ποσοστό δεδομένων για αξιολόγηση
            random_state (int): Seed για αναπαραγωγιμότητα
            
        Returns:
            Tuple[csr_matrix, csr_matrix]: Πίνακες εκπαίδευσης και αξιολόγησης
        """
        if self.interaction_matrix is None:
            raise ValueError("Πρέπει πρώτα να δημιουργήσετε τον πίνακα αλληλεπιδράσεων")
        
        with self.logger.timer("Διαχωρισμός δεδομένων σε train/test"):
            np.random.seed(random_state)
            
            train_matrix = self.interaction_matrix.copy()
            test_matrix = csr_matrix(self.interaction_matrix.shape)
            
            n_users = self.interaction_matrix.shape[0]
            progress_bar = self.logger.create_progress_bar(
                "train_test_split", 
                n_users, 
                "Διαχωρισμός χρηστών"
            )
            
            # Για κάθε χρήστη, μετακινούμε ένα μέρος των αλληλεπιδράσεων στο test set
            for user_idx in range(n_users):
                user_interactions = self.interaction_matrix[user_idx].nonzero()[1]
                
                if len(user_interactions) > 1:
                    n_test = max(1, int(len(user_interactions) * test_size))
                    test_items = np.random.choice(user_interactions, n_test, replace=False)
                    
                    for item_idx in test_items:
                        # Μετακίνηση από train σε test
                        rating = train_matrix[user_idx, item_idx]
                        train_matrix[user_idx, item_idx] = 0
                        test_matrix[user_idx, item_idx] = rating
                
                if progress_bar:
                    progress_bar.update(1)
                elif user_idx % 100 == 0:
                    self.logger.info(f"Επεξεργασία χρήστη {user_idx}/{n_users}")
            
            self.logger.close_progress_bar("train_test_split")
            
            # Αφαίρεση μηδενικών στοιχείων
            self.logger.info("Αφαίρεση μηδενικών στοιχείων...")
            train_matrix.eliminate_zeros()
            test_matrix.eliminate_zeros()
            
            self.logger.info(f"Train set: {train_matrix.nnz} αλληλεπιδράσεις")
            self.logger.info(f"Test set: {test_matrix.nnz} αλληλεπιδράσεις")
            
            return train_matrix, test_matrix
    
    def get_user_artist_features(self) -> Dict[str, np.ndarray]:
        """
        Εξαγωγή χαρακτηριστικών χρηστών και καλλιτεχνών
        
        Δημιουργεί διανύσματα χαρακτηριστικών για χρήστες και καλλιτέχνες
        βασισμένα στα tags και τις κοινωνικές σχέσεις.
        
        Returns:
            Dict[str, np.ndarray]: Χαρακτηριστικά χρηστών και καλλιτεχνών
        """
        features = {}
        
        if self.user_taggedartists_df is not None and self.tags_df is not None:
            # Δημιουργία tag profiles για καλλιτέχνες
            artist_tags = self.user_taggedartists_df.groupby('artist_id')['tag_id'].apply(list).to_dict()
            features['artist_tags'] = artist_tags
            
            # Δημιουργία tag profiles για χρήστες
            user_tags = self.user_taggedartists_df.groupby('user_id')['tag_id'].apply(list).to_dict()
            features['user_tags'] = user_tags
        
        if self.user_friends_df is not None:
            # Δημιουργία κοινωνικών δικτύων
            user_friends = self.user_friends_df.groupby('user_id')['friend_id'].apply(list).to_dict()
            features['user_friends'] = user_friends
        
        return features
    
    def get_dataset_statistics(self) -> Dict[str, any]:
        """
        Υπολογισμός στατιστικών του dataset
        
        Returns:
            Dict[str, any]: Στατιστικά στοιχεία του dataset
        """
        stats = {}
        
        if self.user_artists_df is not None:
            stats['n_users'] = self.user_artists_df['user_id'].nunique()
            stats['n_artists'] = self.user_artists_df['artist_id'].nunique()
            stats['n_interactions'] = len(self.user_artists_df)
            stats['avg_interactions_per_user'] = self.user_artists_df.groupby('user_id').size().mean()
            stats['avg_interactions_per_artist'] = self.user_artists_df.groupby('artist_id').size().mean()
            stats['sparsity'] = 1 - (stats['n_interactions'] / (stats['n_users'] * stats['n_artists']))
        
        if self.user_taggedartists_df is not None:
            stats['n_tags'] = self.user_taggedartists_df['tag_id'].nunique()
            stats['n_tag_assignments'] = len(self.user_taggedartists_df)
        
        if self.user_friends_df is not None:
            stats['n_friendships'] = len(self.user_friends_df)
            stats['avg_friends_per_user'] = self.user_friends_df.groupby('user_id').size().mean()
        
        return stats 
    
    def get_enhanced_user_features(self) -> Dict[str, np.ndarray]:
        """
        Εξαγωγή εμπλουτισμένων χαρακτηριστικών χρηστών που χρησιμοποιούν
        όλα τα διαθέσιμα columns από το dataset
        
        Returns:
            Dict[str, np.ndarray]: Εμπλουτισμένα χαρακτηριστικά χρηστών
        """
        user_features = {}
        
        if self.user_artists_df is None:
            return user_features
        
        for user_id in self.user_artists_df['user_id'].unique():
            features = []
            
            # Βασικά χαρακτηριστικά από user_artists
            user_data = self.user_artists_df[self.user_artists_df['user_id'] == user_id]
            total_plays = user_data['weight'].sum()
            unique_artists = len(user_data['artist_id'].unique())
            avg_plays = total_plays / unique_artists if unique_artists > 0 else 0
            features.extend([total_plays, unique_artists, avg_plays])
            
            # Χρονικά χαρακτηριστικά από user_taggedartists (day, month, year)
            temporal_features = self._get_user_temporal_features(user_id)
            features.extend(temporal_features)
            
            # Κοινωνικά χαρακτηριστικά από user_friends
            social_features = self._get_user_social_features(user_id)
            features.extend(social_features)
            
            user_features[user_id] = np.array(features)
        
        return user_features
    
    def get_enhanced_artist_features(self) -> Dict[str, np.ndarray]:
        """
        Εξαγωγή εμπλουτισμένων χαρακτηριστικών καλλιτεχνών που χρησιμοποιούν
        όλα τα διαθέσιμα columns από το dataset
        
        Returns:
            Dict[str, np.ndarray]: Εμπλουτισμένα χαρακτηριστικά καλλιτεχνών
        """
        artist_features = {}
        
        if self.user_artists_df is None or self.artists_df is None:
            return artist_features
        
        for artist_id in self.user_artists_df['artist_id'].unique():
            features = []
            
            # Βασικά χαρακτηριστικά από user_artists
            artist_data = self.user_artists_df[self.user_artists_df['artist_id'] == artist_id]
            total_plays = artist_data['weight'].sum()
            unique_users = len(artist_data['user_id'].unique())
            avg_plays = total_plays / unique_users if unique_users > 0 else 0
            features.extend([total_plays, unique_users, avg_plays])
            
            # URL χαρακτηριστικά από artists (url, picture_url)
            url_features = self._get_artist_url_features(artist_id)
            features.extend(url_features)
            
            # Tag χαρακτηριστικά
            tag_features = self._get_artist_tag_features(artist_id)
            features.extend(tag_features)
            
            artist_features[artist_id] = np.array(features)
        
        return artist_features
    
    def _get_user_temporal_features(self, user_id: int) -> List[float]:
        """
        Εξαγωγή χρονικών χαρακτηριστικών χρήστη από day, month, year columns
        
        Args:
            user_id (int): ID χρήστη
            
        Returns:
            List[float]: Χρονικά χαρακτηριστικά [avg_day, avg_month, year_range, activity_months]
        """
        if self.user_taggedartists_df is None:
            return [0.0, 0.0, 0.0, 0.0]
        
        user_temporal_data = self.user_taggedartists_df[
            self.user_taggedartists_df['user_id'] == user_id
        ]
        
        if user_temporal_data.empty:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Μέσος όρος ημέρας (1-31)
        avg_day = user_temporal_data['day'].mean()
        
        # Μέσος όρος μήνα (1-12) 
        avg_month = user_temporal_data['month'].mean()
        
        # Εύρος ετών δραστηριότητας
        year_range = user_temporal_data['year'].max() - user_temporal_data['year'].min()
        
        # Αριθμός μοναδικών μηνών δραστηριότητας
        activity_months = len(user_temporal_data[['month', 'year']].drop_duplicates())
        
        return [avg_day, avg_month, float(year_range), float(activity_months)]
    
    def _get_user_social_features(self, user_id: int) -> List[float]:
        """
        Εξαγωγή κοινωνικών χαρακτηριστικών χρήστη
        
        Args:
            user_id (int): ID χρήστη
            
        Returns:
            List[float]: Κοινωνικά χαρακτηριστικά [num_friends, is_popular]
        """
        if self.user_friends_df is None:
            return [0.0, 0.0]
        
        # Αριθμός φίλων
        num_friends = len(self.user_friends_df[self.user_friends_df['user_id'] == user_id])
        
        # Αριθμός ατόμων που τον έχουν ως φίλο (δημοτικότητα)
        num_followers = len(self.user_friends_df[self.user_friends_df['friend_id'] == user_id])
        
        # Δείκτης δημοτικότητας (αν είναι πάνω από τον μέσο όρο)
        avg_followers = self.user_friends_df['friend_id'].value_counts().mean()
        is_popular = 1.0 if num_followers > avg_followers else 0.0
        
        return [float(num_friends), is_popular]
    
    def _get_artist_url_features(self, artist_id: int) -> List[float]:
        """
        Εξαγωγή χαρακτηριστικών από URL και picture_url columns
        
        Args:
            artist_id (int): ID καλλιτέχνη
            
        Returns:
            List[float]: URL χαρακτηριστικά [has_url, has_picture, url_length]
        """
        if self.artists_df is None:
            return [0.0, 0.0, 0.0]
        
        artist_data = self.artists_df[self.artists_df['artist_id'] == artist_id]
        
        if artist_data.empty:
            return [0.0, 0.0, 0.0]
        
        artist_row = artist_data.iloc[0]
        
        # Έλεγχος αν υπάρχει URL
        has_url = 1.0 if pd.notna(artist_row.get('url', '')) and artist_row.get('url', '') != '' else 0.0
        
        # Έλεγχος αν υπάρχει picture URL
        has_picture = 1.0 if pd.notna(artist_row.get('picture_url', '')) and artist_row.get('picture_url', '') != '' else 0.0
        
        # Μήκος URL (ως ένδειξη πληρότητας πληροφοριών)
        url_length = len(str(artist_row.get('url', ''))) if pd.notna(artist_row.get('url', '')) else 0.0
        
        return [has_url, has_picture, url_length]
    
    def _get_artist_tag_features(self, artist_id: int) -> List[float]:
        """
        Εξαγωγή χαρακτηριστικών από tags
        
        Args:
            artist_id (int): ID καλλιτέχνη
            
        Returns:
            List[float]: Tag χαρακτηριστικά [num_tags, tag_diversity]
        """
        if self.user_taggedartists_df is None:
            return [0.0, 0.0]
        
        artist_tags = self.user_taggedartists_df[
            self.user_taggedartists_df['artist_id'] == artist_id
        ]
        
        if artist_tags.empty:
            return [0.0, 0.0]
        
        # Αριθμός μοναδικών tags
        num_tags = artist_tags['tag_id'].nunique()
        
        # Ποικιλομορφία tags (αριθμός διαφορετικών χρηστών που έβαλαν tags)
        tag_diversity = artist_tags['user_id'].nunique()
        
        return [float(num_tags), float(tag_diversity)]