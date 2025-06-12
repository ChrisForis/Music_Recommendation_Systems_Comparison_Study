"""
Αλγόριθμοι Βαθιάς Μάθησης για Σύσταση Μουσικής
==============================================

Αυτό το module περιλαμβάνει τις υλοποιήσεις των αλγορίθμων βαθιάς μάθησης
για σύσταση μουσικής:

1. NeuMF: Neural Collaborative Filtering
2. MultVAE: Variational Autoencoder για Collaborative Filtering

Βιβλιογραφικές Αναφορές:
- He et al. (2017): "Neural Collaborative Filtering" 
  arXiv:1708.05031 [cs]. DOI: 10.48550/arXiv.1708.05031
- Liang et al. (2018): "Variational Autoencoders for Collaborative Filtering"
  arXiv:1802.05814 [stat]. DOI: 10.48550/arXiv.1802.05814
- Schedl (2019): "Deep Learning in Music Recommendation Systems"
  Frontiers in Applied Mathematics and Statistics, Vol. 5, Article 44. DOI: 10.3389/fams.2019.00044
- Oramas et al. (2017): "A Deep Multimodal Approach for Cold-start Music Recommendation"
  Proceedings of the 2nd Workshop on Deep Learning for Recommender Systems (DLRS 2017), 
  pp. 32-37. DOI: 10.1145/3125486.3125492
- Fessahaye et al. (2019): "T-RECSYS: A Novel Music Recommendation System Using Deep Learning"
  2019 IEEE International Conference on Consumer Electronics (ICCE), 
  pp. 1-6. DOI: 10.1109/ICCE.2019.8662028

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class InteractionDataset(Dataset):
    """
    Dataset κλάση για τις αλληλεπιδράσεις χρηστών-αντικειμένων
    
    Μετατρέπει τον sparse matrix σε format κατάλληλο για PyTorch DataLoader.
    """
    
    def __init__(self, interaction_matrix: csr_matrix, negative_sampling: bool = True, 
                 num_negatives: int = 4):
        """
        Αρχικοποίηση του dataset
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
            negative_sampling (bool): Αν θα γίνει negative sampling
            num_negatives (int): Αριθμός αρνητικών δειγμάτων ανά θετικό
        """
        self.interaction_matrix = interaction_matrix
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        # Εξαγωγή θετικών αλληλεπιδράσεων
        self.users, self.items = interaction_matrix.nonzero()
        self.ratings = interaction_matrix.data
        
        # Κανονικοποίηση ratings στο [0, 1]
        if len(np.unique(self.ratings)) > 2:
            self.ratings = (self.ratings - self.ratings.min()) / (self.ratings.max() - self.ratings.min())
        else:
            self.ratings = (self.ratings > 0).astype(float)
        
        if negative_sampling:
            self._generate_negative_samples()
    
    def _generate_negative_samples(self):
        """Δημιουργία αρνητικών δειγμάτων για εκπαίδευση"""
        negative_users = []
        negative_items = []
        negative_ratings = []
        
        n_users, n_items = self.interaction_matrix.shape
        
        for i in range(len(self.users)):
            user = self.users[i]
            # Βρες αντικείμενα που δεν έχει αλληλεπιδράσει ο χρήστης
            user_items = set(self.interaction_matrix[user].nonzero()[1])
            all_items = set(range(n_items))
            negative_candidates = list(all_items - user_items)
            
            if len(negative_candidates) >= self.num_negatives:
                sampled_negatives = np.random.choice(negative_candidates, 
                                                   self.num_negatives, replace=False)
                negative_users.extend([user] * self.num_negatives)
                negative_items.extend(sampled_negatives)
                negative_ratings.extend([0.0] * self.num_negatives)
        
        # Συνδυασμός θετικών και αρνητικών δειγμάτων
        self.users = np.concatenate([self.users, negative_users])
        self.items = np.concatenate([self.items, negative_items])
        self.ratings = np.concatenate([self.ratings, negative_ratings])
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }


class NeuMF(nn.Module):
    """
    Neural Collaborative Filtering (NeuMF)
    
    Ο αλγόριθμος NeuMF συνδυάζει τη Matrix Factorization με νευρωνικά δίκτυα
    για να μοντελοποιήσει μη-γραμμικές σχέσεις μεταξύ χρηστών και αντικειμένων.
    
    Βιβλιογραφικές Αναφορές:
    - He et al. (2017) "Neural Collaborative Filtering" (arXiv:1708.05031) περιγράφει 
      την ανάπτυξη του NeuMF και τις βελτιώσεις που προσφέρει.
    - Fessahaye et al. (2019) "T-RECSYS: A Novel Music Recommendation System Using Deep Learning"
      εφαρμόζει παρόμοιες τεχνικές deep learning για music recommendation.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        """
        Αρχικοποίηση του NeuMF μοντέλου
        
        Args:
            num_users (int): Αριθμός χρηστών
            num_items (int): Αριθμός αντικειμένων
            embedding_dim (int): Διάσταση των embeddings
            hidden_dims (List[int]): Διαστάσεις των κρυφών επιπέδων
            dropout (float): Ποσοστό dropout
        """
        super(NeuMF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings για Matrix Factorization
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        
        # Embeddings για MLP
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Τελικό επίπεδο που συνδυάζει MF και MLP
        self.final_layer = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        
        # Αρχικοποίηση βαρών
        self._init_weights()
    
    def _init_weights(self):
        """Αρχικοποίηση των βαρών του μοντέλου"""
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass του μοντέλου
        
        Args:
            user_ids (torch.Tensor): IDs χρηστών
            item_ids (torch.Tensor): IDs αντικειμένων
            
        Returns:
            torch.Tensor: Προβλεπόμενες βαθμολογίες
        """
        # Matrix Factorization path
        user_emb_mf = self.user_embedding_mf(user_ids)
        item_emb_mf = self.item_embedding_mf(item_ids)
        mf_output = user_emb_mf * item_emb_mf
        
        # MLP path
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Συνδυασμός MF και MLP
        combined = torch.cat([mf_output, mlp_output], dim=-1)
        output = torch.sigmoid(self.final_layer(combined))
        
        return output.squeeze()


class NeuMFRecommender:
    """
    Wrapper κλάση για τον NeuMF αλγόριθμο
    
    Παρέχει interface συμβατό με τους άλλους αλγόριθμους σύστασης.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2, learning_rate: float = 0.001, batch_size: int = 256,
                 epochs: int = 100, device: str = 'cpu'):
        """
        Αρχικοποίηση του NeuMF Recommender
        
        Args:
            embedding_dim (int): Διάσταση embeddings
            hidden_dims (List[int]): Διαστάσεις κρυφών επιπέδων
            dropout (float): Ποσοστό dropout
            learning_rate (float): Ρυθμός μάθησης
            batch_size (int): Μέγεθος batch
            epochs (int): Αριθμός epochs
            device (str): Συσκευή εκτέλεσης ('cpu' ή 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        
        self.model = None
        self.is_fitted = False
        self.num_users = None
        self.num_items = None
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του NeuMF μοντέλου
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        """
        self.num_users, self.num_items = interaction_matrix.shape
        
        # Δημιουργία μοντέλου
        self.model = NeuMF(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Δημιουργία dataset και dataloader
        dataset = InteractionDataset(interaction_matrix, negative_sampling=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer και loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Εκπαίδευση
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        print("NeuMF εκπαιδεύτηκε επιτυχώς")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Πρόβλεψη βαθμολογιών
        
        Args:
            user_ids (List[int]): Λίστα με IDs χρηστών
            item_ids (List[int]): Λίστα με IDs αντικειμένων
            
        Returns:
            np.ndarray: Προβλεπόμενες βαθμολογίες
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor(user_ids, dtype=torch.long).to(self.device)
            item_tensor = torch.tensor(item_ids, dtype=torch.long).to(self.device)
            predictions = self.model(user_tensor, item_tensor)
            return predictions.cpu().numpy()
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True, interaction_matrix: csr_matrix = None) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων για exclusion
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        # Υπολογισμός scores για όλα τα αντικείμενα
        all_items = list(range(self.num_items))
        user_ids = [user_id] * len(all_items)
        scores = self.predict(user_ids, all_items)
        
        # Εξαίρεση ήδη γνωστών αντικειμένων
        if exclude_seen and interaction_matrix is not None:
            seen_items = set(interaction_matrix[user_id].nonzero()[1])
            for item_id in seen_items:
                scores[item_id] = -np.inf
        
        # Ταξινόμηση και επιστροφή top-N
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations


class MultVAE(nn.Module):
    """
    Multinomial Variational Autoencoder για Collaborative Filtering
    
    Ο αλγόριθμος MultVAE χρησιμοποιεί ένα variational autoencoder για τη
    δημιουργία πιθανιστικών αναπαραστάσεων χρηστών και αντικειμένων.
    
    Βιβλιογραφικές Αναφορές:
    - Liang et al. (2018) "Variational Autoencoders for Collaborative Filtering" (arXiv:1802.05814)
      περιγράφει τη χρήση autoencoders για σύσταση με πιθανιστικά μοντέλα.
    - Oramas et al. (2017) "A Deep Multimodal Approach for Cold-start Music Recommendation"
      εφαρμόζει deep learning τεχνικές για music recommendation με multimodal δεδομένα.
    """
    
    def __init__(self, num_items: int, hidden_dims: List[int] = [600, 200], 
                 latent_dim: int = 200, dropout: float = 0.5):
        """
        Αρχικοποίηση του MultVAE μοντέλου
        
        Args:
            num_items (int): Αριθμός αντικειμένων
            hidden_dims (List[int]): Διαστάσεις κρυφών επιπέδων
            latent_dim (int): Διάσταση του latent space
            dropout (float): Ποσοστό dropout
        """
        super(MultVAE, self).__init__()
        
        self.num_items = num_items
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        input_dim = num_items
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers (μέσος όρος και διασπορά)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.Tanh())
            decoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, num_items))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoding στο latent space
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Μέσος όρος και log διασπορά
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick για το VAE
        
        Args:
            mu (torch.Tensor): Μέσος όρος
            logvar (torch.Tensor): Log διασπορά
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoding από το latent space
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass του VAE
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstruction, μέσος όρος, log διασπορά
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class MultVAERecommender:
    """
    Wrapper κλάση για τον MultVAE αλγόριθμο
    
    Παρέχει interface συμβατό με τους άλλους αλγόριθμους σύστασης.
    """
    
    def __init__(self, hidden_dims: List[int] = [600, 200], latent_dim: int = 200,
                 dropout: float = 0.5, learning_rate: float = 0.001, batch_size: int = 500,
                 epochs: int = 100, beta: float = 1.0, device: str = 'cpu'):
        """
        Αρχικοποίηση του MultVAE Recommender
        
        Args:
            hidden_dims (List[int]): Διαστάσεις κρυφών επιπέδων
            latent_dim (int): Διάσταση latent space
            dropout (float): Ποσοστό dropout
            learning_rate (float): Ρυθμός μάθησης
            batch_size (int): Μέγεθος batch
            epochs (int): Αριθμός epochs
            beta (float): Βάρος του KL divergence term
            device (str): Συσκευή εκτέλεσης
        """
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.device = torch.device(device)
        
        self.model = None
        self.is_fitted = False
        self.num_items = None
    
    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Υπολογισμός του VAE loss (ELBO)
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Μέσος όρος
            logvar (torch.Tensor): Log διασπορά
            
        Returns:
            torch.Tensor: VAE loss
        """
        # Reconstruction loss (multinomial likelihood)
        recon_loss = -torch.sum(F.log_softmax(recon_x, dim=1) * x, dim=1)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        return torch.mean(recon_loss + self.beta * kl_loss)
    
    def fit(self, interaction_matrix: csr_matrix):
        """
        Εκπαίδευση του MultVAE μοντέλου
        
        Args:
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
        """
        self.num_items = interaction_matrix.shape[1]
        
        # Δημιουργία μοντέλου
        self.model = MultVAE(
            num_items=self.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Μετατροπή σε dense matrix και κανονικοποίηση
        interaction_dense = interaction_matrix.toarray().astype(np.float32)
        
        # Κανονικοποίηση ανά χρήστη (multinomial)
        user_sums = interaction_dense.sum(axis=1, keepdims=True)
        user_sums[user_sums == 0] = 1  # Αποφυγή διαίρεσης με μηδέν
        interaction_normalized = interaction_dense / user_sums
        
        # Δημιουργία DataLoader
        dataset = torch.utils.data.TensorDataset(torch.tensor(interaction_normalized))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Εκπαίδευση
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                loss = self.vae_loss(recon_x, x, mu, logvar)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        print("MultVAE εκπαιδεύτηκε επιτυχώς")
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_seen: bool = True, interaction_matrix: csr_matrix = None) -> List[Tuple[int, float]]:
        """
        Παραγωγή συστάσεων για έναν χρήστη
        
        Args:
            user_id (int): ID του χρήστη
            n_recommendations (int): Αριθμός συστάσεων
            exclude_seen (bool): Αν θα εξαιρεθούν τα ήδη γνωστά αντικείμενα
            interaction_matrix (csr_matrix): Πίνακας αλληλεπιδράσεων
            
        Returns:
            List[Tuple[int, float]]: Λίστα με (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")
        
        if interaction_matrix is None:
            raise ValueError("Απαιτείται ο πίνακας αλληλεπιδράσεων για συστάσεις")
        
        self.model.eval()
        with torch.no_grad():
            # Προετοιμασία input για τον χρήστη
            user_vector = interaction_matrix[user_id].toarray().astype(np.float32)
            user_sum = user_vector.sum()
            if user_sum > 0:
                user_vector = user_vector / user_sum
            
            user_tensor = torch.tensor(user_vector).to(self.device)
            
            # Πρόβλεψη
            recon_x, _, _ = self.model(user_tensor)
            scores = recon_x.cpu().numpy().flatten()
            
            # Εξαίρεση ήδη γνωστών αντικειμένων
            if exclude_seen:
                seen_items = set(interaction_matrix[user_id].nonzero()[1])
                for item_id in seen_items:
                    scores[item_id] = -np.inf
            
            # Ταξινόμηση και επιστροφή top-N
            top_items = np.argsort(scores)[::-1][:n_recommendations]
            recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
            
            return recommendations 