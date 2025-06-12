"""
Πακέτο Μοντέλων Σύστασης Μουσικής
================================

Αυτό το πακέτο περιλαμβάνει τις υλοποιήσεις των βασικών αλγορίθμων σύστασης:

Collaborative Filtering:
- UserKNN: User-based K-Nearest Neighbors
- ItemKNN: Item-based K-Nearest Neighbors  
- SVD: Singular Value Decomposition

Deep Learning:
- NeuMF: Neural Collaborative Filtering
- MultVAE: Multinomial Variational Autoencoder

Βιβλιογραφικές Αναφορές:
- Linden et al. (2003): "Amazon.com recommendations: item-to-item collaborative filtering"
- Koren et al. (2009): "Matrix Factorization Techniques for Recommender Systems"  
- He et al. (2017): "Neural Collaborative Filtering"
- Liang et al. (2018): "Variational Autoencoders for Collaborative Filtering"
- Bellogín et al. (2010): "A study of heterogeneity in recommendations for a social music service"
- Majumdar (2013): "Music Recommendations based on Implicit Feedback and Social Circles: The Last FM Data Set"
- Schedl (2019): "Deep Learning in Music Recommendation Systems"
- Oramas et al. (2017): "A Deep Multimodal Approach for Cold-start Music Recommendation"
"""

from .collaborative_filtering import UserKNN, ItemKNN, SVDRecommender
from .deep_learning import NeuMFRecommender, MultVAERecommender

__all__ = [
    'UserKNN',
    'ItemKNN', 
    'SVDRecommender',
    'NeuMFRecommender',
    'MultVAERecommender'
] 