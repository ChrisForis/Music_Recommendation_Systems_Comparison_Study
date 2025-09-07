"""
Περιεχομενοκεντρικός Αλγόριθμος Σύστασης με Sentence-BERT
========================================================

Το παρόν module υλοποιεί έναν αλγόριθμο σύστασης καλλιτεχνών βασισμένο αποκλειστικά
στα κειμενικά μεταδεδομένα του HetRec2011 Last.fm dataset (ονόματα καλλιτεχνών και
ετικέτες χρήστη). Αξιοποιεί προ-εκπαιδευμένο Sentence-BERT ώστε να παραγάγει
υψηλού επιπέδου ενσωματώσεις (embeddings) για κάθε καλλιτέχνη και στη συνέχεια
υπολογίζει συστηνόμενα αντικείμενα μέσω συνημιτονοειδούς ομοιότητας.

Βασικές ιδιότητες
-----------------
• Διατηρεί πλήρη συμβατότητα με το πλαίσιο αξιολόγησης του ``evaluation.py``·
  εκθέτει μέθοδο ``recommend`` με ίδια υπογραφή με τους αλγορίθμους του
  ``collaborative_filtering.py`` ώστε να παράγονται άμεσα Recall@K, NDCG@K κ.λπ.
• Περιλαμβάνει caching των embeddings σε δίσκο για δραστική μείωση χρόνου
  επανεκκίνησης.
• Ενσωματώνει εκτενή χειρισμό σφαλμάτων και καταγραφή γεγονότων μέσω του
  κεντρικού logger της εφαρμογής.

Βιβλιογραφικές παραπομπές
------------------------
• Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks*. EMNLP. DOI: 10.18653/v1/D19-1410
• Cantador et al. (2011). *Last.fm HetRec 2011 Dataset*. RecSys Workshop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

# Sentence-BERT (προαιρετικό – αν λείπει γίνεται graceful degradation)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from logger import get_logger
from data_loader import DataLoader

logger = get_logger()

RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# -----------------------------------------------------------------------------

def _build_artist_corpus(
    artists_df: pd.DataFrame,
    user_taggedartists_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    *,
    min_tag_freq: int = 2,
) -> Dict[int, str]:
    """Δημιουργεί κειμενική περιγραφή για κάθε καλλιτέχνη.

    Η συνάρτηση ενοποιεί το όνομα του καλλιτέχνη με τις πιο συχνές ετικέτες του
    (π.χ. «rock», «britpop») ώστε να σχηματίσει ένα σύντομο κείμενο εισόδου στο
    Sentence-BERT. Ετικέτες με συχνότητα < *min_tag_freq* αγνοούνται για
    αποφυγή θορύβου.

    Επιστρέφεται λεξικό *artist_id → κείμενο*.
    """
    # Χάρτης tag_id → tag_value (περιγραφή)
    tag_id_to_val = tags_df.set_index("tag_id")["tag_value"].str.lower().to_dict()

    # Συχνότητες ετικετών ανά καλλιτέχνη
    artist_tag_freq = (
        user_taggedartists_df.groupby(["artist_id", "tag_id"]).size().reset_index(name="freq")
    )

    corpus: Dict[int, str] = {}
    missing_name = 0

    for artist_id, group in artist_tag_freq.groupby("artist_id"):
        # Φιλτράρισμα σπάνιων tags
        top_tags = group[group["freq"] >= min_tag_freq]["tag_id"].tolist()
        tag_tokens = [tag_id_to_val.get(tid, "") for tid in top_tags if tid in tag_id_to_val]
        tag_tokens = [tok for tok in tag_tokens if tok]  # αποβολή κενών

        # Όνομα καλλιτέχνη (υπάρχει πάντα στο artists.dat)
        name_row = artists_df.loc[artists_df["artist_id"] == artist_id, "name"]
        artist_name = str(name_row.iloc[0]) if not name_row.empty else "Unknown"
        if name_row.empty:
            missing_name += 1

        # Τελικό κείμενο
        text = f"{artist_name.lower()}. tags: {', '.join(tag_tokens)}" if tag_tokens else artist_name.lower()
        corpus[artist_id] = text

    if missing_name:
        logger.warning(f"Καλλιτέχνες χωρίς όνομα στο artists.dat: {missing_name}")

    return corpus


def _encode_texts(
    texts: List[str],
    *,
    model_name: str,
    device: str,
    batch_size_cpu: int = 32,
    batch_size_gpu: int = 128,
) -> np.ndarray:
    """Μετατρέπει λίστα κειμένων σε Sentence-BERT embeddings."""
    if SentenceTransformer is None:
        rng = np.random.default_rng(42)
        logger.warning("Sentence-BERT δεν εγκατεστημένο – δημιουργούνται τυχαία embeddings.")
        return rng.standard_normal((len(texts), 384), dtype=np.float32)

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:  # pragma: no cover
        logger.error("Αποτυχία φόρτωσης Sentence-BERT", exception=e)
        raise

    bs = batch_size_gpu if device.startswith("cuda") else batch_size_cpu
    logger.info(f"Υπολογισμός Sentence-BERT embeddings σε συσκευή {device} (batch={bs})…")

    emb = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=True,
        device=device,
    )
    return emb.astype(np.float32)

# -----------------------------------------------------------------------------
# ΚΥΡΙΑ ΚΛΑΣΗ ΣΥΝΙΣΤΩΣΑΣ
# -----------------------------------------------------------------------------

class BERTTagContentRecommender:
    """Σύσταση καλλιτεχνών με χρήση Sentence-BERT επί ετικετών Last.fm."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        cache_file: Path | str = RESULTS_DIR / "artist_tag_embeddings.npy",
        popularity_penalty: float = 0.3,
        fine_tune_epochs: int = 0
    ) -> None:
        self.model_name = model_name
        self.device = device if isinstance(device, str) else str(device)
        self.cache_file = Path(cache_file)
        self.popularity_penalty = popularity_penalty
        self.fine_tune_epochs = max(0, int(fine_tune_epochs))

        # Θα αρχικοποιηθούν στο fit
        self.E: Optional[np.ndarray] = None  # [n_items × d]
        self.E_unit: Optional[np.ndarray] = None
        self.A: Optional[csr_matrix] = None  # interaction matrix
        self.item_popularity: Optional[np.ndarray] = None
        self.idx_to_artist: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(
        self,
        data_loader: DataLoader,
        *,
        train_matrix: Optional[csr_matrix] = None,
        min_tag_freq: int = 2,
    ) -> "BERTTagContentRecommender":
        """Κατασκευάζει τα embeddings και προϋπολογίζει βοηθητικές δομές.

        Καλεί εσωτερικά τις μεθόδους του ``DataLoader`` ώστε να είναι πλήρως
        ευθυγραμμισμένο με το interaction matrix που χρησιμοποιείται σε όλους
        τους αλγορίθμους.
        """
        # ------------------------------------------------------------------
        # 1. Φόρτωση και προεπεξεργασία δεδομένων
        # ------------------------------------------------------------------
        with logger.timer("Φόρτωση Last.fm dataset για BERT recommender"):
            data_loader.load_all_data()
            # Αν έχει ήδη δημιουργηθεί train_matrix σε ανώτερο επίπεδο, χρησιμοποίησέ το ώστε
            # να αποφευχθεί διαρροή δεδομένων (data leakage) από το test set.
            A = (
                train_matrix
                if train_matrix is not None
                else data_loader.create_interaction_matrix(min_interactions=5)
            )
            self.A = A  # Sparsity διατηρείται για later use
            self.idx_to_artist = data_loader.idx_to_artist

        # ------------------------------------------------------------------
        # 2. Δημιουργία / φόρτωση embeddings καλλιτεχνών
        # ------------------------------------------------------------------
        with logger.timer("Δημιουργία κειμενικού corpus καλλιτεχνών"):
                        corpus_map = _build_artist_corpus(
                data_loader.artists_df,
                data_loader.user_taggedartists_df,
                data_loader.tags_df,
                min_tag_freq=min_tag_freq,
            )

        artist_ids_sorted = [data_loader.idx_to_artist[i] for i in range(len(data_loader.idx_to_artist))]
        texts = [corpus_map.get(artist_id, "") for artist_id in artist_ids_sorted]

        # ------------------------------------------------------------------
        # fine-tune του Sentence-BERT στο Last.fm domain
        # ------------------------------------------------------------------
        model_path = self.model_name
        if self.fine_tune_epochs > 0 and SentenceTransformer is not None:
            finetuned_dir = RESULTS_DIR / "finetuned_bert_lastfm"
            if (finetuned_dir / 'config.json').exists():
                logger.info(f"Φόρτωση fine-tuned Sentence-BERT από {finetuned_dir}")
                model_path = str(finetuned_dir)
            else:
                if finetuned_dir.exists():
                    import shutil
                    shutil.rmtree(finetuned_dir)
                logger.info(f"Έναρξη fine-tune Sentence-BERT για {self.fine_tune_epochs} epochs…")
                try:
                    from sentence_transformers import InputExample, losses
                    from torch.utils.data import DataLoader as TorchLoader
                except ImportError as e:
                    logger.error(f"Αποτυχία import για fine-tuning: {e}")
                    logger.info("Συνεχίζουμε με το προ-εκπαιδευμένο μοντέλο...")
                    model_path = self.model_name
                    return self

                # Έλεγχος αν ο χρήστης έχει αλληλεπιδράσει με τουλάχιστον 2 καλλιτέχνες
                rng = np.random.default_rng(42)
                examples = []
                for u in range(self.A.shape[0]):
                    items = np.where(self.A[u].toarray().flatten() > 0)[0]
                    if len(items) < 2:
                        continue
                    a_idx, b_idx = rng.choice(items, size=2, replace=False)
                    examples.append(InputExample(texts=[texts[a_idx], texts[b_idx]], label=1.0))

                if len(examples) < 100:
                    logger.warning("Λιγότερα από 100 θετικά ζεύγη – παράλειψη fine-tune.")
                else:
                    try:
                        st_model = SentenceTransformer(self.model_name, device=self.device)
                        train_loader = TorchLoader(examples, batch_size=32, shuffle=True)
                        loss_fn = losses.MultipleNegativesRankingLoss(st_model)
                        
                        # Χρήση προσωρινού φακέλου για checkpoints για αποφυγή προβλημάτων
                        import tempfile
                        import os
                        
                        # Απενεργοποίηση TensorBoard logs για αποφυγή προβλημάτων με φακέλους
                        os.environ['DISABLE_TENSORBOARD'] = '1'
                        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Απενεργοποίηση TensorFlow warnings
                        os.environ['TENSORBOARD_DISABLE'] = '1'
                        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                        
                        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                            st_model.fit(
                                train_objectives=[(train_loader, loss_fn)],
                                epochs=self.fine_tune_epochs,
                                show_progress_bar=True,
                                output_path=str(finetuned_dir),
                                checkpoint_path=temp_checkpoint_dir,
                                use_amp=False,
                                warmup_steps=0,
                            )
                        logger.info(f"Αποθήκευση fine-tuned μοντέλου σε {finetuned_dir}")
                        model_path = str(finetuned_dir)
                    except Exception as e:
                        logger.error(f"Σφάλμα κατά το fine-tuning: {e}")
                        logger.info("Συνεχίζουμε με το προ-εκπαιδευμένο μοντέλο...")
                        model_path = self.model_name

        # Αν υπάρχει cache και αντιστοιχεί σε σωστές διαστάσεις, χρησιμοποιήσέ το
        try:
            if self.cache_file.exists():
                try:
                    emb = np.load(self.cache_file)
                    if emb.shape[0] != len(texts):
                        raise ValueError("Cache dimension mismatch – επανυπολογισμός απαιτείται.")
                    logger.info(f"Φόρτωση cached embeddings από {self.cache_file} ({emb.shape})")
                except Exception as e:
                    logger.warning("Αποτυχία φόρτωσης cache – επανυπολογισμός.", exception=e)
                    emb = _encode_texts(texts, model_name=model_path, device=self.device)
                    np.save(self.cache_file, emb)
                    logger.info(f"Αποθηκεύτηκαν embeddings σε {self.cache_file.relative_to(Path('.'))}")
            else:
                emb = _encode_texts(texts, model_name=model_path, device=self.device)
                np.save(self.cache_file, emb)
                logger.info(f"Αποθηκεύτηκαν embeddings σε {self.cache_file.relative_to(Path('.'))}")
        except Exception as e:
            logger.error(f"Σφάλμα κατά τη δημιουργία embeddings: {e}")
            # Fallback σε τυχαία embeddings αν αποτύχει το Sentence-BERT
            logger.warning("Fallback σε τυχαία embeddings...")
            rng = np.random.default_rng(42)
            emb = rng.standard_normal((len(texts), 384), dtype=np.float32)
            np.save(self.cache_file, emb)
            logger.info(f"Αποθηκεύτηκαν τυχαία embeddings σε {self.cache_file.relative_to(Path('.'))}")

        # Κανονικοποίηση
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        self.E = emb.astype(np.float32)
        self.E_unit = self.E / norm.astype(np.float32)

        # Δημοτικότητα καλλιτεχνών (χρήσιμη για long-tail boost)
        if self.popularity_penalty > 0:
            self.item_popularity = np.array((A > 0).sum(axis=0)).flatten() + 1

        return self

    # ------------------------------------------------------------------
    # Interface συμβατό με RecommendationEvaluator
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        *,
        exclude_seen: bool = True,
        interaction_matrix: Optional[csr_matrix] = None,
    ) -> List[Tuple[int, float]]:
        """Επιστρέφει λίστα (item_idx, score) ταξινομημένη φθίνουσα."""
        if self.E_unit is None or self.A is None:
            raise RuntimeError("Πρέπει πρώτα να καλέσετε fit() πριν το recommend().")

        A = interaction_matrix if interaction_matrix is not None else self.A
        if A is None:
            raise ValueError("Δεν παρέχεται interaction matrix.")
        if user_id >= A.shape[0]:
            raise IndexError("user_id εκτός ορίων")

        user_items = A[user_id].nonzero()[1]
        if len(user_items) == 0:
            logger.warning(f"Χρήστης {user_id} χωρίς αλληλεπιδράσεις – καμία σύσταση.")
            return []

        profile = self.E_unit[user_items].mean(axis=0, keepdims=True)  # 1×d
        sims_content = (self.E_unit @ profile.T).flatten()

        if exclude_seen and len(user_items) > 0:
            sims_content[user_items] = -np.inf

        if self.popularity_penalty > 0 and self.item_popularity is not None:
            sims_content = sims_content / (self.item_popularity ** self.popularity_penalty)

        top_idx = np.argsort(sims_content)[::-1][:n_recommendations]
        return [(int(i), float(sims_content[i])) for i in top_idx]       

# -----------------------------------------------------------------------------
# ΣΥΝΤΟΜΟ DEMO
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Demo Sentence-BERT Recommender (Last.fm) ===")
    dl = DataLoader()
    rec = BERTTagContentRecommender(device="cpu").fit(dl)
    user0_recs = rec.recommend(user_id=0, n_recommendations=5)
    print("Συστάσεις για χρήστη 0:")
    print(user0_recs)
