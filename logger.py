"""
Σύστημα Καταγραφής και Παρακολούθησης Προόδου
===========================================

Αυτό το module παρέχει ένα ολοκληρωμένο σύστημα καταγραφής (logging) και 
παρακολούθησης προόδου για το σύστημα σύστασης μουσικής. Περιλαμβάνει:

- Detailed logging με διαφορετικά επίπεδα (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Progress bars για εκπαίδευση μοντέλων και επεξεργασία δεδομένων
- Error handling και exception tracking
- Performance monitoring και timing
- Structured logging με timestamps και context

Βιβλιογραφικές Αναφορές:
- Python Logging Cookbook (Python Software Foundation)
- Best Practices for Application Logging 
"""

import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Προειδοποίηση: tqdm δεν είναι διαθέσιμο. Εγκαταστήστε με: pip install tqdm")


class MusicRecommendationLogger:
    """
    Κεντρικό σύστημα καταγραφής για το σύστημα σύστασης μουσικής
    
    Αυτή η κλάση παρέχει ένα ολοκληρωμένο σύστημα καταγραφής που υποστηρίζει:
    - Πολλαπλά επίπεδα logging (console και file)
    - Structured logging με JSON format
    - Performance monitoring
    - Error tracking με stack traces
    - Progress tracking για μακροχρόνιες διεργασίες
    """
    
    def __init__(self, name: str = "MusicRecommendation", log_dir: str = "logs"):
        """
        Αρχικοποίηση του logging συστήματος
        
        Args:
            name (str): Όνομα του logger
            log_dir (str): Φάκελος για αποθήκευση log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Δημιουργία timestamp για unique log files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Αρχικοποίηση loggers
        self.logger = self._setup_logger()
        self.performance_logger = self._setup_performance_logger()
        
        # Performance tracking
        self.timers = {}
        self.counters = {}
        
        # Progress bars storage
        self.progress_bars = {}
        
        self.logger.info("="*80)
        self.logger.info("ΕΚΚΙΝΗΣΗ ΣΥΣΤΗΜΑΤΟΣ ΣΥΣΤΑΣΗΣ ΜΟΥΣΙΚΗΣ")
        self.logger.info("="*80)
        self.logger.info(f"Logger αρχικοποιήθηκε: {datetime.now()}")
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Ρύθμιση του κύριου logger
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Αφαίρεση υπαρχόντων handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler με χρωματισμένο output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler για detailed logs
        log_file = self.log_dir / f"music_recommendation_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_file = self.log_dir / f"errors_{self.timestamp}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """
        Ρύθμιση του performance logger
        
        Returns:
            logging.Logger: Performance logger instance
        """
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.setLevel(logging.INFO)
        
        # Performance log file
        perf_file = self.log_dir / f"performance_{self.timestamp}.json"
        perf_handler = logging.FileHandler(perf_file, encoding='utf-8')
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter('%(message)s')
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        return perf_logger
    
    def debug(self, message: str, **kwargs):
        """Καταγραφή debug μηνύματος"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Καταγραφή info μηνύματος"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Καταγραφή warning μηνύματος"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Καταγραφή error μηνύματος με optional exception"""
        if exception:
            message += f"\nException: {str(exception)}\nTraceback:\n{traceback.format_exc()}"
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Καταγραφή critical μηνύματος"""
        if exception:
            message += f"\nException: {str(exception)}\nTraceback:\n{traceback.format_exc()}"
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """
        Καταγραφή μηνύματος με επιπλέον context
        
        Args:
            level (int): Logging level
            message (str): Μήνυμα προς καταγραφή
            **kwargs: Επιπλέον context information
        """
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {context}"
        
        self.logger.log(level, message)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """
        Καταγραφή performance metrics
        
        Args:
            operation (str): Όνομα της λειτουργίας
            duration (float): Διάρκεια σε δευτερόλεπτα
            **metrics: Επιπλέον μετρικές
        """
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_seconds": duration,
            **metrics
        }
        
        self.performance_logger.info(json.dumps(perf_data, ensure_ascii=False))
        self.info(f"Performance | {operation}: {duration:.2f}s", **metrics)
    
    @contextmanager
    def timer(self, operation: str, **context):
        """
        Context manager για timing operations
        
        Args:
            operation (str): Όνομα της λειτουργίας
            **context: Επιπλέον context
        """
        start_time = time.time()
        self.info(f"Έναρξη: {operation}", **context)
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            self.error(f"Σφάλμα στη λειτουργία: {operation}", exception=e, duration=duration)
            raise
        else:
            duration = time.time() - start_time
            self.log_performance(operation, duration, **context)
            self.info(f"Ολοκλήρωση: {operation} ({duration:.2f}s)", **context)
    
    def create_progress_bar(self, name: str, total: int, description: str = "") -> Optional[Any]:
        """
        Δημιουργία progress bar
        
        Args:
            name (str): Όνομα του progress bar
            total (int): Συνολικός αριθμός items
            description (str): Περιγραφή
            
        Returns:
            Progress bar object ή None αν δεν είναι διαθέσιμο
        """
        if not TQDM_AVAILABLE:
            self.info(f"Progress: {description} (0/{total})")
            return None
        
        pbar = tqdm(
            total=total,
            desc=description,
            unit="items",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        self.progress_bars[name] = pbar
        self.info(f"Δημιουργία progress bar: {name} - {description}")
        return pbar
    
    def update_progress(self, name: str, increment: int = 1, description: str = None):
        """
        Ενημέρωση progress bar
        
        Args:
            name (str): Όνομα του progress bar
            increment (int): Αύξηση
            description (str): Νέα περιγραφή
        """
        if name in self.progress_bars and self.progress_bars[name] is not None:
            pbar = self.progress_bars[name]
            pbar.update(increment)
            if description:
                pbar.set_description(description)
        else:
            # Fallback logging αν δεν υπάρχει progress bar
            if hasattr(self, f'_progress_{name}'):
                current = getattr(self, f'_progress_{name}') + increment
            else:
                current = increment
            setattr(self, f'_progress_{name}', current)
            
            if description:
                self.info(f"Progress: {description} ({current})")
    
    def close_progress_bar(self, name: str):
        """
        Κλείσιμο progress bar
        
        Args:
            name (str): Όνομα του progress bar
        """
        if name in self.progress_bars and self.progress_bars[name] is not None:
            self.progress_bars[name].close()
            del self.progress_bars[name]
            self.info(f"Κλείσιμο progress bar: {name}")
    
    def log_dataset_stats(self, stats: Dict[str, Any]):
        """
        Καταγραφή στατιστικών dataset
        
        Args:
            stats (Dict[str, Any]): Στατιστικά δεδομένων
        """
        self.info("="*60)
        self.info("ΣΤΑΤΙΣΤΙΚΑ DATASET")
        self.info("="*60)
        
        for key, value in stats.items():
            if isinstance(value, float):
                self.info(f"{key}: {value:.4f}")
            else:
                self.info(f"{key}: {value}")
    
    def log_model_training_start(self, model_name: str, parameters: Dict[str, Any]):
        """
        Καταγραφή έναρξης εκπαίδευσης μοντέλου
        
        Args:
            model_name (str): Όνομα μοντέλου
            parameters (Dict[str, Any]): Παράμετροι μοντέλου
        """
        self.info("="*60)
        self.info(f"ΕΝΑΡΞΗ ΕΚΠΑΙΔΕΥΣΗΣ: {model_name}")
        self.info("="*60)
        
        for param, value in parameters.items():
            self.info(f"  {param}: {value}")
    
    def log_model_training_end(self, model_name: str, duration: float, metrics: Dict[str, float]):
        """
        Καταγραφή ολοκλήρωσης εκπαίδευσης μοντέλου
        
        Args:
            model_name (str): Όνομα μοντέλου
            duration (float): Διάρκεια εκπαίδευσης
            metrics (Dict[str, float]): Μετρικές απόδοσης
        """
        self.info("="*60)
        self.info(f"ΟΛΟΚΛΗΡΩΣΗ ΕΚΠΑΙΔΕΥΣΗΣ: {model_name}")
        self.info(f"Διάρκεια: {duration:.2f} δευτερόλεπτα")
        self.info("="*60)
        
        for metric, value in metrics.items():
            self.info(f"  {metric}: {value:.4f}")
    
    def log_evaluation_results(self, model_name: str, results: Dict[str, float]):
        """
        Καταγραφή αποτελεσμάτων αξιολόγησης
        
        Args:
            model_name (str): Όνομα μοντέλου
            results (Dict[str, float]): Αποτελέσματα αξιολόγησης
        """
        self.info("="*60)
        self.info(f"ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ: {model_name}")
        self.info("="*60)
        
        # Ομαδοποίηση μετρικών
        recall_metrics = {k: v for k, v in results.items() if k.startswith('Recall')}
        ndcg_metrics = {k: v for k, v in results.items() if k.startswith('NDCG')}
        hit_rate_metrics = {k: v for k, v in results.items() if k.startswith('Hit_Rate')}
        other_metrics = {k: v for k, v in results.items() 
                        if not any(k.startswith(prefix) for prefix in ['Recall', 'NDCG', 'Hit_Rate'])}
        
        if recall_metrics:
            self.info("Recall@K:")
            for metric, value in recall_metrics.items():
                self.info(f"  {metric}: {value:.4f}")
        
        if ndcg_metrics:
            self.info("NDCG@K:")
            for metric, value in ndcg_metrics.items():
                self.info(f"  {metric}: {value:.4f}")
        
        if hit_rate_metrics:
            self.info("Hit Rate@K:")
            for metric, value in hit_rate_metrics.items():
                self.info(f"  {metric}: {value:.4f}")
        
        if other_metrics:
            self.info("Άλλες Μετρικές:")
            for metric, value in other_metrics.items():
                self.info(f"  {metric}: {value:.4f}")
    
    def close(self):
        """Κλείσιμο όλων των progress bars και handlers"""
        # Κλείσιμο progress bars
        for name in list(self.progress_bars.keys()):
            self.close_progress_bar(name)
        
        self.info("="*80)
        self.info("ΤΕΡΜΑΤΙΣΜΟΣ ΣΥΣΤΗΜΑΤΟΣ ΣΥΣΤΑΣΗΣ ΜΟΥΣΙΚΗΣ")
        self.info("="*80)
        
        # Κλείσιμο handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class ColoredFormatter(logging.Formatter):
    """
    Formatter για χρωματισμένο console output
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Προσθήκη χρώματος
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


# Global logger instance
_global_logger = None

def get_logger() -> MusicRecommendationLogger:
    """
    Λήψη του global logger instance
    
    Returns:
        MusicRecommendationLogger: Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MusicRecommendationLogger()
    return _global_logger

def close_logger():
    """Κλείσιμο του global logger"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None 