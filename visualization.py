"""
Σύστημα Οπτικοποίησης για Συστήματα Σύστασης Μουσικής
====================================================

Αυτό το module παρέχει μοντέρνες, clean οπτικοποιήσεις για την ανάλυση
και σύγκριση των αποτελεσμάτων των αλγορίθμων σύστασης μουσικής.

Χαρακτηριστικά:
- Μοντέρνο design με vibrant χρώματα
- Clean και professional εμφάνιση
- Αυτόματη αποθήκευση σε υψηλή ανάλυση
- Responsive layouts
- Συνεπή χρωματική παλέτα


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Εισαγωγή του logging συστήματος
from logger import get_logger


class ModernVisualizationEngine:
    """
    Κλάση για τη δημιουργία μοντέρνων, clean οπτικοποιήσεων
    
    Παρέχει ένα ολοκληρωμένο σύστημα οπτικοποίησης με συνεπή
    στυλ, vibrant χρώματα και professional εμφάνιση.
    """
    
    def __init__(self, output_dir: str = "plots/"):
        """
        Αρχικοποίηση του visualization engine
        
        Args:
            output_dir (str): Φάκελος αποθήκευσης γραφημάτων
        """
        self.output_dir = output_dir
        self.logger = get_logger()
        
        # Δημιουργία φακέλου αν δεν υπάρχει
        os.makedirs(output_dir, exist_ok=True)
        
        # Ρύθμιση μοντέρνου στυλ
        self._setup_modern_style()
        
        # Ορισμός vibrant χρωματικής παλέτας
        self.colors = {
            'primary': '#2E86AB',      # Μπλε
            'secondary': '#A23B72',    # Μωβ
            'accent': '#F18F01',       # Πορτοκαλί
            'success': '#C73E1D',      # Κόκκινο
            'info': '#6A994E',         # Πράσινο
            'warning': '#F4A261',      # Κίτρινο-πορτοκαλί
            'dark': '#264653',         # Σκούρο πράσινο
            'light': '#E9C46A',       # Ανοιχτό κίτρινο
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
        
        # Παλέτα για πολλαπλά μοντέλα
        self.model_palette = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
            '#6A994E', '#F4A261', '#264653', '#E9C46A',
            '#667eea', '#764ba2', '#f093fb', '#f5576c'
        ]
        
        self.logger.info(f"Αρχικοποίηση ModernVisualizationEngine - Output: {output_dir}")
    
    def _setup_modern_style(self):
        """
        Ρύθμιση μοντέρνου στυλ για τα γραφήματα
        """
        # Ρύθμιση matplotlib για υψηλή ποιότητα
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': False,
            'lines.linewidth': 2.5,
            'lines.markersize': 8
        })
        
        # Ρύθμιση seaborn για μοντέρνο στυλ
        sns.set_style("whitegrid")
        
        sns.set_context("notebook", font_scale=1.1)
    
    def create_model_comparison_chart(self, results_df: pd.DataFrame, 
                                    metric: str = 'Recall@5',
                                    title: Optional[str] = None,
                                    save_name: Optional[str] = None) -> str:
        """
        Δημιουργία μοντέρνου bar chart για σύγκριση μοντέλων
        
        Args:
            results_df (pd.DataFrame): DataFrame με αποτελέσματα μοντέλων
            metric (str): Μετρική προς οπτικοποίηση
            title (str, optional): Τίτλος γραφήματος
            save_name (str, optional): Όνομα αρχείου αποθήκευσης
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = f'Σύγκριση Μοντέλων - {metric}'
        
        if save_name is None:
            save_name = f'model_comparison_{metric.lower().replace("@", "_at_")}'
        
        # Δημιουργία figure με μοντέρνο στυλ
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Προετοιμασία δεδομένων
        if metric in results_df.columns:
            data = results_df[metric].sort_values(ascending=False)
            models = data.index.tolist()
            values = data.values
            
            # Δημιουργία gradient χρωμάτων
            colors = self._create_gradient_colors(len(models))
            
            # Δημιουργία bars με gradient effect
            bars = ax.bar(models, values, color=colors, alpha=0.8, 
                         edgecolor='white', linewidth=1.5)
            
            # Προσθήκη τιμών πάνω από τα bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # Styling
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Μοντέλα Σύστασης', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=14, fontweight='bold')
            
            # Περιστροφή labels για καλύτερη ανάγνωση
            plt.xticks(rotation=45, ha='right')
            
            # Προσθήκη subtle background gradient
            ax.set_facecolor('#fafafa')
            
            # Βελτίωση grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Ρύθμιση y-axis για καλύτερη εμφάνιση
            y_max = max(values) * 1.15
            ax.set_ylim(0, y_max)
            
        else:
            ax.text(0.5, 0.5, f'Μετρική {metric} δεν βρέθηκε', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, color='red')
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def create_metrics_heatmap(self, results_df: pd.DataFrame,
                              title: Optional[str] = None,
                              save_name: Optional[str] = None) -> str:
        """
        Δημιουργία μοντέρνου heatmap για όλες τις μετρικές
        
        Args:
            results_df (pd.DataFrame): DataFrame με αποτελέσματα
            title (str, optional): Τίτλος γραφήματος
            save_name (str, optional): Όνομα αρχείου
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = 'Heatmap Μετρικών Αξιολόγησης'
        
        if save_name is None:
            save_name = 'metrics_heatmap'
        
        # Δημιουργία figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Φιλτράρισμα μόνο αριθμητικών στηλών
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        data = results_df[numeric_cols]
        
        # Δημιουργία custom colormap
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
        n_bins = 100
        cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
        
        # Δημιουργία heatmap
        sns.heatmap(data, annot=True, fmt='.4f', cmap=cmap,
                   cbar_kws={'label': 'Τιμή Μετρικής'},
                   linewidths=0.5, linecolor='white',
                   square=False, ax=ax)
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Μετρικές Αξιολόγησης', fontsize=14, fontweight='bold')
        ax.set_ylabel('Μοντέλα Σύστασης', fontsize=14, fontweight='bold')
        
        # Περιστροφή labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def create_performance_radar_chart(self, results_df: pd.DataFrame,
                                     models_to_compare: Optional[List[str]] = None,
                                     title: Optional[str] = None,
                                     save_name: Optional[str] = None) -> str:
        """
        Δημιουργία radar chart για σύγκριση μοντέλων
        
        Args:
            results_df (pd.DataFrame): DataFrame με αποτελέσματα
            models_to_compare (List[str], optional): Μοντέλα προς σύγκριση
            title (str, optional): Τίτλος γραφήματος
            save_name (str, optional): Όνομα αρχείου
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = 'Radar Chart Επιδόσεων Μοντέλων'
        
        if save_name is None:
            save_name = 'performance_radar'
        
        # Επιλογή μοντέλων
        if models_to_compare is None:
            models_to_compare = results_df.index[:5].tolist()  # Top 5 μοντέλα
        
        # Φιλτράρισμα δεδομένων
        data = results_df.loc[models_to_compare]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
        
        # Κανονικοποίηση δεδομένων (0-1)
        data_norm = (data - data.min()) / (data.max() - data.min())
        
        # Δημιουργία radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Υπολογισμός γωνιών
        angles = np.linspace(0, 2 * np.pi, len(numeric_cols), endpoint=False).tolist()
        angles += angles[:1]  # Κλείσιμο κύκλου
        
        # Σχεδίαση για κάθε μοντέλο
        for i, (model, row) in enumerate(data_norm.iterrows()):
            values = row.tolist()
            values += values[:1]  # Κλείσιμο κύκλου
            
            color = self.model_palette[i % len(self.model_palette)]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Προσθήκη labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(numeric_cols, fontsize=11)
        
        # Styling
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
        ax.grid(True, alpha=0.3)
        
        # Legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def create_training_curves(self, training_history: Dict[str, List[float]],
                             title: Optional[str] = None,
                             save_name: Optional[str] = None) -> str:
        """
        Δημιουργία γραφήματος καμπυλών εκπαίδευσης
        
        Args:
            training_history (Dict[str, List[float]]): Ιστορικό εκπαίδευσης
            title (str, optional): Τίτλος γραφήματος
            save_name (str, optional): Όνομα αρχείου
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = 'Καμπύλες Εκπαίδευσης'
        
        if save_name is None:
            save_name = 'training_curves'
        
        # Δημιουργία figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Σχεδίαση καμπυλών
        for i, (metric, values) in enumerate(training_history.items()):
            epochs = range(1, len(values) + 1)
            color = self.model_palette[i % len(self.model_palette)]
            
            ax.plot(epochs, values, marker='o', linewidth=2.5, 
                   markersize=6, label=metric, color=color, alpha=0.8)
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Τιμή Μετρικής', fontsize=14, fontweight='bold')
        
        # Grid και legend
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Background
        ax.set_facecolor('#fafafa')
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def create_dataset_overview(self, data_stats: Dict[str, Any],
                              title: Optional[str] = None,
                              save_name: Optional[str] = None) -> str:
        """
        Δημιουργία οπτικοποίησης επισκόπησης dataset
        
        Args:
            data_stats (Dict[str, Any]): Στατιστικά dataset
            title (str, optional): Τίτλος γραφήματος
            save_name (str, optional): Όνομα αρχείου
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = 'Επισκόπηση Dataset'
        
        if save_name is None:
            save_name = 'dataset_overview'
        
        # Δημιουργία subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Βασικά στατιστικά (pie chart)
        if 'basic_stats' in data_stats:
            stats = data_stats['basic_stats']
            labels = list(stats.keys())
            values = list(stats.values())
            colors = self.model_palette[:len(labels)]
            
            wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors,
                                              autopct='%1.0f', startangle=90,
                                              textprops={'fontsize': 10})
            ax1.set_title('Βασικά Στατιστικά', fontsize=14, fontweight='bold')
        
        # 2. Κατανομή αλληλεπιδράσεων (histogram)
        if 'interaction_distribution' in data_stats:
            dist_data = data_stats['interaction_distribution']
            ax2.hist(dist_data, bins=30, color=self.colors['primary'], 
                    alpha=0.7, edgecolor='white', linewidth=1)
            ax2.set_title('Κατανομή Αλληλεπιδράσεων', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Αριθμός Αλληλεπιδράσεων')
            ax2.set_ylabel('Συχνότητα')
        
        # 3. Top καλλιτέχνες (bar chart)
        if 'top_artists' in data_stats:
            top_artists = data_stats['top_artists']
            artists = list(top_artists.keys())[:10]
            counts = list(top_artists.values())[:10]
            
            bars = ax3.barh(artists, counts, color=self.colors['accent'], alpha=0.8)
            ax3.set_title('Top 10 Καλλιτέχνες', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Αριθμός Αλληλεπιδράσεων')
        
        # 4. Sparsity visualization
        if 'sparsity' in data_stats:
            sparsity = data_stats['sparsity']
            density = 1 - sparsity
            
            # Donut chart για sparsity
            sizes = [density, sparsity]
            labels = ['Density', 'Sparsity']
            colors = [self.colors['success'], self.colors['warning']]
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.2f%%', startangle=90,
                                              pctdistance=0.85)
            
            # Δημιουργία donut effect
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax4.add_artist(centre_circle)
            ax4.set_title('Matrix Sparsity', fontsize=14, fontweight='bold')
        
        # Ρύθμιση layout
        plt.tight_layout()
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def create_comprehensive_dashboard(self, results_df: pd.DataFrame,
                                     data_stats: Optional[Dict[str, Any]] = None,
                                     title: Optional[str] = None,
                                     save_name: Optional[str] = None) -> str:
        """
        Δημιουργία ολοκληρωμένου dashboard με όλες τις οπτικοποιήσεις
        
        Args:
            results_df (pd.DataFrame): Αποτελέσματα μοντέλων
            data_stats (Dict[str, Any], optional): Στατιστικά dataset
            title (str, optional): Τίτλος dashboard
            save_name (str, optional): Όνομα αρχείου
            
        Returns:
            str: Διαδρομή αποθηκευμένου αρχείου
        """
        if title is None:
            title = 'Ολοκληρωμένο Dashboard Αποτελεσμάτων'
        
        if save_name is None:
            save_name = 'comprehensive_dashboard'
        
        # Δημιουργία μεγάλου figure με subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
        
        # Layout: 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model comparison (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'Recall@5' in results_df.columns:
            data = results_df['Recall@5'].sort_values(ascending=False)
            colors = self._create_gradient_colors(len(data))
            bars = ax1.bar(data.index, data.values, color=colors, alpha=0.8)
            ax1.set_title('Σύγκριση Μοντέλων - Recall@5', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. NDCG comparison (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'NDCG@5' in results_df.columns:
            data = results_df['NDCG@5'].sort_values(ascending=False)
            ax2.barh(data.index, data.values, color=self.colors['secondary'], alpha=0.8)
            ax2.set_title('NDCG@5', fontsize=12, fontweight='bold')
        
        # 3. Heatmap (middle row, spans all columns)
        ax3 = fig.add_subplot(gs[1, :])
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sns.heatmap(results_df[numeric_cols], annot=True, fmt='.3f',
                       cmap='viridis', ax=ax3, cbar_kws={'shrink': 0.8})
            ax3.set_title('Heatmap Όλων των Μετρικών', fontsize=14, fontweight='bold')
        
        # 4. Coverage vs Accuracy scatter (bottom-left)
        ax4 = fig.add_subplot(gs[2, 0])
        if 'Coverage' in results_df.columns and 'Recall@5' in results_df.columns:
            scatter = ax4.scatter(results_df['Coverage'], results_df['Recall@5'],
                                c=range(len(results_df)), cmap='plasma',
                                s=100, alpha=0.7, edgecolors='white', linewidth=1)
            ax4.set_xlabel('Coverage')
            ax4.set_ylabel('Recall@5')
            ax4.set_title('Coverage vs Accuracy', fontsize=12, fontweight='bold')
        
        # 5. Hit Rate comparison (bottom-middle)
        ax5 = fig.add_subplot(gs[2, 1])
        if 'Hit_Rate@5' in results_df.columns:
            data = results_df['Hit_Rate@5'].sort_values(ascending=False)
            ax5.pie(data.values, labels=data.index, autopct='%1.1f%%',
                   colors=self.model_palette[:len(data)])
            ax5.set_title('Hit Rate@5 Distribution', fontsize=12, fontweight='bold')
        
        # 6. MRR comparison (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2])
        if 'MRR' in results_df.columns:
            data = results_df['MRR'].sort_values(ascending=False)
            ax6.plot(data.index, data.values, marker='o', linewidth=2.5,
                    markersize=8, color=self.colors['accent'])
            ax6.set_title('MRR Scores', fontsize=12, fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
        
        # Αποθήκευση
        filepath = self._save_plot(fig, save_name)
        plt.close(fig)
        
        return filepath
    
    def _create_gradient_colors(self, n_colors: int) -> List[str]:
        """
        Δημιουργία gradient χρωμάτων
        
        Args:
            n_colors (int): Αριθμός χρωμάτων
            
        Returns:
            List[str]: Λίστα hex χρωμάτων
        """
        if n_colors <= len(self.model_palette):
            return self.model_palette[:n_colors]
        
        # Δημιουργία gradient
        colors = []
        for i in range(n_colors):
            ratio = i / (n_colors - 1) if n_colors > 1 else 0
            # Interpolation μεταξύ δύο χρωμάτων
            start_color = np.array([0.18, 0.53, 0.67])  # Μπλε
            end_color = np.array([0.64, 0.23, 0.45])    # Μωβ
            
            color = start_color + ratio * (end_color - start_color)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    def _save_plot(self, fig, filename: str) -> str:
        """
        Αποθήκευση γραφήματος σε υψηλή ανάλυση
        
        Args:
            fig: Matplotlib figure
            filename (str): Όνομα αρχείου
            
        Returns:
            str: Πλήρης διαδρομή αρχείου
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"{filename}_{timestamp}.png")
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        self.logger.info(f"Γράφημα αποθηκεύτηκε: {filepath}")
        return filepath
    
    def generate_all_visualizations(self, results_df: pd.DataFrame,
                                  data_stats: Optional[Dict[str, Any]] = None,
                                  training_history: Optional[Dict[str, List[float]]] = None) -> Dict[str, str]:
        """
        Δημιουργία όλων των οπτικοποιήσεων
        
        Args:
            results_df (pd.DataFrame): Αποτελέσματα μοντέλων
            data_stats (Dict[str, Any], optional): Στατιστικά dataset
            training_history (Dict[str, List[float]], optional): Ιστορικό εκπαίδευσης
            
        Returns:
            Dict[str, str]: Λεξικό με διαδρομές αρχείων
        """
        self.logger.info("Δημιουργία όλων των οπτικοποιήσεων...")
        
        generated_files = {}
        
        try:
            # 1. Model comparison charts
            for metric in ['Recall@5', 'NDCG@5', 'Hit_Rate@5', 'MRR']:
                if metric in results_df.columns:
                    filepath = self.create_model_comparison_chart(results_df, metric)
                    generated_files[f'comparison_{metric}'] = filepath
            
            # 2. Metrics heatmap
            filepath = self.create_metrics_heatmap(results_df)
            generated_files['heatmap'] = filepath
            
            # 3. Radar chart
            filepath = self.create_performance_radar_chart(results_df)
            generated_files['radar'] = filepath
            
            # 4. Dataset overview
            if data_stats:
                filepath = self.create_dataset_overview(data_stats)
                generated_files['dataset_overview'] = filepath
            
            # 5. Training curves
            if training_history:
                filepath = self.create_training_curves(training_history)
                generated_files['training_curves'] = filepath
            
            # 6. Comprehensive dashboard
            filepath = self.create_comprehensive_dashboard(results_df, data_stats)
            generated_files['dashboard'] = filepath
            
            self.logger.info(f"Δημιουργήθηκαν {len(generated_files)} οπτικοποιήσεις")
            
        except Exception as e:
            self.logger.error("Σφάλμα κατά τη δημιουργία οπτικοποιήσεων", exception=e)
        
        return generated_files


def create_quick_visualization(results_df: pd.DataFrame, 
                             output_dir: str = "plots/") -> Dict[str, str]:
    """
    Γρήγορη δημιουργία βασικών οπτικοποιήσεων
    
    Args:
        results_df (pd.DataFrame): Αποτελέσματα μοντέλων
        output_dir (str): Φάκελος εξόδου
        
    Returns:
        Dict[str, str]: Διαδρομές δημιουργημένων αρχείων
    """
    viz_engine = ModernVisualizationEngine(output_dir)
    
    generated_files = {}
    
    # Δημιουργία βασικών γραφημάτων
    if 'Recall@5' in results_df.columns:
        filepath = viz_engine.create_model_comparison_chart(results_df, 'Recall@5')
        generated_files['recall_comparison'] = filepath
    
    # Heatmap
    filepath = viz_engine.create_metrics_heatmap(results_df)
    generated_files['metrics_heatmap'] = filepath
    
    return generated_files 