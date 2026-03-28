import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SuggestionAnalyzer:
    def __init__(self, file1_path, file2_path, output_dir='/tmp/'):
        """
        Initialize analyzer with two suggestion files
        file1: Better performing model (fewer suggestions)
        file2: Worse performing model (more suggestions)
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_dir = output_dir
        self.df1 = None
        self.df2 = None
        self.all_suggestions_df1 = []
        self.all_suggestions_df2 = []
        
    def parse_suggestions_file(self, filepath):
        """Parse suggestions from file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        data = json.loads(content)
        parsed_data = []
        
        for entry in data:
            parts = entry.split('<---->')
            if len(parts) >= 11:
                suggestions_text = parts[2].replace('Suggestions:', '')
                suggestions = [s.strip() for s in suggestions_text.split('- ') if s.strip()]
                
                parsed_entry = {
                    'question': parts[0].replace('Question:', ''),
                    'answer': parts[1].replace('Answer:', ''),
                    'suggestions': suggestions_text,
                    'suggestions_list': suggestions,
                    'num_suggestions': len(suggestions),
                    'bleu': float(parts[3].replace('BLEU:', '')),
                    'functionality': float(parts[4].replace('Functionality:', '')),
                    'relaxed_functionality': float(parts[5].replace('Relaxed Functionality:', '')),
                    'syntax': float(parts[6].replace('Syntax:', '')),
                    'last_bleu': float(parts[7].replace('Last BLEU:', '')),
                    'initial_bleu': float(parts[8].replace('Initial BLEU:', '')),
                    'initial_relaxed_functionality': float(parts[9].replace('Initial Relaxed Functionality:', '')),
                    'task_id': parts[10].replace('Task ID:', ''),
                    'reference_answer': parts[11].replace('Reference Answer:', '') if len(parts) > 11 else ''
                }
                parsed_data.append(parsed_entry)
        
        return pd.DataFrame(parsed_data)
    
    def load_data(self):
        """Load and parse both datasets"""
        self.df1 = self.parse_suggestions_file(self.file1_path)
        self.df2 = self.parse_suggestions_file(self.file2_path)
        
        # Extract all suggestions
        for _, row in self.df1.iterrows():
            self.all_suggestions_df1.extend(row['suggestions_list'])
        
        for _, row in self.df2.iterrows():
            self.all_suggestions_df2.extend(row['suggestions_list'])
        
        print(f"Loaded {len(self.df1)} entries from Model 1 (Better)")
        print(f"Loaded {len(self.df2)} entries from Model 2 (Worse)")
        print(f"Total suggestions: Model 1={len(self.all_suggestions_df1)}, Model 2={len(self.all_suggestions_df2)}")
    
    def calculate_specificity_score(self, suggestion):
        """Calculate how specific a suggestion is (higher = more specific/overfitted)"""
        specific_patterns = [
            (r'\b\d+\b', 2),  # Specific numbers
            (r'`[^`]+`', 3),  # Code snippets
            (r'\[[^\]]+\]', 2),  # Specific indices
            (r'==\s*\d+', 3),  # Exact comparisons
            (r'tb_\w+', 2),  # Specific signal names
            (r'assert property', 1),  # Specific constructs
            (r'\$\w+', 2),  # SystemVerilog specific
            (r'@\(posedge\s+\w+\)', 2),  # Clock edges
        ]
        
        specificity = 0
        word_count = len(suggestion.split())
        
        for pattern, weight in specific_patterns:
            matches = len(re.findall(pattern, suggestion))
            specificity += matches * weight
        
        return specificity / max(word_count, 1)
    
    def extract_suggestion_templates(self, suggestions):
        """Extract generalized templates from suggestions"""
        templates = []
        
        for suggestion in suggestions:
            # Replace specific values with placeholders
            template = suggestion
            template = re.sub(r'\btb_\w+', '<SIGNAL>', template)
            template = re.sub(r'\b\d+\b', '<NUM>', template)
            template = re.sub(r'`[^`]+`', '<CODE>', template)
            template = re.sub(r'\[[^\]]+\]', '<INDEX>', template)
            template = re.sub(r'==\s*[^\s]+', '== <VALUE>', template)
            template = re.sub(r'!=\s*[^\s]+', '!= <VALUE>', template)
            templates.append(template)
        
        return templates
    
    def analyze_task_level_patterns(self):
        """Analyze patterns at task level"""
        task_analysis = {}
        
        # Get common task IDs
        common_tasks = set(self.df1['task_id']) & set(self.df2['task_id'])
        
        for task_id in common_tasks:
            df1_task = self.df1[self.df1['task_id'] == task_id]
            df2_task = self.df2[self.df2['task_id'] == task_id]
            
            if not df1_task.empty and not df2_task.empty:
                sugg1 = df1_task.iloc[0]['suggestions_list']
                sugg2 = df2_task.iloc[0]['suggestions_list']
                
                # Calculate various metrics
                set1, set2 = set(sugg1), set(sugg2)
                overlap = len(set1 & set2) / max(len(set1 | set2), 1)
                
                # Template analysis
                templates1 = self.extract_suggestion_templates(sugg1)
                templates2 = self.extract_suggestion_templates(sugg2)
                template_overlap = len(set(templates1) & set(templates2)) / max(len(set(templates1) | set(templates2)), 1)
                
                task_analysis[task_id] = {
                    'num_sugg_model1': len(sugg1),
                    'num_sugg_model2': len(sugg2),
                    'exact_overlap': overlap,
                    'template_overlap': template_overlap,
                    'performance_diff': df1_task.iloc[0]['functionality'] - df2_task.iloc[0]['functionality'],
                    'bleu_diff': df1_task.iloc[0]['bleu'] - df2_task.iloc[0]['bleu']
                }
        
        return task_analysis
    
    def calculate_overfitting_metrics(self, df, suggestions):
        """Calculate comprehensive overfitting metrics"""
        metrics = {}
        
        # 1. Uniqueness ratio
        unique_suggestions = set(suggestions)
        metrics['uniqueness_ratio'] = len(unique_suggestions) / len(suggestions) if suggestions else 0
        
        # 2. Average specificity
        specificities = [self.calculate_specificity_score(s) for s in suggestions[:100]]
        metrics['avg_specificity'] = np.mean(specificities) if specificities else 0
        
        # 3. Template diversity
        templates = self.extract_suggestion_templates(suggestions)
        unique_templates = set(templates)
        metrics['template_diversity'] = len(unique_templates) / len(templates) if templates else 0
        
        # 4. Vocabulary richness
        all_words = ' '.join(suggestions).lower().split()
        unique_words = set(all_words)
        metrics['vocab_richness'] = len(unique_words) / len(all_words) if all_words else 0
        
        # 5. Length variance (low variance may indicate templating)
        lengths = [len(s.split()) for s in suggestions]
        metrics['length_std'] = np.std(lengths) if lengths else 0
        metrics['length_mean'] = np.mean(lengths) if lengths else 0
        metrics['length_cv'] = metrics['length_std'] / metrics['length_mean'] if metrics['length_mean'] > 0 else 0
        
        # 6. Repetition patterns
        bigram_vec = CountVectorizer(ngram_range=(2, 2), max_features=100)
        if len(suggestions) > 1:
            bigrams = bigram_vec.fit_transform(suggestions)
            bigram_counts = bigrams.sum(axis=0).A1
            metrics['bigram_repetition'] = np.mean(bigram_counts) / len(suggestions)
        else:
            metrics['bigram_repetition'] = 0
        
        # 7. Cross-task reusability
        task_suggestion_map = defaultdict(list)
        for _, row in df.iterrows():
            task_suggestion_map[row['task_id']].extend(row['suggestions_list'])
        
        suggestion_task_count = Counter()
        for task_suggestions in task_suggestion_map.values():
            for sugg in set(task_suggestions):
                suggestion_task_count[sugg] += 1
        
        reused_suggestions = sum(1 for count in suggestion_task_count.values() if count > 1)
        metrics['cross_task_reuse'] = reused_suggestions / len(suggestion_task_count) if suggestion_task_count else 0
        
        # Calculate overall overfitting score (lower = more overfitted)
        metrics['overfitting_score'] = (
            metrics['uniqueness_ratio'] * 0.25 +
            (1 - metrics['avg_specificity']) * 0.20 +
            metrics['template_diversity'] * 0.20 +
            metrics['vocab_richness'] * 0.15 +
            metrics['length_cv'] * 0.10 +
            metrics['cross_task_reuse'] * 0.10
        )
        
        return metrics
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization with multiple subplots"""
        fig = plt.figure(figsize=(24, 20))
        
        # Calculate metrics
        metrics1 = self.calculate_overfitting_metrics(self.df1, self.all_suggestions_df1)
        metrics2 = self.calculate_overfitting_metrics(self.df2, self.all_suggestions_df2)
        task_patterns = self.analyze_task_level_patterns()
        
        # 1. Overfitting Score Comparison
        ax1 = plt.subplot(4, 4, 1)
        models = ['Model 1\n(Better Test)', 'Model 2\n(Worse Test)']
        scores = [metrics1['overfitting_score'], metrics2['overfitting_score']]
        colors = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in scores]
        bars = ax1.bar(models, scores, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Overfitting Score', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Overfitting Score\n(Higher = Less Overfitted)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2., score + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Detailed Metrics Radar Chart
        ax2 = plt.subplot(4, 4, 2, projection='polar')
        metrics_names = ['Uniqueness', 'Template\nDiversity', 'Vocab\nRichness', 
                        'Length\nVariance', 'Cross-task\nReuse']
        metrics_values1 = [metrics1['uniqueness_ratio'], metrics1['template_diversity'],
                          metrics1['vocab_richness'], metrics1['length_cv'], 
                          metrics1['cross_task_reuse']]
        metrics_values2 = [metrics2['uniqueness_ratio'], metrics2['template_diversity'],
                          metrics2['vocab_richness'], metrics2['length_cv'], 
                          metrics2['cross_task_reuse']]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        metrics_values1 = np.concatenate((metrics_values1, [metrics_values1[0]]))
        metrics_values2 = np.concatenate((metrics_values2, [metrics_values2[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2.plot(angles, metrics_values1, 'o-', linewidth=2, label='Model 1 (Better)', color='#2ecc71')
        ax2.fill(angles, metrics_values1, alpha=0.25, color='#2ecc71')
        ax2.plot(angles, metrics_values2, 'o-', linewidth=2, label='Model 2 (Worse)', color='#e74c3c')
        ax2.fill(angles, metrics_values2, alpha=0.25, color='#e74c3c')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_names)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.set_title('Overfitting Indicators', fontsize=14, fontweight='bold', pad=20)
        
        # 3. Task-level Performance Difference
        ax3 = plt.subplot(4, 4, 3)
        if task_patterns:
            task_data = pd.DataFrame(task_patterns).T
            ax3.scatter(task_data['num_sugg_model2'] - task_data['num_sugg_model1'],
                       task_data['performance_diff'], 
                       s=100, alpha=0.6, c=task_data['template_overlap'],
                       cmap='RdYlGn', edgecolor='black', linewidth=1)
            ax3.set_xlabel('Suggestion Count Diff (Model2 - Model1)', fontsize=11)
            ax3.set_ylabel('Performance Diff (Model1 - Model2)', fontsize=11)
            ax3.set_title('Task-level: More Suggestions vs Performance', fontsize=14, fontweight='bold')
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('Template Overlap', rotation=270, labelpad=15)
        
        # 4. Specificity Distribution
        ax4 = plt.subplot(4, 4, 4)
        spec1 = [self.calculate_specificity_score(s) for s in self.all_suggestions_df1[:200]]
        spec2 = [self.calculate_specificity_score(s) for s in self.all_suggestions_df2[:200]]
        
        ax4.violinplot([spec1], positions=[0], widths=0.7, showmeans=True, showmedians=True)
        vp1 = ax4.violinplot([spec1], positions=[0], widths=0.7, showmeans=True, showmedians=True)
        vp2 = ax4.violinplot([spec2], positions=[1], widths=0.7, showmeans=True, showmedians=True)
        
        for pc in vp1['bodies']:
            pc.set_facecolor('#2ecc71')
            pc.set_alpha(0.7)
        for pc in vp2['bodies']:
            pc.set_facecolor('#e74c3c')
            pc.set_alpha(0.7)
            
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Model 1\n(Better)', 'Model 2\n(Worse)'])
        ax4.set_ylabel('Specificity Score', fontsize=11)
        ax4.set_title('Suggestion Specificity Distribution\n(Lower = More Generic)', 
                     fontsize=14, fontweight='bold')
        
        # 5. Template Reuse Patterns
        ax5 = plt.subplot(4, 4, 5)
        templates1 = self.extract_suggestion_templates(self.all_suggestions_df1)
        templates2 = self.extract_suggestion_templates(self.all_suggestions_df2)
        template_counts1 = Counter(templates1)
        template_counts2 = Counter(templates2)
        
        # Get reuse distribution
        reuse_dist1 = Counter(template_counts1.values())
        reuse_dist2 = Counter(template_counts2.values())
        
        max_reuse = max(max(reuse_dist1.keys(), default=0), max(reuse_dist2.keys(), default=0))
        x_range = range(1, min(max_reuse + 1, 11))
        
        counts1 = [reuse_dist1.get(i, 0) for i in x_range]
        counts2 = [reuse_dist2.get(i, 0) for i in x_range]
        
        x = np.arange(len(x_range))
        width = 0.35
        ax5.bar(x - width/2, counts1, width, label='Model 1', color='#2ecc71', alpha=0.8)
        ax5.bar(x + width/2, counts2, width, label='Model 2', color='#e74c3c', alpha=0.8)
        ax5.set_xlabel('Template Reuse Count', fontsize=11)
        ax5.set_ylabel('Number of Templates', fontsize=11)
        ax5.set_title('Template Reuse Distribution', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(x_range)
        ax5.legend()
        
        # 6. Suggestion Length Distribution
        ax6 = plt.subplot(4, 4, 6)
        lengths1 = [len(s.split()) for s in self.all_suggestions_df1]
        lengths2 = [len(s.split()) for s in self.all_suggestions_df2]
        
        ax6.hist(lengths1, bins=30, alpha=0.6, label=f'Model 1 (μ={np.mean(lengths1):.1f}, σ={np.std(lengths1):.1f})', 
                color='#2ecc71', density=True, edgecolor='black')
        ax6.hist(lengths2, bins=30, alpha=0.6, label=f'Model 2 (μ={np.mean(lengths2):.1f}, σ={np.std(lengths2):.1f})', 
                color='#e74c3c', density=True, edgecolor='black')
        ax6.set_xlabel('Suggestion Length (words)', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Length Distribution Comparison', fontsize=14, fontweight='bold')
        ax6.legend()
        
        # 7. Semantic Similarity Heatmap
        ax7 = plt.subplot(4, 4, 7)
        # Sample suggestions for similarity analysis
        sample_size = min(50, len(self.all_suggestions_df1), len(self.all_suggestions_df2))
        sample1 = self.all_suggestions_df1[:sample_size]
        sample2 = self.all_suggestions_df2[:sample_size]
        
        vectorizer = TfidfVectorizer(max_features=100)
        combined = sample1 + sample2
        tfidf = vectorizer.fit_transform(combined)
        similarity_matrix = cosine_similarity(tfidf)
        
        # Create block matrix visualization
        im = ax7.imshow(similarity_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax7.set_xlabel('Suggestion Index', fontsize=11)
        ax7.set_ylabel('Suggestion Index', fontsize=11)
        ax7.set_title('Suggestion Similarity Matrix', fontsize=14, fontweight='bold')
        ax7.axhline(y=sample_size, color='blue', linewidth=2)
        ax7.axvline(x=sample_size, color='blue', linewidth=2)
        ax7.text(sample_size/2, -5, 'Model 1', ha='center', fontweight='bold')
        ax7.text(sample_size + sample_size/2, -5, 'Model 2', ha='center', fontweight='bold')
        plt.colorbar(im, ax=ax7)
        
        # 8. Category Distribution
        ax8 = plt.subplot(4, 4, 8)
        categories = {
            "Temporal": ["delay", "|->", "|=>", "##", "eventually", "strong"],
            "Logical": ["&&", "||", "!", "and", "or", "not"],
            "Comparison": ["===", "!==", "==", "!=", "<", ">"],
            "Structural": ["antecedent", "consequent", "implication"],
            "Signal": ["signal", "grant", "request", "fifo", "push", "pop"]
        }
        
        cat_counts1 = defaultdict(int)
        cat_counts2 = defaultdict(int)
        
        for sugg in self.all_suggestions_df1:
            sugg_lower = sugg.lower()
            for cat, keywords in categories.items():
                if any(kw in sugg_lower for kw in keywords):
                    cat_counts1[cat] += 1
                    
        for sugg in self.all_suggestions_df2:
            sugg_lower = sugg.lower()
            for cat, keywords in categories.items():
                if any(kw in sugg_lower for kw in keywords):
                    cat_counts2[cat] += 1
        
        cats = list(categories.keys())
        x = np.arange(len(cats))
        width = 0.35
        
        counts1 = [cat_counts1[cat] for cat in cats]
        counts2 = [cat_counts2[cat] for cat in cats]
        
        ax8.bar(x - width/2, counts1, width, label='Model 1', color='#2ecc71', alpha=0.8)
        ax8.bar(x + width/2, counts2, width, label='Model 2', color='#e74c3c', alpha=0.8)
        ax8.set_xlabel('Category', fontsize=11)
        ax8.set_ylabel('Count', fontsize=11)
        ax8.set_title('Suggestion Categories', fontsize=14, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(cats, rotation=45, ha='right')
        ax8.legend()
        
        # 9. t-SNE Visualization
        ax9 = plt.subplot(4, 4, 9)
        if len(combined) > 10:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
            tsne_results = tsne.fit_transform(tfidf.toarray())
            
            ax9.scatter(tsne_results[:sample_size, 0], tsne_results[:sample_size, 1],
                       c='#2ecc71', label='Model 1', alpha=0.7, s=100, edgecolor='black')
            ax9.scatter(tsne_results[sample_size:, 0], tsne_results[sample_size:, 1],
                       c='#e74c3c', label='Model 2', alpha=0.7, s=100, edgecolor='black')
            ax9.set_xlabel('t-SNE Component 1', fontsize=11)
            ax9.set_ylabel('t-SNE Component 2', fontsize=11)
            ax9.set_title('Suggestion Embeddings (t-SNE)', fontsize=14, fontweight='bold')
            ax9.legend()
        
        # 10. Performance vs Suggestion Count
        ax10 = plt.subplot(4, 4, 10)
        ax10.scatter(self.df1['num_suggestions'], self.df1['functionality'],
                    c='#2ecc71', label='Model 1', alpha=0.7, s=100, edgecolor='black')
        ax10.scatter(self.df2['num_suggestions'], self.df2['functionality'],
                    c='#e74c3c', label='Model 2', alpha=0.7, s=100, edgecolor='black')
        ax10.set_xlabel('Number of Suggestions', fontsize=11)
        ax10.set_ylabel('Functionality Score', fontsize=11)
        ax10.set_title('Performance vs Suggestion Count', fontsize=14, fontweight='bold')
        ax10.legend()
        
        # 11. Improvement Distribution
        ax11 = plt.subplot(4, 4, 11)
        improvement1 = self.df1['functionality'] - self.df1['initial_relaxed_functionality']
        improvement2 = self.df2['functionality'] - self.df2['initial_relaxed_functionality']
        
        ax11.hist(improvement1, bins=20, alpha=0.6, label='Model 1', color='#2ecc71', 
                 density=True, edgecolor='black')
        ax11.hist(improvement2, bins=20, alpha=0.6, label='Model 2', color='#e74c3c', 
                 density=True, edgecolor='black')
        ax11.set_xlabel('Functionality Improvement', fontsize=11)
        ax11.set_ylabel('Density', fontsize=11)
        ax11.set_title('Improvement Distribution', fontsize=14, fontweight='bold')
        ax11.legend()
        
        # 12. Top Repeated Suggestions
        ax12 = plt.subplot(4, 4, 12)
        suggestion_counts1 = Counter(self.all_suggestions_df1)
        suggestion_counts2 = Counter(self.all_suggestions_df2)
        
        top_repeated1 = suggestion_counts1.most_common(5)
        top_repeated2 = suggestion_counts2.most_common(5)
        
        # Truncate long suggestions for display
        def truncate(s, n=30):
            return s[:n] + '...' if len(s) > n else s
        
        if top_repeated1:
            labels1 = [truncate(s[0]) for s in top_repeated1]
            counts1 = [s[1] for s in top_repeated1]
            y1 = np.arange(len(labels1))
            ax12.barh(y1, counts1, height=0.35, label='Model 1', 
                     color='#2ecc71', alpha=0.8)
        
        if top_repeated2:
            labels2 = [truncate(s[0]) for s in top_repeated2]
            counts2 = [s[1] for s in top_repeated2]
            y2 = np.arange(len(labels2)) + 0.4
            ax12.barh(y2, counts2, height=0.35, label='Model 2', 
                     color='#e74c3c', alpha=0.8)
            
            all_labels = []
            for i in range(max(len(labels1), len(labels2))):
                if i < len(labels1):
                    all_labels.append(labels1[i])
                if i < len(labels2):
                    all_labels.append(labels2[i])
            
            ax12.set_yticks(np.arange(max(len(labels1), len(labels2))) * 0.75 + 0.2)
            ax12.set_yticklabels(all_labels[:max(len(labels1), len(labels2))], fontsize=8)
        
        ax12.set_xlabel('Repetition Count', fontsize=11)
        ax12.set_title('Most Repeated Suggestions', fontsize=14, fontweight='bold')
        ax12.legend()
        
        # 13-16: Statistical Tests and Summary
        ax13 = plt.subplot(4, 4, 13)
        ax13.axis('off')
        
        # Perform statistical tests
        stats_results = []
        
        # T-test for suggestion counts
        t_stat, p_val = stats.ttest_ind(self.df1['num_suggestions'], self.df2['num_suggestions'])
        stats_results.append(f"Suggestion Count: t={t_stat:.3f}, p={p_val:.3f}")
        
        # T-test for functionality
        t_stat, p_val = stats.ttest_ind(self.df1['functionality'], self.df2['functionality'])
        stats_results.append(f"Functionality: t={t_stat:.3f}, p={p_val:.3f}")
        
        # T-test for BLEU
        t_stat, p_val = stats.ttest_ind(self.df1['bleu'], self.df2['bleu'])
        stats_results.append(f"BLEU Score: t={t_stat:.3f}, p={p_val:.3f}")
        
        # Display statistical tests
        ax13.text(0.1, 0.9, "Statistical Tests (t-test)", fontsize=14, fontweight='bold',
                 transform=ax13.transAxes)
        for i, result in enumerate(stats_results):
            ax13.text(0.1, 0.7 - i*0.15, result, fontsize=11, transform=ax13.transAxes)
        
        # Summary metrics table
        ax14 = plt.subplot(4, 4, 14)
        ax14.axis('off')
        
        summary_data = [
            ['Metric', 'Model 1', 'Model 2', 'Better?'],
            ['Total Suggestions', len(self.all_suggestions_df1), len(self.all_suggestions_df2), 
             '✓' if len(self.all_suggestions_df1) < len(self.all_suggestions_df2) else '✗'],
            ['Unique Ratio', f"{metrics1['uniqueness_ratio']:.3f}", f"{metrics2['uniqueness_ratio']:.3f}",
             '✓' if metrics1['uniqueness_ratio'] > metrics2['uniqueness_ratio'] else '✗'],
            ['Avg Specificity', f"{metrics1['avg_specificity']:.3f}", f"{metrics2['avg_specificity']:.3f}",
             '✓' if metrics1['avg_specificity'] < metrics2['avg_specificity'] else '✗'],
            ['Template Diversity', f"{metrics1['template_diversity']:.3f}", f"{metrics2['template_diversity']:.3f}",
             '✓' if metrics1['template_diversity'] > metrics2['template_diversity'] else '✗'],
            ['Overfitting Score', f"{metrics1['overfitting_score']:.3f}", f"{metrics2['overfitting_score']:.3f}",
             '✓' if metrics1['overfitting_score'] > metrics2['overfitting_score'] else '✗']
        ]
        
        # Create table
        cell_text = summary_data[1:]
        col_labels = summary_data[0]
        
        table = ax14.table(cellText=cell_text, colLabels=col_labels,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the "Better?" column
        for i in range(1, len(cell_text) + 1):
            if cell_text[i-1][3] == '✓':
                table[(i, 3)].set_facecolor('#90EE90')
            else:
                table[(i, 3)].set_facecolor('#FFB6C1')
        
        ax14.set_title('Summary Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # 15. Box plot comparison of key metrics
        ax15 = plt.subplot(4, 4, 15)
        
        # Prepare data for box plot
        box_data = [
            [self.df1['functionality'].values, self.df2['functionality'].values],
            [self.df1['bleu'].values, self.df2['bleu'].values],
            [(self.df1['functionality'] - self.df1['initial_relaxed_functionality']).values,
             (self.df2['functionality'] - self.df2['initial_relaxed_functionality']).values]
        ]
        
        positions = []
        for i in range(3):
            positions.extend([i*3, i*3+1])
        
        bp = ax15.boxplot([item for sublist in box_data for item in sublist],
                          positions=positions, widths=0.6,
                          patch_artist=True)
        
        # Color the boxes
        colors = ['#2ecc71', '#e74c3c'] * 3
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax15.set_xticks([0.5, 3.5, 6.5])
        ax15.set_xticklabels(['Functionality', 'BLEU', 'Improvement'], rotation=0)
        ax15.set_ylabel('Score', fontsize=11)
        ax15.set_title('Performance Metrics Distribution', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', alpha=0.7, label='Model 1'),
                          Patch(facecolor='#e74c3c', alpha=0.7, label='Model 2')]
        ax15.legend(handles=legend_elements)
        
        # 16. Final conclusion
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('off')
        
        # Determine which model is less overfitted
        conclusion = "ANALYSIS CONCLUSION\n" + "="*30 + "\n\n"
        
        if metrics1['overfitting_score'] > metrics2['overfitting_score']:
            conclusion += "Model 1 (Better Test Performance) shows:\n"
            conclusion += "• LESS overfitting\n"
            conclusion += "• Higher suggestion diversity\n"
            conclusion += "• Better generalization\n\n"
            conclusion += "Model 2 (Worse Test Performance) shows:\n"
            conclusion += "• MORE overfitting\n"
            conclusion += "• More repetitive suggestions\n"
            conclusion += "• Template memorization\n"
        else:
            conclusion += "Model 2 (Worse Test Performance) shows:\n"
            conclusion += "• LESS overfitting (unexpected!)\n"
            conclusion += "• Higher suggestion diversity\n\n"
            conclusion += "Model 1 (Better Test Performance) shows:\n"
            conclusion += "• MORE overfitting (but better test scores)\n"
            conclusion += "• May indicate better pattern learning\n"
        
        ax16.text(0.1, 0.5, conclusion, fontsize=11, transform=ax16.transAxes,
                 verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle('Comprehensive Suggestion Overfitting Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = f"{self.output_dir}/suggestion_overfitting_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive analysis to: {output_path}")
        
        return fig, metrics1, metrics2
    
    def create_focused_comparison(self):
        """Create a focused comparison visualization"""
        fig = plt.figure(figsize=(16, 10))
        
        metrics1 = self.calculate_overfitting_metrics(self.df1, self.all_suggestions_df1)
        metrics2 = self.calculate_overfitting_metrics(self.df2, self.all_suggestions_df2)
        
        # 1. Main overfitting indicators
        ax1 = plt.subplot(2, 3, 1)
        
        indicators = ['Uniqueness\nRatio', 'Template\nDiversity', 'Vocab\nRichness', 
                     'Cross-task\nReuse', 'Overall\nScore']
        values1 = [metrics1['uniqueness_ratio'], metrics1['template_diversity'],
                  metrics1['vocab_richness'], metrics1['cross_task_reuse'],
                  metrics1['overfitting_score']]
        values2 = [metrics2['uniqueness_ratio'], metrics2['template_diversity'],
                  metrics2['vocab_richness'], metrics2['cross_task_reuse'],
                  metrics2['overfitting_score']]
        
        x = np.arange(len(indicators))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, values1, width, label='Model 1 (Better Test)', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, values2, width, label='Model 2 (Worse Test)', 
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_ylabel('Score (Higher = Less Overfitted)', fontsize=12, fontweight='bold')
        ax1.set_title('Overfitting Indicators Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(indicators)
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Suggestion pattern analysis
        ax2 = plt.subplot(2, 3, 2)
        
        # Analyze suggestion patterns
        pattern_categories = {
            'Replace operator': ['Replace', 'operator', '|->'],
            'Add/Remove condition': ['Add', 'Remove', 'condition'],
            'Use function': ['Use', 'function', '$onehot'],
            'Change timing': ['timing', 'delay', '##['],
            'Modify comparison': ['===', '!==', '!=', '==']
        }
        
        cat_counts1 = {}
        cat_counts2 = {}
        
        for cat, keywords in pattern_categories.items():
            cat_counts1[cat] = sum(1 for s in self.all_suggestions_df1 
                                  if any(kw in s for kw in keywords))
            cat_counts2[cat] = sum(1 for s in self.all_suggestions_df2 
                                  if any(kw in s for kw in keywords))
        
        # Normalize by total suggestions
        total1 = len(self.all_suggestions_df1)
        total2 = len(self.all_suggestions_df2)
        
        cats = list(pattern_categories.keys())
        norm_counts1 = [cat_counts1[cat]/total1*100 if total1 > 0 else 0 for cat in cats]
        norm_counts2 = [cat_counts2[cat]/total2*100 if total2 > 0 else 0 for cat in cats]
        
        x = np.arange(len(cats))
        ax2.bar(x - width/2, norm_counts1, width, label='Model 1', color='#2ecc71', alpha=0.8)
        ax2.bar(x + width/2, norm_counts2, width, label='Model 2', color='#e74c3c', alpha=0.8)
        
        ax2.set_ylabel('Percentage of Suggestions (%)', fontsize=11)
        ax2.set_title('Suggestion Pattern Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cats, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance impact
        ax3 = plt.subplot(2, 3, 3)
        
        # Calculate efficiency: improvement per suggestion
        self.df1['efficiency'] = (self.df1['functionality'] - self.df1['initial_relaxed_functionality']) / np.maximum(self.df1['num_suggestions'], 1)
        self.df2['efficiency'] = (self.df2['functionality'] - self.df2['initial_relaxed_functionality']) / np.maximum(self.df2['num_suggestions'], 1)
        
        # Create violin plot for efficiency
        parts = ax3.violinplot([self.df1['efficiency'].dropna(), self.df2['efficiency'].dropna()],
                               positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
        
        colors = ['#2ecc71', '#e74c3c']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Model 1\n(Better)', 'Model 2\n(Worse)'])
        ax3.set_ylabel('Improvement per Suggestion', fontsize=11)
        ax3.set_title('Suggestion Efficiency', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Repetition analysis
        ax4 = plt.subplot(2, 3, 4)
        
        # Count unique vs repeated
        counter1 = Counter(self.all_suggestions_df1)
        counter2 = Counter(self.all_suggestions_df2)
        
        unique1 = sum(1 for count in counter1.values() if count == 1)
        repeated1 = sum(1 for count in counter1.values() if count > 1)
        unique2 = sum(1 for count in counter2.values() if count == 1)
        repeated2 = sum(1 for count in counter2.values() if count > 1)
        
        categories = ['Unique\nSuggestions', 'Repeated\nPatterns']
        model1_counts = [unique1, repeated1]
        model2_counts = [unique2, repeated2]
        
        x = np.arange(len(categories))
        ax4.bar(x - width/2, model1_counts, width, label='Model 1', color='#2ecc71', alpha=0.8)
        ax4.bar(x + width/2, model2_counts, width, label='Model 2', color='#e74c3c', alpha=0.8)
        
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Unique vs Repeated Suggestions', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Similarity distribution
        ax5 = plt.subplot(2, 3, 5)
        
        # Calculate pairwise similarities within each model
        if len(self.all_suggestions_df1) > 10 and len(self.all_suggestions_df2) > 10:
            sample1 = np.random.choice(self.all_suggestions_df1, min(100, len(self.all_suggestions_df1)), replace=False)
            sample2 = np.random.choice(self.all_suggestions_df2, min(100, len(self.all_suggestions_df2)), replace=False)
            
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf1 = vectorizer.fit_transform(sample1)
            tfidf2 = vectorizer.transform(sample2)
            
            sim1 = cosine_similarity(tfidf1)
            sim2 = cosine_similarity(tfidf2)
            
            # Get upper triangle (excluding diagonal)
            upper_tri_indices = np.triu_indices_from(sim1, k=1)
            sim1_values = sim1[upper_tri_indices]
            
            upper_tri_indices2 = np.triu_indices_from(sim2, k=1)
            sim2_values = sim2[upper_tri_indices2]
            
            ax5.hist(sim1_values, bins=30, alpha=0.6, label=f'Model 1 (μ={np.mean(sim1_values):.3f})',
                    color='#2ecc71', density=True, edgecolor='black')
            ax5.hist(sim2_values, bins=30, alpha=0.6, label=f'Model 2 (μ={np.mean(sim2_values):.3f})',
                    color='#e74c3c', density=True, edgecolor='black')
            
            ax5.set_xlabel('Pairwise Similarity', fontsize=11)
            ax5.set_ylabel('Density', fontsize=11)
            ax5.set_title('Within-Model Suggestion Similarity', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Key findings summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        findings = "KEY FINDINGS\n" + "="*25 + "\n\n"
        
        if metrics1['overfitting_score'] > metrics2['overfitting_score']:
            findings += "✓ Model 1 shows LESS overfitting:\n"
            findings += f"  • {(metrics1['uniqueness_ratio']-metrics2['uniqueness_ratio'])*100:.1f}% higher uniqueness\n"
            findings += f"  • {(metrics1['template_diversity']-metrics2['template_diversity'])*100:.1f}% more diverse templates\n"
            findings += f"  • {len(self.all_suggestions_df1)} total suggestions\n\n"
            
            findings += "✗ Model 2 shows MORE overfitting:\n"
            findings += f"  • Higher repetition rate\n"
            findings += f"  • Lower template diversity\n"
            findings += f"  • {len(self.all_suggestions_df2)} total suggestions\n\n"
            
            findings += "CONCLUSION:\nFewer, diverse suggestions\n→ Better generalization"
        else:
            findings += "Unexpected result:\n"
            findings += "Model with better test performance\n"
            findings += "shows more overfitting indicators.\n\n"
            findings += "Possible explanations:\n"
            findings += "• Task-specific optimization\n"
            findings += "• Effective memorization\n"
            findings += "• Different complexity patterns"
        
        ax6.text(0.05, 0.5, findings, fontsize=11, transform=ax6.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle('Focused Overfitting Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = f"{self.output_dir}/focused_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved focused comparison to: {output_path}")
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        metrics1 = self.calculate_overfitting_metrics(self.df1, self.all_suggestions_df1)
        metrics2 = self.calculate_overfitting_metrics(self.df2, self.all_suggestions_df2)
        task_patterns = self.analyze_task_level_patterns()
        
        report = {
            'summary': {
                'model1_overfitting_score': metrics1['overfitting_score'],
                'model2_overfitting_score': metrics2['overfitting_score'],
                'less_overfitted': 'Model 1' if metrics1['overfitting_score'] > metrics2['overfitting_score'] else 'Model 2',
                'total_suggestions_model1': len(self.all_suggestions_df1),
                'total_suggestions_model2': len(self.all_suggestions_df2)
            },
            'detailed_metrics': {
                'model1': metrics1,
                'model2': metrics2
            },
            'task_level_analysis': task_patterns,
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Statistical tests
        if len(self.df1) > 0 and len(self.df2) > 0:
            t_stat, p_val = stats.ttest_ind(self.df1['num_suggestions'], self.df2['num_suggestions'])
            report['statistical_tests']['suggestion_count'] = {'t_statistic': t_stat, 'p_value': p_val}
            
            t_stat, p_val = stats.ttest_ind(self.df1['functionality'], self.df2['functionality'])
            report['statistical_tests']['functionality'] = {'t_statistic': t_stat, 'p_value': p_val}
        
        # Generate recommendations
        if metrics1['overfitting_score'] > metrics2['overfitting_score']:
            report['recommendations'].append("Model 1 shows better generalization. Continue with this approach.")
            report['recommendations'].append("Focus on maintaining suggestion diversity.")
            report['recommendations'].append("Avoid increasing suggestion count unnecessarily.")
        else:
            report['recommendations'].append("Model 2 shows unexpected patterns. Investigate further.")
            report['recommendations'].append("Consider task-specific analysis.")
            report['recommendations'].append("Review training data distribution.")
        
        # Save report
        report_path = f"{self.output_dir}/overfitting_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to: {report_path}")
        
        return report
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Suggestion Overfitting Analysis...")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Generate visualizations
        print("\nGenerating comprehensive visualization...")
        fig1, metrics1, metrics2 = self.create_comprehensive_visualization()
        
        print("\nGenerating focused comparison...")
        fig2 = self.create_focused_comparison()
        
        # Generate report
        print("\nGenerating analysis report...")
        report = self.generate_report()
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Model 1 (Better Test Performance):")
        print(f"  - Overfitting Score: {metrics1['overfitting_score']:.3f}")
        print(f"  - Total Suggestions: {len(self.all_suggestions_df1)}")
        print(f"  - Unique Ratio: {metrics1['uniqueness_ratio']:.3f}")
        
        print(f"\nModel 2 (Worse Test Performance):")
        print(f"  - Overfitting Score: {metrics2['overfitting_score']:.3f}")
        print(f"  - Total Suggestions: {len(self.all_suggestions_df2)}")
        print(f"  - Unique Ratio: {metrics2['uniqueness_ratio']:.3f}")
        
        print(f"\nConclusion: Model {'1' if metrics1['overfitting_score'] > metrics2['overfitting_score'] else '2'} shows LESS overfitting")
        print("="*50)
        
        return report

# Usage example
if __name__ == "__main__":
    # Update these paths to your actual file locations
    file1 = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-20T00-01-20.119404_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions_high_scores.txt'
    file2 = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions_high_scores.txt'
    output_dir = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/suggestions_pkl/output'
    
    analyzer = SuggestionAnalyzer(file1, file2, output_dir)
    report = analyzer.run_full_analysis()