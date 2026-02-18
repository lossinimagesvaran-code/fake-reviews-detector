import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
import string
from time import time

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC  # Use LinearSVC instead of SVC for speed
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords', quiet=True)

class FakeReviewTrainer:
    """Train and evaluate fake review detection models"""
    
    def __init__(self, data_path='data/Preprocessed Fake Reviews Detection Dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        self.tfidf_transformer = None
        
    def load_data(self):
        """Load the preprocessed dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Drop unnamed column if present
            if 'Unnamed: 0' in self.df.columns:
                self.df.drop('Unnamed: 0', axis=1, inplace=True)
                print("  Dropped 'Unnamed: 0' column")
            
            # Drop any null values
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            if len(self.df) < initial_rows:
                print(f"  Dropped {initial_rows - len(self.df)} rows with null values")
            
            return True
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.data_path}")
            print("  Please ensure the dataset is in the 'data' folder")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*70)
        print("Exploratory Data Analysis")
        print("="*70)
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nLabel distribution:")
        label_counts = self.df['label'].value_counts()
        print(label_counts)
        print(f"\nLabel meanings:")
        print("  CG --> Computer Generated (Fake)")
        print("  OR --> Original Review (Human-written, Authentic)")
        
        print("\nClass percentages:")
        print(label_counts / len(self.df) * 100)
        
        # Add length column for analysis
        self.df['length'] = self.df['text_'].apply(len)
        
        print("\nReview length statistics by label:")
        print(self.df.groupby('label')['length'].describe())
        
        return True
    
    def visualize_data(self):
        """Create visualizations of the data"""
        print("\n" + "="*70)
        print("Creating Visualizations")
        print("="*70)
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Label distribution pie chart
        plt.figure(figsize=(8, 6))
        label_counts = self.df['label'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Distribution of Fake vs Real Reviews', fontsize=14, fontweight='bold')
        plt.savefig('plots/label_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Rating distribution by label
        plt.figure(figsize=(10, 6))
        rating_by_label = pd.crosstab(self.df['rating'], self.df['label'])
        rating_by_label.plot(kind='bar', stacked=True, color=colors)
        plt.title('Rating Distribution by Label', fontsize=14, fontweight='bold')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend(title='Label')
        plt.tight_layout()
        plt.savefig('plots/rating_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. Review length distribution
        plt.figure(figsize=(12, 6))
        for label in ['CG', 'OR']:
            subset = self.df[self.df['label'] == label]
            plt.hist(subset['length'], bins=50, alpha=0.5, label=label, density=True)
        plt.title('Review Length Distribution by Label', fontsize=14, fontweight='bold')
        plt.xlabel('Review Length (characters)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('plots/length_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved to 'plots/' directory")
    
    def prepare_data(self, test_size=0.35, random_state=42):
        """Prepare data for training"""
        print("\n" + "="*70)
        print("Data Preparation")
        print("="*70)
        
        # Use a subset of data for faster training (optional - remove if you want full dataset)
        # Uncomment the next line if you want to use a subset for faster testing
        # self.df = self.df.sample(n=10000, random_state=random_state)
        
        # Separate features and labels
        X = self.df['text_']
        y = self.df['label']
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"\nClass distribution in full dataset:")
        print(y.value_counts())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        print(f"\nClass distribution in training set:")
        print(self.y_train.value_counts())
        
        return True
    
    def create_pipelines(self):
        """Create pipelines for different classifiers with optimized parameters"""
        # Common preprocessing steps - limit features for speed
        self.vectorizer = CountVectorizer(
            max_features=3000,  # Reduced from 5000 for speed
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_transformer = TfidfTransformer()
        
        # Define pipelines for each classifier with optimized parameters
        self.models = {
            'Multinomial Naive Bayes': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', MultinomialNB())
            ]),
            
            'Logistic Regression': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', LogisticRegression(max_iter=500, random_state=42, n_jobs=-1))
            ]),
            
            'Support Vector Classifier': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', LinearSVC(random_state=42, max_iter=2000, dual=False))  # Much faster than SVC
            ]),
            
            'Random Forest': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
            ]),
            
            'Decision Tree': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', DecisionTreeClassifier(random_state=42))
            ]),
            
            'K-Nearest Neighbors': Pipeline([
                ('vect', self.vectorizer),
                ('tfidf', self.tfidf_transformer),
                ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
            ])
        }
        
        print(f"\n✓ Created {len(self.models)} optimized model pipelines")
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("Training and Evaluating All Models")
        print("="*70)
        
        results = []
        
        for name, pipeline in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            # Train
            start_time = time()
            pipeline.fit(self.X_train, self.y_train)
            train_time = time() - start_time
            
            # Predict
            y_pred = pipeline.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score (use fewer folds for speed)
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=3)
            
            # Store results
            self.results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_time': train_time,
                'predictions': y_pred
            }
            
            print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"✓ Training Time: {train_time:.2f} seconds")
            
            # Quick summary
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print(f"✓ True Positives: {tp}, True Negatives: {tn}")
            print(f"✓ False Positives: {fp}, False Negatives: {fn}")
            
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy*100:.2f}%",
                'CV Score': f"{cv_scores.mean()*100:.2f}%",
                'Train Time (s)': f"{train_time:.2f}"
            })
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['pipeline']
        
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   Test Accuracy: {self.results[best_model_name]['accuracy']*100:.2f}%")
        print("="*70)
        
        # Display results table
        results_df = pd.DataFrame(results)
        print("\n📊 Model Performance Summary:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def visualize_results(self):
        """Visualize model comparison results"""
        print("\n" + "="*70)
        print("Creating Model Comparison Visualizations")
        print("="*70)
        
        # Prepare data
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] * 100 for m in models]
        train_times = [self.results[m]['train_time'] for m in models]
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        accuracies_sorted = [accuracies[i] for i in sorted_indices]
        
        # Bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models_sorted, accuracies_sorted, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies_sorted):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Training time chart
        plt.figure(figsize=(12, 6))
        time_sorted = [train_times[list(self.results.keys()).index(m)] for m in models_sorted]
        bars = plt.bar(models_sorted, time_sorted, color='lightcoral')
        plt.xlabel('Model')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/training_time.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✓ Comparison visualizations saved to 'plots/' directory")
    
    def save_models(self, output_dir='models'):
        """Save trained models and transformers"""
        print("\n" + "="*70)
        print("Saving Models")
        print("="*70)
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        best_model_path = os.path.join(output_dir, 'best_model.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Best model saved: {best_model_path}")
        
        # Save vectorizer separately
        vect_path = os.path.join(output_dir, 'vectorizer.pkl')
        with open(vect_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Vectorizer saved: {vect_path}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'best_accuracy': self.results[self.best_model_name]['accuracy'],
            'num_models': len(self.models),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
        
        # Save performance summary as CSV
        summary = []
        for name, results in self.results.items():
            summary.append({
                'Model': name,
                'Test Accuracy': f"{results['accuracy']*100:.2f}%",
                'CV Mean': f"{results['cv_mean']*100:.2f}%",
                'Training Time (s)': f"{results['train_time']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
        print(f"✓ Performance summary saved: {os.path.join(output_dir, 'performance_summary.csv')}")
    
    def test_predictions(self):
        """Test the best model with sample predictions"""
        print("\n" + "="*70)
        print("Sample Predictions")
        print("="*70)
        
        # Get sample reviews from test set
        sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
        
        for idx in sample_indices:
            review = self.X_test.iloc[idx]
            true_label = self.y_test.iloc[idx]
            pred_label = self.best_model.predict([review])[0]
            
            # Get decision function for confidence (LinearSVC doesn't have predict_proba)
            if hasattr(self.best_model.named_steps['clf'], 'decision_function'):
                decision = self.best_model.decision_function([review])[0]
                confidence = abs(decision) / (max(abs(decision), 1)) * 100
            else:
                confidence = None
            
            print(f"\nReview: {review[:100]}..." if len(review) > 100 else f"Review: {review}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {pred_label}")
            if confidence:
                print(f"Confidence: {confidence:.1f}%")
            print(f"{'✓ CORRECT' if true_label == pred_label else '✗ INCORRECT'}")
            print("-" * 50)
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        print("="*70)
        print("FAKE REVIEWS DETECTION - MODEL TRAINING (OPTIMIZED)")
        print("="*70)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Visualize data
        self.visualize_data()
        
        # Step 4: Prepare data
        if not self.prepare_data():
            return False
        
        # Step 5: Create pipelines
        self.create_pipelines()
        
        # Step 6: Train and evaluate all models
        results_df = self.train_and_evaluate_all()
        
        # Step 7: Visualize results
        self.visualize_results()
        
        # Step 8: Save models
        self.save_models()
        
        # Step 9: Test predictions
        self.test_predictions()
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Run 'python fake_review_predictor.py' for command-line predictions")
        print("2. Run 'streamlit run streamlit_app.py' for web interface")
        
        return True

def main():
    """Main function"""
    trainer = FakeReviewTrainer()
    trainer.run_pipeline()

if __name__ == "__main__":
    main()