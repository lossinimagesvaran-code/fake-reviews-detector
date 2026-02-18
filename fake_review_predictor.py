import numpy as np
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

class FakeReviewPredictor:
    """Predict whether reviews are fake or real using trained models"""
    
    def __init__(self, model_dir='models'):
        """
        Initialize the predictor
        
        Parameters:
        model_dir (str): Directory containing saved models
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.results = None
        
    def load_model(self):
        """Load the trained best model"""
        try:
            # Load best model
            model_path = os.path.join(self.model_dir, 'best_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded best model from {model_path}")
            
            # Load vectorizer
            vect_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            with open(vect_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✓ Loaded vectorizer from {vect_path}")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✓ Model: {self.metadata['best_model']}")
                print(f"✓ Accuracy: {self.metadata['best_accuracy']*100:.2f}%")
            
            # Load results for comparison
            results_path = os.path.join(self.model_dir, 'model_results.pkl')
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    self.results = pickle.load(f)
            
            return True
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model: {e}")
            print("\nPlease run 'python train_fake_review_model.py' first to train the model.")
            return False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to string if needed
        text = str(text)
        
        # Basic cleaning
        text = text.lower().strip()
        
        return text
    
    def predict(self, review_text, return_probability=True):
        """
        Predict if a review is fake or real
        
        Parameters:
        review_text (str): The review text to analyze
        return_probability (bool): Whether to return probability scores
        
        Returns:
        dict: Prediction results
        """
        # Preprocess
        review_clean = self.preprocess_text(review_text)
        
        # Make prediction
        prediction = self.model.predict([review_clean])[0]
        
        result = {
            'review': review_text[:100] + '...' if len(review_text) > 100 else review_text,
            'prediction': prediction,
            'is_fake': prediction == 'CG',
            'is_real': prediction == 'OR'
        }
        
        # Get probability if available
        if return_probability and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([review_clean])[0]
            classes = self.model.classes_
            
            for i, cls in enumerate(classes):
                result[f'prob_{cls}'] = proba[i]
            
            result['confidence'] = max(proba) * 100
        
        return result
    
    def predict_batch(self, reviews):
        """
        Predict multiple reviews at once
        
        Parameters:
        reviews (list): List of review texts
        
        Returns:
        list: List of prediction results
        """
        results = []
        for review in reviews:
            results.append(self.predict(review))
        return results
    
    def analyze_review_features(self, review_text):
        """
        Analyze text features that might indicate fake reviews
        
        Parameters:
        review_text (str): Review text
        
        Returns:
        dict: Feature analysis
        """
        text = str(review_text).lower()
        words = text.split()
        
        features = {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words': len(set(words)),
            'unique_ratio': len(set(words)) / len(words) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0,
        }
        
        # Common fake review patterns
        suspicious_patterns = [
            'love it', 'great product', 'awesome', 'perfect',
            'highly recommend', 'best ever', 'amazing'
        ]
        
        features['suspicious_pattern_count'] = sum(
            1 for pattern in suspicious_patterns if pattern in text
        )
        
        return features
    
    def compare_models(self, review_text):
        """
        Compare predictions from all available models
        
        Parameters:
        review_text (str): Review text
        
        Returns:
        dict: Predictions from all models
        """
        if not self.results:
            return {"error": "No model results available"}
        
        predictions = {}
        for name, result in self.results.items():
            pipeline = result['pipeline']
            pred = pipeline.predict([review_text])[0]
            predictions[name] = pred
        
        return predictions

def main():
    """Command-line interface"""
    print("="*70)
    print("FAKE REVIEWS DETECTION - COMMAND LINE PREDICTOR")
    print("="*70)
    
    # Initialize predictor
    predictor = FakeReviewPredictor()
    
    # Load model
    if not predictor.load_model():
        sys.exit(1)
    
    print("\nEnter a review to analyze (or 'quit' to exit, 'batch' for batch mode)")
    print("-" * 70)
    
    while True:
        print("\nOptions:")
        print("1. Single review prediction")
        print("2. Batch prediction (multiple reviews)")
        print("3. Compare all models")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Single review
            review = input("\nEnter review text: ").strip()
            
            if not review:
                print("Please enter a review.")
                continue
            
            print("\n" + "="*50)
            print("ANALYZING REVIEW")
            print("="*50)
            
            # Get prediction
            result = predictor.predict(review)
            
            print(f"\n📝 Review: {result['review']}")
            print(f"\n🎯 Prediction: {result['prediction']}")
            
            if result['is_fake']:
                print("⚠️ This appears to be a FAKE (computer-generated) review")
            else:
                print("✓ This appears to be a REAL (human-written) review")
            
            if 'confidence' in result:
                print(f"📊 Confidence: {result['confidence']:.1f}%")
                if 'prob_CG' in result:
                    print(f"   Fake probability: {result['prob_CG']*100:.1f}%")
                    print(f"   Real probability: {result['prob_OR']*100:.1f}%")
            
            # Analyze features
            features = predictor.analyze_review_features(review)
            print("\n📈 Text Features:")
            print(f"   Length: {features['length']} characters")
            print(f"   Words: {features['word_count']}")
            print(f"   Unique words: {features['unique_words']} ({features['unique_ratio']*100:.1f}%)")
            print(f"   Exclamations: {features['exclamation_count']}")
            print(f"   Suspicious patterns: {features['suspicious_pattern_count']}")
            
        elif choice == '2':
            # Batch prediction
            print("\nEnter multiple reviews (one per line, empty line to finish):")
            reviews = []
            while True:
                line = input().strip()
                if not line:
                    break
                reviews.append(line)
            
            if not reviews:
                print("No reviews entered.")
                continue
            
            print(f"\nProcessing {len(reviews)} reviews...")
            results = predictor.predict_batch(reviews)
            
            # Display results
            print("\n" + "="*60)
            print("BATCH PREDICTION RESULTS")
            print("="*60)
            
            for i, result in enumerate(results, 1):
                status = "FAKE" if result['is_fake'] else "REAL"
                confidence = result.get('confidence', 0)
                print(f"{i:2}. [{status}] {result['review'][:50]}... (conf: {confidence:.1f}%)")
            
            # Summary
            fake_count = sum(1 for r in results if r['is_fake'])
            real_count = len(results) - fake_count
            
            print("\n" + "="*40)
            print("SUMMARY")
            print("="*40)
            print(f"Total reviews: {len(results)}")
            print(f"Fake (CG): {fake_count} ({fake_count/len(results)*100:.1f}%)")
            print(f"Real (OR): {real_count} ({real_count/len(results)*100:.1f}%)")
            
        elif choice == '3':
            # Compare all models
            review = input("\nEnter review text: ").strip()
            
            if not review:
                print("Please enter a review.")
                continue
            
            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            
            predictions = predictor.compare_models(review)
            
            for model, pred in predictions.items():
                status = "FAKE" if pred == 'CG' else "REAL"
                print(f"{model:25}: {status}")
            
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()