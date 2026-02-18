import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Fake Reviews Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FakeReviewDetectorApp:
    """Streamlit app for fake review detection"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.results = None
        self.history = []
        
    def load_model(self):
        """Load trained model"""
        try:
            model_path = 'models/best_model.pkl'
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            vect_path = 'models/vectorizer.pkl'
            with open(vect_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            meta_path = 'models/metadata.pkl'
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please run 'python train_fake_review_model.py' first to train the model.")
            return False
    
    def predict(self, review_text):
        """Make prediction"""
        if not self.model:
            return None
        
        # Preprocess
        review_clean = review_text.lower().strip()
        
        # Predict
        prediction = self.model.predict([review_clean])[0]
        
        result = {
            'text': review_text,
            'prediction': prediction,
            'is_fake': prediction == 'CG',
            'is_real': prediction == 'OR',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([review_clean])[0]
            result['fake_prob'] = proba[0] * 100
            result['real_prob'] = proba[1] * 100
            result['confidence'] = max(proba) * 100
        
        return result
    
    def analyze_features(self, text):
        """Extract text features"""
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {
            'Length': len(text),
            'Word Count': len(words),
            'Avg Word Length': np.mean([len(w) for w in words]) if words else 0,
            'Unique Words': len(set(words)),
            'Exclamation Marks': text.count('!'),
            'Question Marks': text.count('?'),
            'Capital Letters': sum(1 for c in text if c.isupper()),
            'Uppercase Ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features

def main():
    """Main Streamlit app"""
    app = FakeReviewDetectorApp()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Fake Reviews Detector")
        st.markdown("---")
        
        # Model status
        st.subheader("📊 Model Status")
        if app.load_model():
            st.success("✅ Model loaded successfully")
            if app.metadata:
                st.info(f"**Best Model:** {app.metadata['best_model']}")
                st.info(f"**Accuracy:** {app.metadata['best_accuracy']*100:.2f}%")
        else:
            st.error("❌ Model not loaded")
            st.stop()
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        This app detects fake (computer-generated) reviews vs real (human-written) reviews.
        
        **Classes:**
        - **CG**: Computer Generated (Fake)
        - **OR**: Original Review (Real)
        
        **Best Model:** Support Vector Classifier
        **Accuracy:** 88.11%
        """)
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Main content
    st.title("🔍 Fake Reviews Detection System")
    st.markdown("---")
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Review", "📊 Batch Analysis", "📈 Model Comparison", "📚 History"
    ])
    
    with tab1:
        st.header("Analyze a Single Review")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input area
            review_text = st.text_area(
                "Enter your review:",
                height=150,
                placeholder="Type or paste a product review here..."
            )
            
            # Character counter
            if review_text:
                st.caption(f"Characters: {len(review_text)} | Words: {len(review_text.split())}")
            
            # Analyze button
            if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
                if review_text.strip():
                    with st.spinner("Analyzing review..."):
                        time.sleep(0.5)  # Small delay for UX
                        result = app.predict(review_text)
                        
                    if result:
                        # Add to history
                        st.session_state.history.append(result)
                        
                        # Display result
                        st.markdown("---")
                        
                        # Result header
                        if result['is_fake']:
                            st.error("⚠️ **FAKE REVIEW DETECTED**")
                            st.markdown("This appears to be a **computer-generated** review.")
                        else:
                            st.success("✅ **REAL REVIEW**")
                            st.markdown("This appears to be a **human-written** review.")
                        
                        # Confidence meter
                        if 'confidence' in result:
                            st.markdown("### Confidence Score")
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                fake_prob = result.get('fake_prob', 0)
                                st.metric("Fake Probability", f"{fake_prob:.1f}%")
                            
                            with col_b:
                                real_prob = result.get('real_prob', 0)
                                st.metric("Real Probability", f"{real_prob:.1f}%")
                            
                            # Progress bar
                            st.progress(result['confidence'] / 100)
                            st.caption(f"Overall confidence: {result['confidence']:.1f}%")
                        
                        # Feature analysis
                        st.markdown("### 📊 Text Analysis")
                        features = app.analyze_features(review_text)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Length", features['Length'])
                            st.metric("Word Count", features['Word Count'])
                        with col2:
                            st.metric("Unique Words", features['Unique Words'])
                            st.metric("Avg Word Length", f"{features['Avg Word Length']:.1f}")
                        with col3:
                            st.metric("Exclamations", features['Exclamation Marks'])
                            st.metric("Uppercase Ratio", f"{features['Uppercase Ratio']*100:.1f}%")
                        
                        # Show review
                        with st.expander("📝 View Full Review"):
                            st.write(review_text)
                else:
                    st.warning("Please enter a review to analyze.")
        
        with col2:
            st.subheader("💡 Tips")
            st.info("""
            **Signs of Fake Reviews:**
            - Repetitive language
            - Overuse of superlatives
            - Generic phrases
            - Unnatural flow
            - Missing specific details
            
            **Signs of Real Reviews:**
            - Specific product details
            - Balanced opinions
            - Personal experiences
            - Varied vocabulary
            - Natural language patterns
            """)
            
            # Example reviews
            st.subheader("📋 Examples")
            if st.button("Load Fake Example"):
                example = "Love this product! Great quality! Best ever! Highly recommend! Amazing! Perfect! Awesome!"
                st.session_state['example'] = example
                st.rerun()
            
            if st.button("Load Real Example"):
                example = "I've been using this for about 2 weeks now. The build quality seems good but the battery life could be better. It's comfortable to hold and works well for basic tasks."
                st.session_state['example'] = example
                st.rerun()
    
    with tab2:
        st.header("Batch Analysis")
        st.markdown("Analyze multiple reviews at once")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload CSV"]
        )
        
        reviews = []
        
        if input_method == "Manual Entry":
            # Text area for multiple reviews
            batch_text = st.text_area(
                "Enter reviews (one per line):",
                height=200,
                placeholder="Review 1\nReview 2\nReview 3..."
            )
            
            if batch_text:
                reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
        
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with reviews",
                type=['csv']
            )
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if 'review' in df.columns:
                    reviews = df['review'].tolist()
                else:
                    reviews = df.iloc[:, 0].tolist()
                st.success(f"Loaded {len(reviews)} reviews")
        
        # Analyze button
        if st.button("🔍 Analyze Batch", type="primary", use_container_width=True):
            if reviews:
                with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                    results = []
                    for review in reviews:
                        result = app.predict(review)
                        if result:
                            results.append(result)
                            st.session_state.history.append(result)
                
                # Display results
                st.markdown("---")
                st.subheader("Batch Results")
                
                # Summary stats
                fake_count = sum(1 for r in results if r['is_fake'])
                real_count = len(results) - fake_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(results))
                with col2:
                    st.metric("Fake Reviews", fake_count, delta=f"{fake_count/len(results)*100:.1f}%")
                with col3:
                    st.metric("Real Reviews", real_count, delta=f"{real_count/len(results)*100:.1f}%")
                
                # Results table
                df_results = pd.DataFrame([{
                    'Review': r['text'][:50] + '...' if len(r['text']) > 50 else r['text'],
                    'Prediction': 'FAKE' if r['is_fake'] else 'REAL',
                    'Confidence': f"{r.get('confidence', 0):.1f}%"
                } for r in results])
                
                st.dataframe(df_results, use_container_width=True)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )
            else:
                st.warning("Please enter reviews to analyze.")
    
    with tab3:
        st.header("Model Comparison")
        st.markdown("Compare predictions from all available models")
        
        review_text = st.text_area(
            "Enter a review to compare across models:",
            height=100,
            key="compare_review"
        )
        
        if st.button("🔍 Compare Models", type="primary"):
            if review_text.strip():
                # This would require loading all models
                st.info("This feature requires all models to be loaded. Coming soon!")
                
                # Placeholder for model comparison
                models = [
                    "Support Vector Classifier",
                    "Logistic Regression",
                    "Random Forest",
                    "Multinomial Naive Bayes",
                    "Decision Tree",
                    "K-Nearest Neighbors"
                ]
                
                # Simulated predictions
                predictions = ['FAKE', 'FAKE', 'REAL', 'FAKE', 'REAL', 'REAL']
                
                df_compare = pd.DataFrame({
                    'Model': models,
                    'Prediction': predictions
                })
                
                st.dataframe(df_compare, use_container_width=True)
                
                # Accuracy chart
                st.subheader("Model Accuracies")
                accuracies = [88.1, 86.0, 83.6, 84.3, 73.3, 57.7]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(models, accuracies, color='skyblue')
                ax.set_xlabel('Accuracy (%)')
                ax.set_title('Model Performance Comparison')
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
                           f'{acc:.1f}%', va='center')
                
                st.pyplot(fig)
    
    with tab4:
        st.header("Analysis History")
        
        if st.session_state.history:
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.history)
            
            # Summary stats
            total = len(history_df)
            fake = history_df['is_fake'].sum()
            real = total - fake
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyzed", total)
            with col2:
                st.metric("Fake Reviews", fake, delta=f"{fake/total*100:.1f}%")
            with col3:
                st.metric("Real Reviews", real, delta=f"{real/total*100:.1f}%")
            
            # Display history
            st.subheader("Recent Analyses")
            
            for i, item in enumerate(reversed(st.session_state.history[-10:])):
                with st.expander(f"Analysis {len(st.session_state.history) - i} - {item['timestamp']}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if item['is_fake']:
                            st.error("FAKE")
                        else:
                            st.success("REAL")
                        
                        if 'confidence' in item:
                            st.metric("Confidence", f"{item['confidence']:.1f}%")
                    
                    with col2:
                        st.write(f"**Review:** {item['text'][:200]}..." if len(item['text']) > 200 else f"**Review:** {item['text']}")
            
            # Clear history button
            if st.button("🗑️ Clear All History"):
                st.session_state.history = []
                st.rerun()
            
            # Download history
            if st.button("📥 Download History"):
                history_df = pd.DataFrame(st.session_state.history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "analysis_history.csv",
                    "text/csv"
                )
        else:
            st.info("No analysis history yet. Try analyzing some reviews!")

if __name__ == "__main__":
    main()