#!/usr/bin/env python3
"""
YouTube Video Performance Predictor
Deep Learning Neural Network to predict view count and like count
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    print("Installing TensorFlow...")
    os.system("pip install tensorflow")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Installing visualization libraries...")
    os.system("pip install matplotlib seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

class YouTubePredictor:
    def __init__(self):
        self.model = None
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_columns = []
        self.stats = {}
        self.use_simple_model = False
        
    def load_data(self, file_path):
        """Load and initial preprocessing of YouTube data from CSV or JSON."""
        print(f"📊 Loading data from {file_path}...")
        
        # Determine file format
        if file_path.endswith('.json'):
            # Load JSON format (from batch scraper)
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            # Extract video data from JSON structure
            video_records = []
            for video in batch_data.get('videos', []):
                if not video.get('success', False) or not video.get('metadata'):
                    continue
                
                metadata = video['metadata']
                transcript = video.get('transcript', {})
                
                record = {
                    'video_id': metadata.get('video_id', ''),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'uploader': metadata.get('uploader', ''),
                    'upload_date': metadata.get('upload_date', ''),
                    'duration_seconds': metadata.get('duration', 0),
                    'view_count': metadata.get('view_count', 0),
                    'like_count': metadata.get('like_count', 0),
                    'comment_count': metadata.get('comment_count', 0),
                    'tags': ', '.join(metadata.get('tags', [])),
                    'categories': ', '.join(metadata.get('categories', [])),
                    'url': metadata.get('url', ''),
                    'transcript_text': transcript.get('transcript_text', ''),
                    'transcript_language': transcript.get('language_used', ''),
                    'transcript_auto_generated': transcript.get('auto_generated', False),
                    'word_count': transcript.get('word_count', 0),
                    'has_transcript': bool(transcript.get('transcript_text')),
                    'scraped_at': video.get('scraped_at', '')
                }
                
                video_records.append(record)
            
            df = pd.DataFrame(video_records)
            
        elif file_path.endswith('.csv'):
            # Load CSV format
            df = pd.read_csv(file_path)
            
        else:
            raise ValueError("File must be either .csv or .json format")
        
        print(f"   Loaded {len(df)} videos")
        
        # For training data, we need view_count and like_count
        # For prediction data, these columns might not exist
        required_columns = ['title']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Basic data cleaning (only if we have the columns)
        if 'view_count' in df.columns and 'like_count' in df.columns:
            # Training mode - clean the target variables
            df = df.dropna(subset=['title', 'view_count', 'like_count'])
            df = df[df['view_count'] > 0]  # Remove videos with 0 views
            df = df[df['like_count'] >= 0]  # Remove invalid like counts
            print(f"   After cleaning: {len(df)} videos")
        else:
            # Prediction mode - just clean the title
            df = df.dropna(subset=['title'])
            print(f"   Ready for prediction: {len(df)} videos")
        
        return df
    
    def extract_features(self, df):
        """Extract and engineer features from the raw data."""
        print("🔧 Engineering features...")
        
        # Create a copy for feature engineering
        features_df = df.copy()
        
        # 1. Text preprocessing
        features_df['title_clean'] = features_df['title'].fillna('').astype(str)
        features_df['description_clean'] = features_df['description'].fillna('').astype(str)
        
        # 2. Text length features
        features_df['title_length'] = features_df['title_clean'].str.len()
        features_df['description_length'] = features_df['description_clean'].str.len()
        features_df['title_word_count'] = features_df['title_clean'].str.split().str.len()
        
        # 3. Title sentiment features (simple)
        features_df['title_caps_ratio'] = features_df['title_clean'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        features_df['title_exclamation'] = features_df['title_clean'].str.count('!').astype(int)
        features_df['title_question'] = features_df['title_clean'].str.count('\\?').astype(int)
        
        # 4. Duration features
        features_df['duration_minutes'] = features_df['duration_seconds'].fillna(0) / 60
        features_df['duration_category'] = pd.cut(
            features_df['duration_minutes'], 
            bins=[0, 1, 5, 15, 30, 60, float('inf')], 
            labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extremely_long']
        )
        
        # 5. Upload timing features
        features_df['upload_date'] = pd.to_datetime(features_df['upload_date'], format='%Y%m%d', errors='coerce')
        features_df['upload_year'] = features_df['upload_date'].dt.year
        features_df['upload_month'] = features_df['upload_date'].dt.month
        features_df['upload_weekday'] = features_df['upload_date'].dt.dayofweek
        features_df['days_since_upload'] = (datetime.now() - features_df['upload_date']).dt.days
        
        # 6. Channel features
        features_df['uploader_clean'] = features_df['uploader'].fillna('unknown').astype(str)
        
        # 7. Content category features
        features_df['categories_clean'] = features_df['categories'].fillna('').astype(str)
        features_df['tags_clean'] = features_df['tags'].fillna('').astype(str)
        features_df['tags_count'] = features_df['tags_clean'].str.split(',').str.len()
        
        # 8. Engagement ratio (only calculate if we have the data - not for predictions!)
        if 'view_count' in features_df.columns and 'like_count' in features_df.columns:
            features_df['like_view_ratio'] = features_df['like_count'] / (features_df['view_count'] + 1)
        else:
            # For predictions, we don't have these values yet
            features_df['like_view_ratio'] = 0.0  # Neutral default
        
        return features_df
    
    def prepare_ml_features(self, df, is_training=True):
        """Prepare features for machine learning."""
        print("🎯 Preparing ML features...")
        
        # Numerical features
        numerical_features = [
            'title_length', 'description_length', 'title_word_count',
            'title_caps_ratio', 'title_exclamation', 'title_question',
            'duration_minutes', 'upload_year', 'upload_month', 'upload_weekday',
            'days_since_upload', 'tags_count'
        ]
        
        # Handle missing values
        for col in numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if is_training else 0)
        
        # Scale numerical features
        if is_training:
            self.scalers['numerical'] = StandardScaler()
            numerical_scaled = self.scalers['numerical'].fit_transform(df[numerical_features])
        else:
            numerical_scaled = self.scalers['numerical'].transform(df[numerical_features])
        
        # Categorical features
        categorical_features = ['duration_category']
        categorical_encoded = np.array([]).reshape(len(df), 0)
        
        for col in categorical_features:
            if col in df.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    encoded = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    encoded = []
                    for val in df[col].astype(str):
                        if val in self.encoders[col].classes_:
                            encoded.append(self.encoders[col].transform([val])[0])
                        else:
                            encoded.append(0)  # Default to first class
                    encoded = np.array(encoded)
                
                encoded = encoded.reshape(-1, 1)
                categorical_encoded = np.hstack([categorical_encoded, encoded])
        
        # Text features using TF-IDF
        text_features = ['title_clean']
        text_vectorized = np.array([]).reshape(len(df), 0)
        
        for col in text_features:
            if col in df.columns:
                if is_training:
                    self.vectorizers[col] = TfidfVectorizer(
                        max_features=100, 
                        stop_words='english',
                        ngram_range=(1, 2),
                        lowercase=True
                    )
                    vectorized = self.vectorizers[col].fit_transform(df[col]).toarray()
                else:
                    vectorized = self.vectorizers[col].transform(df[col]).toarray()
                
                text_vectorized = np.hstack([text_vectorized, vectorized])
        
        # Combine all features
        all_features = np.hstack([numerical_scaled, categorical_encoded, text_vectorized])
        
        print(f"   Feature shape: {all_features.shape}")
        return all_features
    
    def build_model(self, input_dim, simple=False):
        """Build the neural network model."""
        print("🧠 Building neural network...")
        
        # Input layer
        inputs = keras.Input(shape=(input_dim,))
        
        if simple:
            # Simplified model for small datasets
            print("   Using simplified architecture for small dataset")
            x = layers.Dense(32, activation='relu')(inputs)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(16, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        else:
            # Full model for large datasets
            x = layers.Dense(512, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output branches
        # View count prediction (log scale)
        view_branch = layers.Dense(16 if simple else 32, activation='relu', name='view_branch')(x)
        view_output = layers.Dense(1, activation='linear', name='view_count')(view_branch)
        
        # Like count prediction (log scale)
        like_branch = layers.Dense(16 if simple else 32, activation='relu', name='like_branch')(x)
        like_output = layers.Dense(1, activation='linear', name='like_count')(like_branch)
        
        # Create model
        model = keras.Model(
            inputs=inputs,
            outputs=[view_output, like_output],
            name='youtube_predictor'
        )
        
        # Compile with custom loss weights
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'view_count': 'mse',
                'like_count': 'mse'
            },
            loss_weights={
                'view_count': 1.0,
                'like_count': 0.5  # Like count is generally harder to predict
            },
            metrics={
                'view_count': ['mae'],
                'like_count': ['mae']
            }
        )
        
        return model
    
    def train(self, data_path, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model on YouTube data from CSV or JSON file."""
        print("🚀 Starting training pipeline...")
        
        # Load and preprocess data
        df = self.load_data(data_path)
        df = self.extract_features(df)
        
        # Check if we have enough data
        min_samples = 100
        if len(df) < min_samples:
            print(f"\n⚠️  WARNING: Only {len(df)} samples found!")
            print(f"   Minimum recommended: {min_samples} samples")
            print(f"   For good results: 500+ samples")
            print(f"\n💡 Solutions:")
            print(f"   • Scrape more data: python youtube_scraper.py -t 500")
            print(f"   • Use transfer learning with pre-trained weights")
            print(f"   • Try a simpler model architecture")
            
            if len(df) < 20:
                print(f"\n❌ Cannot train with < 20 samples. Exiting...")
                return None
            
            # Adjust model architecture for small datasets
            print(f"\n🔧 Using simplified model for small dataset...")
            self.use_simple_model = True
        else:
            self.use_simple_model = False
        
        # Prepare features
        X = self.prepare_ml_features(df, is_training=True)
        
        # Prepare targets (log transform for better training)
        if 'view_count' not in df.columns or 'like_count' not in df.columns:
            raise ValueError("Training data must include view_count and like_count columns")
            
        y_views = np.log1p(df['view_count'].values)
        y_likes = np.log1p(df['like_count'].values)
        
        # Store statistics for later use
        self.stats = {
            'view_count_mean': df['view_count'].mean(),
            'view_count_std': df['view_count'].std(),
            'like_count_mean': df['like_count'].mean(),
            'like_count_std': df['like_count'].std(),
            'training_samples': len(df)
        }
        
        # Split data
        if len(df) < 50:
            # For very small datasets, use less data for validation
            validation_split = 0.1
            batch_size = min(batch_size, len(df) // 3)  # Smaller batch size
        
        X_train, X_test, y_views_train, y_views_test, y_likes_train, y_likes_test = train_test_split(
            X, y_views, y_likes, test_size=validation_split, random_state=42
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Build model
        self.model = self.build_model(X.shape[1], simple=self.use_simple_model)
        print(f"   Model parameters: {self.model.count_params():,}")
        
        # Adjust training parameters for small datasets
        if len(df) < 50:
            epochs = min(epochs, 30)  # Fewer epochs for small datasets
            patience = 5  # Less patience
            print(f"   Adjusted epochs to {epochs} for small dataset")
        else:
            patience = 15
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=max(3, patience//3), min_lr=1e-6
        )
        
        # Train model
        print("🎯 Training model...")
        history = self.model.fit(
            X_train, 
            {'view_count': y_views_train, 'like_count': y_likes_train},
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, {'view_count': y_views_test, 'like_count': y_likes_test}),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        print("📊 Evaluating model...")
        test_loss = self.model.evaluate(
            X_test, 
            {'view_count': y_views_test, 'like_count': y_likes_test},
            verbose=0
        )
        
        # Make predictions for metrics
        pred_views, pred_likes = self.model.predict(X_test, verbose=0)
        pred_views = np.expm1(pred_views.flatten())
        pred_likes = np.expm1(pred_likes.flatten())
        actual_views = np.expm1(y_views_test)
        actual_likes = np.expm1(y_likes_test)
        
        # Calculate metrics
        view_mae = mean_absolute_error(actual_views, pred_views)
        like_mae = mean_absolute_error(actual_likes, pred_likes)
        view_r2 = r2_score(actual_views, pred_views)
        like_r2 = r2_score(actual_likes, pred_likes)
        
        print(f"\n📈 Model Performance:")
        print(f"   View Count - MAE: {view_mae:,.0f}, R²: {view_r2:.3f}")
        print(f"   Like Count - MAE: {like_mae:,.0f}, R²: {like_r2:.3f}")
        
        # Provide guidance based on dataset size and performance
        if len(df) < 100:
            print(f"\n💡 Performance Tips:")
            print(f"   • Current dataset: {len(df)} videos")
            print(f"   • For better accuracy, collect 500+ videos")
            print(f"   • Run: python youtube_scraper.py -t 500")
            print(f"   • Try diverse search terms for better generalization")
        
        return history
    
    def predict_single(self, video_data):
        """Predict views and likes for a single video."""
        if self.model is None:
            raise ValueError("Model not trained! Run train() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([video_data])
        df = self.extract_features(df)
        X = self.prepare_ml_features(df, is_training=False)
        
        # Make prediction
        pred_views, pred_likes = self.model.predict(X, verbose=0)
        
        # Convert back from log scale
        pred_views = int(np.expm1(pred_views[0][0]))
        pred_likes = int(np.expm1(pred_likes[0][0]))
        
        return pred_views, pred_likes
    
    def save_model(self, path="models/youtube_predictor"):
        """Save the trained model and preprocessors."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(f"{path}/model.h5")
        
        # Save preprocessors
        with open(f"{path}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(f"{path}/encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(f"{path}/vectorizers.pkl", 'wb') as f:
            pickle.dump(self.vectorizers, f)
        
        with open(f"{path}/stats.json", 'w') as f:
            json.dump(self.stats, f)
        
        print(f"💾 Model saved to {path}/")
    
    def load_model(self, path="models/youtube_predictor"):
        """Load a trained model and preprocessors."""
        try:
            # Load model with custom objects to handle compatibility issues
            custom_objects = {
                'mse': 'mean_squared_error',
                'mae': 'mean_absolute_error'
            }
            
            self.model = keras.models.load_model(
                f"{path}/model.h5", 
                custom_objects=custom_objects,
                compile=False  # Skip compilation to avoid metric issues
            )
            
            # Recompile with current TensorFlow version
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss={
                    'view_count': 'mse',
                    'like_count': 'mse'
                },
                loss_weights={
                    'view_count': 1.0,
                    'like_count': 0.5
                },
                metrics={
                    'view_count': ['mae'],
                    'like_count': ['mae']
                }
            )
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        # Load preprocessors
        with open(f"{path}/scalers.pkl", 'rb') as f:
            self.scalers = pickle.load(f)
        
        with open(f"{path}/encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)
        
        with open(f"{path}/vectorizers.pkl", 'rb') as f:
            self.vectorizers = pickle.load(f)
        
        with open(f"{path}/stats.json", 'r') as f:
            self.stats = json.load(f)
        
        print(f"📂 Model loaded from {path}/")

def interactive_prediction(model_path="models/youtube_predictor"):
    """Interactive terminal interface for predictions."""
    print("\n🎬 YouTube Video Performance Predictor")
    print("=" * 50)
    
    # Load model
    predictor = YouTubePredictor()
    try:
        predictor.load_model(model_path)
    except Exception as e:
        print(f"❌ No trained model found at {model_path}!")
        print(f"   Error: {e}")
        print("\n💡 Solutions:")
        print("   1. Train a model first:")
        print("      python3 main.py --train ../data/your_dataset.csv")
        print("   2. Check the model path:")
        print(f"      ls {model_path}/")
        print("   3. Use a different model path:")
        print("      python3 main.py --predict --model-path ./path/to/model")
        return
    
    print("📝 Enter video details for prediction:")
    print("(Press Enter for defaults)")
    
    while True:
        try:
            # Get user input
            title = input("\n🎯 Video Title: ").strip()
            if not title:
                title = "How to Build Neural Networks"
            
            description = input("📄 Description: ").strip()
            if not description:
                description = "Learn machine learning and neural networks in this comprehensive tutorial"
            
            duration = input("⏱️  Duration (seconds): ").strip()
            if not duration:
                duration = "600"
            duration = int(duration)
            
            uploader = input("👤 Channel Name: ").strip()
            if not uploader:
                uploader = "TechChannel"
            
            categories = input("🏷️  Categories (comma-separated): ").strip()
            if not categories:
                categories = "Education"
            
            tags = input("🔖 Tags (comma-separated): ").strip()
            if not tags:
                tags = "tutorial,machine learning,AI"
            
            # Create video data (without target variables we're trying to predict!)
            video_data = {
                'title': title,
                'description': description,
                'duration_seconds': duration,
                'uploader': uploader,
                'categories': categories,
                'tags': tags,
                'upload_date': datetime.now().strftime('%Y%m%d'),
                # NOTE: We don't include view_count and like_count since we're predicting them!
            }
            
            # Make prediction
            print("\n🧠 Making prediction...")
            pred_views, pred_likes = predictor.predict_single(video_data)
            
            # Display results
            print(f"\n🎯 Predictions:")
            print(f"   👀 Expected Views: {pred_views:,}")
            print(f"   👍 Expected Likes: {pred_likes:,}")
            print(f"   📊 Like Rate: {(pred_likes/pred_views*100):.2f}%")
            
            # Show context
            avg_views = predictor.stats.get('view_count_mean', 0)
            avg_likes = predictor.stats.get('like_count_mean', 0)
            
            print(f"\n📈 Context (training data averages):")
            print(f"   Average Views: {avg_views:,.0f}")
            print(f"   Average Likes: {avg_likes:,.0f}")
            
            if pred_views > avg_views * 1.5:
                print("   🚀 This video is predicted to perform above average!")
            elif pred_views < avg_views * 0.5:
                print("   📉 This video may underperform")
            else:
                print("   📊 This video is predicted to perform around average")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Ask to continue
        continue_pred = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
        if continue_pred != 'y':
            break

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Performance Predictor')
    parser.add_argument('--train', help='Path to training data file (.csv or .json)')
    parser.add_argument('--predict', action='store_true', help='Start interactive prediction')
    parser.add_argument('--model-path', default='models/youtube_predictor', help='Path to model directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if args.train:
        # Training mode
        predictor = YouTubePredictor()
        history = predictor.train(args.train, epochs=args.epochs, batch_size=args.batch_size)
        
        # Only save model if training was successful
        if history is not None:
            predictor.save_model(args.model_path)
            print(f"\n✅ Training complete! Model saved to {args.model_path}")
            print(f"🚀 Use predictions: python3 main.py --predict --model-path {args.model_path}")
        else:
            print("\n❌ Training failed! Please collect more data and try again.")
            print("\n💡 Quick fix:")
            print("   python3 captions3.py -t 100 --output-csv ../data/more_data.csv")
            print("   python clean_data.py")
            print("   python3 main.py --train ../data/clean_youtube_dataset.csv")
        
    elif args.predict:
        # Prediction mode
        interactive_prediction(args.model_path)
        
    else:
        print("Usage:")
        print("  Train: python predictor.py --train ../data/youtube_dataset.csv")
        print("  Train: python predictor.py --train ../data/youtube_batch_20241201_143022.json")
        print("  Predict: python predictor.py --predict")

if __name__ == "__main__":
    main()

# Example usage:
# python youtube_predictor.py --train ../data/youtube_dataset_20241201_143022.csv --epochs 50
# python youtube_predictor.py --train ../data/youtube_batch_20241201_143022.json --epochs 50
# python youtube_predictor.py --predict