import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
import re
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ContentDataset(Dataset):
    """Custom dataset for social media content data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class IntrigueModel(nn.Module):
    """Neural network model for predicting engagement metrics"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(IntrigueModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer for 3 targets: likes, comments, views
        layers.append(nn.Linear(prev_size, 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class IntriguePredictor:
    """Main class for the Intrigue system"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.feature_names = []
        self.model_trained = False
        
    def extract_text_features(self, text):
        """Extract features from text content"""
        if pd.isna(text):
            text = ""
        
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(str(text))
        features['word_count'] = len(str(text).split())
        features['sentence_count'] = len(str(text).split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in str(text).split()]) if str(text).split() else 0
        
        # Engagement indicators
        features['exclamation_count'] = str(text).count('!')
        features['question_count'] = str(text).count('?')
        features['caps_ratio'] = sum(1 for c in str(text) if c.isupper()) / len(str(text)) if len(str(text)) > 0 else 0
        features['hashtag_count'] = str(text).count('#')
        features['mention_count'] = str(text).count('@')
        
        # Emotional indicators
        positive_words = ['amazing', 'awesome', 'incredible', 'fantastic', 'wonderful', 'great', 'love', 'best']
        features['positive_words'] = sum(1 for word in positive_words if word in str(text).lower())
        
        action_words = ['watch', 'see', 'check', 'try', 'get', 'buy', 'click', 'follow', 'share']
        features['action_words'] = sum(1 for word in action_words if word in str(text).lower())
        
        return features
    
    def prepare_features(self, df):
        """Prepare features from the dataset"""
        print("Extracting features from content...")
        
        # Extract text features for both title and script
        title_features = df['title'].apply(self.extract_text_features)
        script_features = df['script'].apply(self.extract_text_features)
        
        # Convert to dataframes
        title_df = pd.DataFrame(title_features.tolist())
        title_df.columns = [f'title_{col}' for col in title_df.columns]
        
        script_df = pd.DataFrame(script_features.tolist())
        script_df.columns = [f'script_{col}' for col in script_df.columns]
        
        # Combine features
        feature_df = pd.concat([title_df, script_df], axis=1)
        
        # Add TF-IDF features for title and script combined
        combined_text = df['title'].fillna('') + ' ' + df['script'].fillna('')
        tfidf_features = self.text_vectorizer.fit_transform(combined_text).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        final_features = pd.concat([feature_df, tfidf_df], axis=1)
        
        self.feature_names = final_features.columns.tolist()
        return final_features.values
    
    def prepare_targets(self, df):
        """Prepare target variables (likes, comments, views)"""
        targets = df[['likes', 'comments', 'views']].values
        # Log transform to handle large values
        targets = np.log1p(targets)
        return targets

def print_banner():
    print("""
    
██╗███╗   ██╗████████╗██████╗ ██╗ ██████╗ ██╗   ██╗███████╗    
██║████╗  ██║╚══██╔══╝██╔══██╗██║██╔════╝ ██║   ██║██╔════╝    
██║██╔██╗ ██║   ██║   ██████╔╝██║██║  ███╗██║   ██║█████╗      
██║██║╚██╗██║   ██║   ██╔══██╗██║██║   ██║██║   ██║██╔══╝      
██║██║ ╚████║   ██║   ██║  ██║██║╚██████╔╝╚██████╔╝███████╗    
╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝  ╚═════╝ ╚══════╝    
-----------------------------------------------------------
Intrigue: AI-Powered Social Media Engagement Predictor
Analyzes content to predict virality and engagement metrics
-----------------------------------------------------------
Options:
    1). Collect data
    2). Clean and process data
    3). Generate model
    4). Test video
    5). View analytics                                             

    """)

def collect_data():
    """Function to collect or load social media data"""
    print("\n=== DATA COLLECTION ===")
    print("1. Load existing CSV file")
    print("2. Create sample dataset")
    print("3. Data collection guidelines")
    
    choice = input("Select option (1-3): ")
    
    if choice == "1":
        file_path = input("Enter CSV file path: ")
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Successfully loaded {len(df)} records")
            print("\nDataset preview:")
            print(df.head())
            
            # Validate required columns
            required_cols = ['title', 'script', 'likes', 'comments', 'views']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠ Missing required columns: {missing_cols}")
            else:
                print("✓ All required columns present")
                
        except Exception as e:
            print(f"✗ Error loading file: {e}")
    
    elif choice == "2":
        print("Creating sample dataset...")
        create_sample_dataset()
    
    elif choice == "3":
        print("\n=== DATA COLLECTION GUIDELINES ===")
        print("Required CSV format: title, script, likes, comments, views")
        print("\nData sources:")
        print("- Use only public, ethically sourced data")
        print("- Respect platform terms of service")
        print("- Consider data privacy and consent")
        print("- Ensure diverse and representative samples")
        print("\nRecommended data size: 1000+ samples for training")

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    print("Generating sample dataset...")
    
    # Sample titles and scripts
    sample_data = []
    
    titles = [
        "Amazing Life Hack You Need to Try!",
        "Day in My Life as a Content Creator",
        "Cooking the Perfect Pasta",
        "Why This Changed Everything",
        "You Won't Believe What Happened",
        "Simple Morning Routine",
        "Best Tips for Productivity",
        "Trying Viral Food Trends",
        "Behind the Scenes Magic",
        "Epic Transformation Story"
    ]
    
    scripts = [
        "In this video I'll show you an incredible life hack that will save you time and money every day",
        "Good morning everyone! Today I'm taking you through my typical day as a content creator",
        "Today we're making the most delicious pasta you've ever tasted using just 5 ingredients",
        "I discovered something that completely changed my perspective and I had to share it with you",
        "Wait until you see what happens next - this story is absolutely incredible and totally true",
        "Let me walk you through my simple 10-minute morning routine that sets up my entire day",
        "These productivity tips have helped thousands of people achieve their goals faster",
        "Testing out all the viral food trends from social media to see which ones actually work",
        "Ever wondered how we create our content? Here's the behind-the-scenes process",
        "This transformation took 6 months and the results are absolutely mind-blowing"
    ]
    
    # Generate realistic engagement data
    np.random.seed(42)
    for i in range(100):
        title = np.random.choice(titles)
        script = np.random.choice(scripts)
        
        # Simulate engagement based on content quality
        base_engagement = np.random.exponential(1000)
        likes = int(base_engagement * np.random.uniform(0.8, 1.2))
        comments = int(likes * np.random.uniform(0.05, 0.15))
        views = int(likes * np.random.uniform(10, 50))
        
        sample_data.append({
            'title': title,
            'script': script,
            'likes': likes,
            'comments': comments,
            'views': views
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_social_media_data.csv', index=False)
    print(f"✓ Created sample dataset with {len(df)} records")
    print("✓ Saved as 'sample_social_media_data.csv'")

def clean_and_process_data():
    """Clean and preprocess the social media data"""
    print("\n=== DATA CLEANING & PROCESSING ===")
    
    file_path = input("Enter CSV file path (or press Enter for sample data): ")
    if not file_path:
        file_path = 'sample_social_media_data.csv'
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        
        # Data cleaning
        print("\nCleaning data...")
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['script'] = df['script'].fillna('')
        
        # Remove rows with missing engagement metrics
        df = df.dropna(subset=['likes', 'comments', 'views'])
        print(f"Removed rows with missing engagement data: {initial_count - len(df)} rows")
        
        # Remove outliers (very high or very low engagement)
        for col in ['likes', 'comments', 'views']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Final dataset: {len(df)} records")
        
        # Save cleaned data
        output_path = 'cleaned_' + os.path.basename(file_path)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved cleaned data to {output_path}")
        
        # Show data statistics
        print("\nData Statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"✗ Error processing data: {e}")

def generate_model():
    """Train the machine learning model"""
    print("\n=== MODEL TRAINING ===")
    
    file_path = input("Enter cleaned CSV file path: ")
    try:
        df = pd.read_csv(file_path)
        print(f"Training on {len(df)} records")
        
        # Initialize predictor
        predictor = IntriguePredictor()
        
        # Prepare features and targets
        features = predictor.prepare_features(df)
        targets = predictor.prepare_targets(df)
        
        # Scale features
        features_scaled = predictor.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Feature dimensions: {features_scaled.shape[1]}")
        
        # Create datasets and dataloaders
        train_dataset = ContentDataset(X_train, y_train)
        test_dataset = ContentDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = IntrigueModel(input_size=features_scaled.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        num_epochs = 100
        train_losses = []
        val_losses = []
        
        print("\nStarting training...")
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Evaluate model
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = model(batch_features)
                predictions.extend(outputs.numpy())
                actuals.extend(batch_targets.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        print("\n=== MODEL EVALUATION ===")
        metrics = ['Likes', 'Comments', 'Views']
        for i, metric in enumerate(metrics):
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            r2 = r2_score(actuals[:, i], predictions[:, i])
            print(f"{metric} - MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model and components
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_architecture': {
                'input_size': features_scaled.shape[1],
                'hidden_sizes': [512, 256, 128],
                'dropout_rate': 0.3
            },
            'scaler': predictor.scaler,
            'text_vectorizer': predictor.text_vectorizer,
            'feature_names': predictor.feature_names
        }
        
        torch.save(model_data, 'intrigue_model.pth')
        print("✓ Model saved as 'intrigue_model.pth'")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig('training_history.png')
        print("✓ Training history saved as 'training_history.png'")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")

def test_video():
    """Test the trained model on new content"""
    print("\n=== CONTENT TESTING ===")
    
    try:
        # Load model
        model_data = torch.load('intrigue_model.pth')
        
        # Reconstruct model
        arch = model_data['model_architecture']
        model = IntrigueModel(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            dropout_rate=arch['dropout_rate']
        )
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Load preprocessing components
        scaler = model_data['scaler']
        text_vectorizer = model_data['text_vectorizer']
        
        print("✓ Model loaded successfully")
        
        # Get user input
        print("\nEnter content details:")
        title = input("Title: ")
        script = input("Script/Description: ")
        
        # Create predictor instance for feature extraction
        predictor = IntriguePredictor()
        predictor.text_vectorizer = text_vectorizer
        
        # Create temporary dataframe
        test_df = pd.DataFrame({
            'title': [title],
            'script': [script]
        })
        
        # Extract features
        title_features = predictor.extract_text_features(title)
        script_features = predictor.extract_text_features(script)
        
        # Combine features
        combined_features = {}
        for key, value in title_features.items():
            combined_features[f'title_{key}'] = value
        for key, value in script_features.items():
            combined_features[f'script_{key}'] = value
        
        # Get TF-IDF features
        combined_text = title + ' ' + script
        tfidf_features = text_vectorizer.transform([combined_text]).toarray()[0]
        
        # Combine all features
        all_features = list(combined_features.values()) + list(tfidf_features)
        
        # Scale features
        features_scaled = scaler.transform([all_features])
        
        # Make prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            prediction = model(features_tensor).numpy()[0]
        
        # Transform predictions back (reverse log transform)
        predicted_likes = int(np.expm1(prediction[0]))
        predicted_comments = int(np.expm1(prediction[1]))
        predicted_views = int(np.expm1(prediction[2]))
        
        print("\n=== ENGAGEMENT PREDICTION ===")
        print(f"📈 Predicted Likes: {predicted_likes:,}")
        print(f"💬 Predicted Comments: {predicted_comments:,}")
        print(f"👀 Predicted Views: {predicted_views:,}")
        
        # Calculate engagement rate
        engagement_rate = (predicted_likes + predicted_comments) / predicted_views if predicted_views > 0 else 0
        print(f"📊 Engagement Rate: {engagement_rate:.2%}")
        
        # Provide recommendations
        print("\n=== CONTENT ANALYSIS ===")
        
        # Analyze text features
        if title_features['exclamation_count'] > 0:
            print("✓ Good use of excitement with exclamation marks")
        else:
            print("💡 Consider adding excitement with exclamation marks")
            
        if title_features['question_count'] > 0:
            print("✓ Engaging question in title")
            
        if title_features['hashtag_count'] > 0:
            print("✓ Good hashtag usage for discoverability")
        else:
            print("💡 Consider adding relevant hashtags")
            
        if script_features['word_count'] > 20:
            print("✓ Adequate description length")
        else:
            print("💡 Consider expanding your description")
            
        # Virality score
        virality_score = min(100, (predicted_likes / 1000) * 10 + (engagement_rate * 100) * 5)
        print(f"\n🔥 Virality Score: {virality_score:.1f}/100")
        
        if virality_score >= 80:
            print("🌟 High viral potential!")
        elif virality_score >= 60:
            print("📈 Good engagement potential")
        elif virality_score >= 40:
            print("📊 Moderate engagement expected")
        else:
            print("💡 Consider optimizing content for better engagement")
        
    except FileNotFoundError:
        print("✗ Model not found. Please train a model first (option 3)")
    except Exception as e:
        print(f"✗ Error testing content: {e}")

def view_analytics():
    """View analytics and insights"""
    print("\n=== ANALYTICS DASHBOARD ===")
    
    file_path = input("Enter CSV file path for analysis: ")
    try:
        df = pd.read_csv(file_path)
        
        print(f"\nDataset Overview: {len(df)} records")
        
        # Basic statistics
        print("\n=== ENGAGEMENT STATISTICS ===")
        print(df[['likes', 'comments', 'views']].describe())
        
        # Correlation analysis
        print("\n=== CORRELATION MATRIX ===")
        corr_matrix = df[['likes', 'comments', 'views']].corr()
        print(corr_matrix)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution plots
        axes[0, 0].hist(df['likes'], bins=30, alpha=0.7)
        axes[0, 0].set_title('Likes Distribution')
        axes[0, 0].set_xlabel('Likes')
        
        axes[0, 1].hist(df['views'], bins=30, alpha=0.7)
        axes[0, 1].set_title('Views Distribution')
        axes[0, 1].set_xlabel('Views')
        
        # Scatter plots
        axes[1, 0].scatter(df['views'], df['likes'], alpha=0.6)
        axes[1, 0].set_title('Views vs Likes')
        axes[1, 0].set_xlabel('Views')
        axes[1, 0].set_ylabel('Likes')
        
        # Engagement rate
        engagement_rate = (df['likes'] + df['comments']) / df['views']
        axes[1, 1].hist(engagement_rate, bins=30, alpha=0.7)
        axes[1, 1].set_title('Engagement Rate Distribution')
        axes[1, 1].set_xlabel('Engagement Rate')
        
        plt.tight_layout()
        plt.savefig('analytics_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Analytics dashboard saved as 'analytics_dashboard.png'")
        
        # Top performing content
        print("\n=== TOP PERFORMING CONTENT ===")
        top_content = df.nlargest(5, 'likes')[['title', 'likes', 'comments', 'views']]
        print(top_content)
        
    except Exception as e:
        print(f"✗ Error generating analytics: {e}")

if __name__ == "__main__":
    print_banner()
    
    while True:
        option = input("Select an option (1-5) or 'q' to quit: ")
        
        if option == "1":
            collect_data()
        elif option == "2":
            clean_and_process_data()
        elif option == "3":
            generate_model()
        elif option == "4":
            test_video()
        elif option == "5":
            view_analytics()
        elif option.lower() == 'q':
            print("Thanks for using Intrigue! 🚀")
            break
        else:
            print("Invalid option selected.")
        
        print("\n" + "="*60)