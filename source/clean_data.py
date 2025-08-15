#!/usr/bin/env python3
"""
clean_data.py - YouTube Data Cleaner and Combiner
Combines multiple YouTube batch files and cleans the data for ML training.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import re
from datetime import datetime
import numpy as np

class YouTubeDataCleaner:
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'total_videos_found': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'transcript_errors': 0,
            'duplicates_removed': 0,
            'invalid_data_removed': 0,
            'final_clean_videos': 0
        }
    
    def load_batch_file(self, json_file):
        """Load a single batch JSON file and extract video data."""
        print(f"   📁 Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            print(f"      ❌ Error reading file: {e}")
            return []
        
        videos = []
        batch_videos = batch_data.get('videos', [])
        
        for video in batch_videos:
            self.stats['total_videos_found'] += 1
            
            # Check if video extraction was successful
            if not video.get('success', False):
                self.stats['failed_videos'] += 1
                continue
            
            metadata = video.get('metadata')
            if not metadata:
                self.stats['failed_videos'] += 1
                continue
            
            transcript = video.get('transcript', {})
            
            # Handle transcript errors gracefully
            transcript_text = ""
            transcript_error = False
            word_count = 0
            
            if 'error' in transcript:
                self.stats['transcript_errors'] += 1
                transcript_error = True
            elif transcript.get('transcript_text'):
                transcript_text = transcript['transcript_text']
                word_count = transcript.get('word_count', len(transcript_text.split()))
            
            # Create clean record
            record = {
                'video_id': metadata.get('video_id', ''),
                'title': self.clean_text(metadata.get('title', '')),
                'description': self.clean_text(metadata.get('description', '')),
                'uploader': self.clean_text(metadata.get('uploader', '')),
                'upload_date': metadata.get('upload_date', ''),
                'duration_seconds': self.safe_int(metadata.get('duration', 0)),
                'view_count': self.safe_int(metadata.get('view_count', 0)),
                'like_count': self.safe_int(metadata.get('like_count', 0)),
                'comment_count': self.safe_int(metadata.get('comment_count', 0)),
                'tags': self.clean_tags(metadata.get('tags', [])),
                'categories': self.clean_categories(metadata.get('categories', [])),
                'url': metadata.get('url', ''),
                'transcript_text': self.clean_text(transcript_text),
                'transcript_language': transcript.get('language_used', ''),
                'transcript_auto_generated': transcript.get('auto_generated', False),
                'word_count': word_count,
                'has_transcript': bool(transcript_text and not transcript_error),
                'transcript_error': transcript_error,
                'scraped_at': video.get('scraped_at', ''),
                'source_file': json_file.name
            }
            
            videos.append(record)
            self.stats['successful_videos'] += 1
        
        print(f"      ✅ Extracted {len(videos)} valid videos")
        return videos
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '').replace('\ufeff', '')
        
        return text
    
    def clean_tags(self, tags):
        """Clean and format tags."""
        if not tags:
            return ""
        
        if isinstance(tags, str):
            # Already a string, just clean it
            return self.clean_text(tags)
        
        if isinstance(tags, list):
            # Join list into comma-separated string
            clean_tags = [self.clean_text(tag) for tag in tags if tag and str(tag).strip()]
            return ', '.join(clean_tags)
        
        return ""
    
    def clean_categories(self, categories):
        """Clean and format categories."""
        if not categories:
            return ""
        
        if isinstance(categories, str):
            return self.clean_text(categories)
        
        if isinstance(categories, list):
            clean_cats = [self.clean_text(cat) for cat in categories if cat and str(cat).strip()]
            return ', '.join(clean_cats)
        
        return ""
    
    def safe_int(self, value):
        """Safely convert value to integer."""
        try:
            if pd.isna(value) or value is None or value == '':
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def validate_record(self, record):
        """Validate that a record has minimum required data."""
        # Must have basic metadata
        if not record.get('video_id') or not record.get('title'):
            return False
        
        # Must have positive view count
        if record.get('view_count', 0) <= 0:
            return False
        
        # Like count must be non-negative
        if record.get('like_count', 0) < 0:
            return False
        
        # Duration should be reasonable
        duration = record.get('duration_seconds', 0)
        if duration < 0 or duration > 86400:  # 0 to 24 hours
            return False
        
        return True
    
    def remove_duplicates(self, df):
        """Remove duplicate videos based on video_id."""
        before_count = len(df)
        
        # Remove duplicates, keeping the first occurrence
        df_clean = df.drop_duplicates(subset=['video_id'], keep='first')
        
        duplicates = before_count - len(df_clean)
        self.stats['duplicates_removed'] = duplicates
        
        if duplicates > 0:
            print(f"   🔄 Removed {duplicates} duplicate videos")
        
        return df_clean
    
    def clean_dataset(self, df):
        """Perform final dataset cleaning."""
        print("🧹 Performing final data cleaning...")
        
        # Remove invalid records
        before_count = len(df)
        df_valid = df[df.apply(self.validate_record, axis=1)]
        invalid_removed = before_count - len(df_valid)
        self.stats['invalid_data_removed'] = invalid_removed
        
        if invalid_removed > 0:
            print(f"   ❌ Removed {invalid_removed} invalid records")
        
        # Fill missing values
        df_valid['description'] = df_valid['description'].fillna('')
        df_valid['uploader'] = df_valid['uploader'].fillna('Unknown')
        df_valid['tags'] = df_valid['tags'].fillna('')
        df_valid['categories'] = df_valid['categories'].fillna('')
        df_valid['transcript_text'] = df_valid['transcript_text'].fillna('')
        
        # Convert upload_date to proper format
        df_valid['upload_date'] = pd.to_datetime(df_valid['upload_date'], format='%Y%m%d', errors='coerce')
        
        # Add derived features for ML
        df_valid['title_length'] = df_valid['title'].str.len()
        df_valid['description_length'] = df_valid['description'].str.len()
        df_valid['duration_minutes'] = df_valid['duration_seconds'] / 60
        df_valid['like_view_ratio'] = df_valid['like_count'] / (df_valid['view_count'] + 1)
        df_valid['has_description'] = (df_valid['description_length'] > 0)
        df_valid['has_tags'] = (df_valid['tags'].str.len() > 0)
        
        self.stats['final_clean_videos'] = len(df_valid)
        
        return df_valid
    
    def combine_batches(self, data_dir="../data", output_file="clean_youtube_dataset.csv"):
        """Combine multiple YouTube batch files into one clean CSV."""
        print("🔍 Scanning for YouTube batch files...")
        
        data_path = Path(data_dir)
        json_files = list(data_path.glob("youtube_batch_*.json"))
        
        if not json_files:
            print(f"❌ No batch files found in {data_dir}")
            print("   Looking for files matching: youtube_batch_*.json")
            return None
        
        self.stats['total_files'] = len(json_files)
        print(f"   Found {len(json_files)} batch files")
        
        # Process all batch files
        all_videos = []
        for json_file in sorted(json_files):
            videos = self.load_batch_file(json_file)
            all_videos.extend(videos)
        
        if not all_videos:
            print("❌ No valid videos found in any batch file")
            return None
        
        # Create DataFrame
        print(f"\n📊 Creating dataset from {len(all_videos)} videos...")
        df = pd.DataFrame(all_videos)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Clean the dataset
        df_clean = self.clean_dataset(df)
        
        # Save to CSV
        output_path = data_path / output_file
        df_clean.to_csv(output_path, index=False)
        
        # Print summary
        self.print_summary(output_path)
        
        return output_path
    
    def print_summary(self, output_path):
        """Print detailed summary of the cleaning process."""
        print(f"\n📈 Data Cleaning Summary")
        print(f"=" * 50)
        print(f"   📁 Files processed: {self.stats['total_files']}")
        print(f"   🎬 Videos found: {self.stats['total_videos_found']}")
        print(f"   ✅ Successful extractions: {self.stats['successful_videos']}")
        print(f"   ❌ Failed extractions: {self.stats['failed_videos']}")
        print(f"   📝 Transcript errors: {self.stats['transcript_errors']}")
        print(f"   🔄 Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"   🗑️  Invalid records removed: {self.stats['invalid_data_removed']}")
        print(f"   🎯 Final clean dataset: {self.stats['final_clean_videos']} videos")
        print(f"   💾 Saved to: {output_path}")
        
        success_rate = (self.stats['successful_videos'] / max(self.stats['total_videos_found'], 1)) * 100
        transcript_rate = ((self.stats['successful_videos'] - self.stats['transcript_errors']) / max(self.stats['successful_videos'], 1)) * 100
        
        print(f"\n📊 Quality Metrics")
        print(f"=" * 50)
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Transcript success: {transcript_rate:.1f}%")
        
        if self.stats['final_clean_videos'] >= 100:
            print(f"   🎉 Great! You have enough data for good ML training")
        elif self.stats['final_clean_videos'] >= 50:
            print(f"   👍 Good dataset size for initial training")
        else:
            print(f"   💡 Consider collecting more data for better accuracy")
            print(f"   Recommendation: python3 captions3.py -t 500")

def main():
    parser = argparse.ArgumentParser(description='Clean and combine YouTube batch data')
    parser.add_argument('--data-dir', default='../data', help='Directory containing batch files')
    parser.add_argument('--output', default='clean_youtube_dataset.csv', help='Output CSV filename')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    cleaner = YouTubeDataCleaner()
    output_path = cleaner.combine_batches(args.data_dir, args.output)
    
    if output_path:
        print(f"\n✅ Data cleaning complete!")
        print(f"🚀 Ready for training:")
        print(f"   python3 main.py --train {output_path}")
        
        if args.stats:
            # Show additional statistics
            df = pd.read_csv(output_path)
            print(f"\n📊 Dataset Statistics")
            print(f"=" * 50)
            print(f"   View count range: {df['view_count'].min():,} - {df['view_count'].max():,}")
            print(f"   Average views: {df['view_count'].mean():,.0f}")
            print(f"   Like count range: {df['like_count'].min():,} - {df['like_count'].max():,}")
            print(f"   Average likes: {df['like_count'].mean():,.0f}")
            print(f"   Videos with transcripts: {df['has_transcript'].sum()}")
            print(f"   Average duration: {df['duration_minutes'].mean():.1f} minutes")
            print(f"   Unique uploaders: {df['uploader'].nunique()}")

if __name__ == "__main__":
    main()

# Example usage:
# python clean_data.py                                    # Clean data in ../data/
# python clean_data.py --data-dir ./data --output my.csv  # Custom paths
# python clean_data.py --stats                            # Show detailed stats