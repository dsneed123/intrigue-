#!/usr/bin/env python3
"""
YouTube Metadata & Transcript Scraper with Auto-Discovery
A free tool to automatically find and scrape YouTube videos for ML training data.
"""

import os
import json
import csv
import argparse
import random
import time
from datetime import datetime
from typing import Dict, List, Optional
import re
from pathlib import Path

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
except ImportError:
    print("Installing required packages...")
    os.system("pip install youtube-transcript-api")
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter

try:
    import yt_dlp
except ImportError:
    print("Installing yt-dlp...")
    os.system("pip install yt-dlp")
    import yt_dlp

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing web scraping packages...")
    os.system("pip install requests beautifulsoup4")
    import requests
    from bs4 import BeautifulSoup

class YouTubeBatchScraper:
    def __init__(self, delay_range=(1, 3)):
        self.formatter = TextFormatter()
        self.delay_range = delay_range
        self.scraped_urls = set()
        
        # User agents to rotate for better success rate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        # Popular search terms for diverse content
        self.search_categories = [
            # Educational
            "tutorial", "how to", "explained", "learn", "education", "science",
            "technology", "programming", "coding", "math", "physics", "history",
            
            # Entertainment
            "comedy", "funny", "entertainment", "music", "gaming", "sports",
            "travel", "food", "cooking", "art", "movies", "reviews",
            
            # News & Current Events
            "news", "economics", "business",
            "health", "environment", "climate", "society",
            
            # Lifestyle
            "fitness", "wellness", "lifestyle", "fashion", "beauty",
            "relationships", "motivation", "productivity", "mindfulness",
            
            # Hobbies & Interests
            "diy", "crafts", "gardening", "photography", "music production",
            "animation", "design", "writing", "books", "podcast"
        ]
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/shorts\/([^&\n?#]+)',
            r'youtube\.com\/live\/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def search_youtube_videos(self, query: str, max_results: int = 50) -> List[str]:
        """Search for YouTube videos and return list of URLs."""
        video_urls = []
        
        search_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
            'user_agent': random.choice(self.user_agents),
            'referer': 'https://www.youtube.com/',
            'ignoreerrors': True,
        }
        
        search_query = f"ytsearch{max_results}:{query}"
        
        try:
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                search_results = ydl.extract_info(search_query, download=False)
                
                if search_results and 'entries' in search_results:
                    for entry in search_results['entries']:
                        if entry and entry.get('url'):
                            video_urls.append(entry['url'])
                        elif entry and entry.get('id'):
                            video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                            
        except Exception as e:
            print(f"   ⚠️  Search warning for '{query}': {str(e)[:50]}...")
        
        return video_urls
    
    def discover_videos(self, target_count: int, search_terms: List[str] = None) -> List[str]:
        """Automatically discover YouTube videos to scrape."""
        if not search_terms:
            search_terms = random.sample(self.search_categories, min(len(self.search_categories), 10))
        
        discovered_urls = []
        
        print(f"🔍 Discovering videos using {len(search_terms)} search terms...")
        
        videos_per_search = max(10, target_count // len(search_terms))
        
        for i, term in enumerate(search_terms):
            if len(discovered_urls) >= target_count:
                break
                
            print(f"   Searching: '{term}' ({i+1}/{len(search_terms)})")
            
            try:
                urls = self.search_youtube_videos(term, videos_per_search)
                
                # Filter out duplicates and already scraped
                new_urls = [url for url in urls if url not in self.scraped_urls and url not in discovered_urls]
                discovered_urls.extend(new_urls)
                
                # Add delay to be respectful
                time.sleep(random.uniform(*self.delay_range))
                
            except Exception as e:
                print(f"   ❌ Error searching '{term}': {e}")
                continue
        
        # Shuffle for diversity
        random.shuffle(discovered_urls)
        
        return discovered_urls[:target_count]
    
    def get_video_metadata(self, video_url: str, max_retries: int = 3, proxy: str = None) -> Dict:
        """Extract comprehensive metadata from YouTube video with retry logic."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        last_error = None
        
        for attempt in range(max_retries):
            # Configure yt-dlp with anti-detection measures
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': random.choice(self.user_agents),
                'referer': 'https://www.youtube.com/',
                'extractor_retries': 1,
                'fragment_retries': 1,
                'skip_unavailable_fragments': True,
                'ignoreerrors': True,
                'no_check_certificate': True,
                'prefer_insecure': False,
                'geo_bypass': True,
                'socket_timeout': 30,
                # Add cookie and session handling
                'http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            }
            # Add proxy if provided
            if proxy:
                ydl_opts['proxy'] = proxy
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    
                    if not info:
                        raise Exception("No video information extracted")
                    
                    # Check if we actually got meaningful data
                    title = info.get('title')
                    if not title or title == 'N/A' or 'Private video' in str(title):
                        raise Exception("Failed to extract meaningful metadata")
                    
                    metadata = {
                        'video_id': video_id,
                        'title': title,
                        'description': info.get('description', 'N/A'),
                        'uploader': info.get('uploader', 'N/A'),
                        'upload_date': info.get('upload_date', 'N/A'),
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0),
                        'comment_count': info.get('comment_count', 0),
                        'tags': info.get('tags', []),
                        'categories': info.get('categories', []),
                        'thumbnail': info.get('thumbnail', 'N/A'),
                        'url': video_url,
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    return metadata
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check for specific error types
                if "403" in error_str or "Forbidden" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3 + random.uniform(1, 3)  # Longer backoff for 403s
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("YouTube blocked request (403 Forbidden)")
                
                elif "Private video" in error_str or "unavailable" in error_str:
                    raise Exception("Video is private or unavailable")
                
                elif "fragment" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Video fragments not accessible")
                
                else:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                
        raise Exception(f"Failed after {max_retries} attempts: {str(last_error)}")
    
    def get_transcript(self, video_url: str, languages: List[str] = ['en']) -> Dict:
        """Extract transcript/captions from YouTube video."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            transcript_data = {
                'video_id': video_id,
                'available_languages': [],
                'transcript_text': None,
                'language_used': None,
                'auto_generated': False,
                'word_count': 0
            }
            
            # Get available languages
            for transcript in transcript_list:
                transcript_data['available_languages'].append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated
                })
            
            # Try to fetch transcript
            transcript_fetched = None
            for lang in languages:
                try:
                    transcript_fetched = transcript_list.find_transcript([lang])
                    transcript_data['language_used'] = lang
                    transcript_data['auto_generated'] = transcript_fetched.is_generated
                    break
                except:
                    continue
            
            if not transcript_fetched:
                try:
                    transcript_fetched = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                    transcript_data['language_used'] = 'en'
                    transcript_data['auto_generated'] = transcript_fetched.is_generated
                except:
                    try:
                        transcript_fetched = transcript_list.find_generated_transcript(['en'])
                        transcript_data['language_used'] = 'en'
                        transcript_data['auto_generated'] = True
                    except:
                        available = list(transcript_list)
                        if available:
                            transcript_fetched = available[0]
                            transcript_data['language_used'] = available[0].language_code
                            transcript_data['auto_generated'] = available[0].is_generated
            
            if transcript_fetched:
                transcript_json = transcript_fetched.fetch()
                transcript_text = self.formatter.format_transcript(transcript_json)
                transcript_data['transcript_text'] = transcript_text
                transcript_data['word_count'] = len(transcript_text.split())
            
            return transcript_data
            
        except Exception as e:
            return {
                'video_id': video_id,
                'error': f"Error extracting transcript: {str(e)}",
                'transcript_text': None,
                'word_count': 0
            }
    
    def scrape_batch(self, target_count: int, search_terms: List[str] = None, 
                    include_transcript: bool = True, languages: List[str] = ['en'],
                    output_dir: str = "../data", custom_csv: str = None, custom_json: str = None,
                    proxy: str = None) -> Dict:
        """Scrape a batch of YouTube videos automatically."""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            'batch_info': {
                'target_count': target_count,
                'started_at': datetime.now().isoformat(),
                'search_terms': search_terms or 'auto-generated',
                'include_transcript': include_transcript,
                'languages': languages
            },
            'videos': [],
            'stats': {
                'total_discovered': 0,
                'successful_scrapes': 0,
                'failed_scrapes': 0,
                'videos_with_transcripts': 0,
                'total_words': 0
            }
        }
        
        # Discover videos
        print(f"🎯 Target: {target_count} videos")
        discovered_urls = self.discover_videos(target_count * 2, search_terms)  # Get extra in case some fail
        results['stats']['total_discovered'] = len(discovered_urls)
        
        print(f"✅ Discovered {len(discovered_urls)} videos")
        print(f"🚀 Starting batch scrape...")
        
        # Scrape each video
        for i, url in enumerate(discovered_urls[:target_count]):
            if results['stats']['successful_scrapes'] >= target_count:
                break
                
            print(f"   📹 {i+1}/{min(target_count, len(discovered_urls))}: Processing video...")
            
            video_result = {
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'metadata': None,
                'transcript': None,
                'success': False,
                'errors': []
            }
            
            try:
                # Get metadata with better error handling
                video_result['metadata'] = self.get_video_metadata(url, proxy=proxy)
                
                # Double-check we got valid data
                if not video_result['metadata']:
                    raise Exception("Metadata extraction returned None")
                
                title = video_result['metadata'].get('title', '')
                if not title or title == 'N/A':
                    raise Exception("No valid title extracted")
                
                # Get transcript if requested
                if include_transcript:
                    transcript_data = self.get_transcript(url, languages)
                    video_result['transcript'] = transcript_data
                    
                    if transcript_data.get('transcript_text'):
                        results['stats']['videos_with_transcripts'] += 1
                        results['stats']['total_words'] += transcript_data.get('word_count', 0)
                
                video_result['success'] = True
                results['stats']['successful_scrapes'] += 1
                self.scraped_urls.add(url)
                
                print(f"      ✅ Success: {title[:50]}...")
                
            except Exception as e:
                error_msg = str(e)
                video_result['errors'].append(error_msg)
                video_result['success'] = False
                results['stats']['failed_scrapes'] += 1
                
                # Show specific error type with better categorization
                if "403" in error_msg or "Forbidden" in error_msg or "blocked" in error_msg.lower():
                    print(f"      🚫 Blocked: YouTube is blocking this request")
                elif "Private video" in error_msg or "unavailable" in error_msg:
                    print(f"      🔒 Unavailable: Video is private/deleted/restricted")
                elif "fragment" in error_msg:
                    print(f"      📺 Stream issue: Video fragments not accessible")
                elif "timeout" in error_msg.lower():
                    print(f"      ⏰ Timeout: Request took too long")
                else:
                    print(f"      ❌ Failed: {error_msg[:50]}...")
            
            results['videos'].append(video_result)
            
            # Adaptive delay based on success/failure
            base_delay = random.uniform(*self.delay_range)
            if not video_result.get('success', False):
                base_delay *= 2.5  # Much longer delay after failures
                if "403" in str(video_result.get('errors', [])):
                    base_delay *= 2  # Even longer for 403s
            
            time.sleep(base_delay)
        
        # Save results
        results['batch_info']['completed_at'] = datetime.now().isoformat()
        
        # Export to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        if custom_json:
            json_file = Path(output_dir) / custom_json
        else:
            json_file = Path(output_dir) / f"youtube_batch_{timestamp}.json"
            
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        if custom_csv:
            csv_file = Path(output_dir) / custom_csv
        else:
            csv_file = Path(output_dir) / f"youtube_dataset_{timestamp}.csv"
            
        self.export_batch_to_csv(results, csv_file)
        
        # Print summary with helpful tips
        stats = results['stats']
        success_rate = stats['successful_scrapes'] / (stats['successful_scrapes'] + stats['failed_scrapes']) * 100 if (stats['successful_scrapes'] + stats['failed_scrapes']) > 0 else 0
        
        print(f"\n📊 Batch Complete!")
        print(f"   ✅ Successful: {stats['successful_scrapes']}")
        print(f"   ❌ Failed: {stats['failed_scrapes']}")
        print(f"   📝 With transcripts: {stats['videos_with_transcripts']}")
        print(f"   📖 Total words: {stats['total_words']:,}")
        print(f"   🎯 Success rate: {success_rate:.1f}%")
        print(f"   💾 Data saved to: {output_dir}/")
        
        # Give helpful tips based on success rate
        if success_rate < 50:
            print(f"\n💡 Low success rate detected! Try these solutions:")
            print(f"   • Use --safe-mode for slower, more reliable scraping")
            print(f"   • Try different search terms (avoid news/politics)")
            print(f"   • Use a VPN to change your IP address")
            print(f"   • Run smaller batches (-t 50) and combine results")
        elif success_rate < 80:
            print(f"\n💡 Consider using --safe-mode for better reliability")
        
        return results
    
    def export_batch_to_csv(self, batch_results: Dict, filename: str):
        """Export batch results to CSV format suitable for ML training."""
        
        rows = []
        for video in batch_results['videos']:
            if not video['success'] or not video['metadata']:
                continue
            
            metadata = video['metadata']
            transcript = video.get('transcript', {})
            
            row = {
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
            
            rows.append(row)
        
        if rows:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='YouTube Batch Scraper for ML Training Data')
    
    # Batch scraping options
    parser.add_argument('-t', '--target', type=int, help='Target number of videos to scrape')
    parser.add_argument('--search-terms', nargs='+', help='Custom search terms')
    parser.add_argument('--output-dir', default='../data', help='Output directory (default: ../data)')
    
    # Single video options
    parser.add_argument('url', nargs='?', help='Single YouTube video URL')
    
    # Export options
    parser.add_argument('--output-csv', help='Custom CSV filename')
    parser.add_argument('--output-json', help='Custom JSON filename')
    
    # General options
    parser.add_argument('--no-transcript', action='store_true', help='Skip transcript extraction')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Preferred transcript languages')
    parser.add_argument('--delay', nargs=2, type=float, default=[1, 3], 
                       help='Delay range between requests (min max)')
    parser.add_argument('--safe-mode', action='store_true', 
                       help='Use slower, safer scraping (5-10 sec delays)')
    parser.add_argument('--proxy', help='HTTP proxy (e.g., http://proxy:8080)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    
    args = parser.parse_args()
    
    if not args.target and not args.url:
        parser.error("Either specify a target count (-t) for batch scraping or provide a URL")
    
    # Adjust delays for safe mode
    delay_range = tuple(args.delay)
    if args.safe_mode:
        delay_range = (5, 10)
        print("🐌 Safe mode enabled: Using 5-10 second delays")
    
    scraper = YouTubeBatchScraper(delay_range=delay_range)
    
    if args.target:
        # Batch scraping mode
        scraper.scrape_batch(
            target_count=args.target,
            search_terms=args.search_terms,
            include_transcript=not args.no_transcript,
            languages=args.languages,
            output_dir=args.output_dir,
            custom_csv=args.output_csv,
            custom_json=args.output_json,
            proxy=args.proxy
        )
    else:
        # Single video mode (legacy)
        print("Single video mode - use batch mode with -t for training data collection")

if __name__ == "__main__":
    main()

# Example usage for ML training data:
# python youtube_scraper.py -t 500                                    # Scrape 500 random videos
# python youtube_scraper.py -t 100 --search-terms "python tutorial"  # 100 Python tutorials
# python youtube_scraper.py -t 250 --output-dir ./my_data             # Custom output directory
# python youtube_scraper.py -t 500 --output-csv my_dataset.csv        # Custom CSV filename
# python youtube_scraper.py -t 100 --output-json results.json --output-csv training_data.csv
# python youtube_scraper.py -t 500 --safe-mode                        # Slower, safer scraping
# python youtube_scraper.py -t 500 --proxy http://proxy:8080          # Use HTTP proxy
# python youtube_scraper.py -t 1000 --delay 2 5                      # Custom delay range