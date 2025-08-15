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
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    print("Installing youtube-transcript-api...")
    os.system("pip install youtube-transcript-api")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import TextFormatter
        TRANSCRIPT_API_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: youtube-transcript-api not available. Transcripts will be skipped.")
        YouTubeTranscriptApi = None
        TextFormatter = None
        TRANSCRIPT_API_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    PROXY_SCRAPING_AVAILABLE = True
except ImportError:
    print("Installing web scraping packages...")
    os.system("pip install requests beautifulsoup4")
    try:
        import requests
        from bs4 import BeautifulSoup
        PROXY_SCRAPING_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: Proxy scraping not available. Using direct connection only.")
        PROXY_SCRAPING_AVAILABLE = False
        requests = None
        BeautifulSoup = None

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

try:
    import requests
    from bs4 import BeautifulSoup
    PROXY_SCRAPING_AVAILABLE = True
except ImportError:
    print("Installing web scraping packages...")
    os.system("pip install requests beautifulsoup4")
    try:
        import requests
        from bs4 import BeautifulSoup
        PROXY_SCRAPING_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: Proxy scraping not available. Using direct connection only.")
        PROXY_SCRAPING_AVAILABLE = False

class ProxyManager:
    def __init__(self):
        self.working_proxies = []
        self.current_proxy_index = 0
        self.failed_proxies = set()
        self.last_fetch_time = 0
        self.fetch_interval = 300  # Fetch new proxies every 5 minutes
        
    def fetch_free_proxies(self) -> List[str]:
        """Fetch free proxies from multiple sources."""
        if not PROXY_SCRAPING_AVAILABLE or not requests:
            return []
        
        proxies = []
        
        # Source 1: Free Proxy List
        try:
            print("   🔍 Fetching proxies from free-proxy-list...")
            response = requests.get("https://free-proxy-list.net/", timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': 'proxylisttable'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows[:20]:  # Limit to first 20
                    cols = row.find_all('td')
                    if len(cols) >= 7:
                        ip = cols[0].text.strip()
                        port = cols[1].text.strip()
                        https = cols[6].text.strip()
                        
                        if https == 'yes':  # Only HTTPS proxies
                            proxy = f"http://{ip}:{port}"
                            proxies.append(proxy)
        except Exception as e:
            print(f"   ⚠️  Failed to fetch from free-proxy-list: {e}")
        
        # Source 2: ProxyScrape API
        try:
            print("   🔍 Fetching proxies from proxyscrape...")
            api_url = "https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=5000&country=all&ssl=yes&anonymity=all"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                proxy_list = response.text.strip().split('\n')
                for proxy in proxy_list[:10]:  # Limit to first 10
                    if ':' in proxy:
                        proxy = f"http://{proxy.strip()}"
                        proxies.append(proxy)
        except Exception as e:
            print(f"   ⚠️  Failed to fetch from proxyscrape: {e}")
        
        # Source 3: PubProxy API
        try:
            print("   🔍 Fetching proxies from pubproxy...")
            api_url = "http://pubproxy.com/api/proxy?limit=10&format=txt&https=true"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                proxy_list = response.text.strip().split('\n')
                for proxy in proxy_list:
                    if ':' in proxy:
                        proxy = f"http://{proxy.strip()}"
                        proxies.append(proxy)
        except Exception as e:
            print(f"   ⚠️  Failed to fetch from pubproxy: {e}")
        
        # Remove duplicates and failed proxies
        unique_proxies = list(set(proxies))
        fresh_proxies = [p for p in unique_proxies if p not in self.failed_proxies]
        
        print(f"   📡 Found {len(fresh_proxies)} new proxies")
        return fresh_proxies
    
    def test_proxy(self, proxy: str, timeout: int = 5) -> bool:
        """Test if a proxy is working."""
        if not requests:
            return False
        
        try:
            test_url = "https://httpbin.org/ip"
            response = requests.get(
                test_url, 
                proxies={'http': proxy, 'https': proxy},
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            return response.status_code == 200
        except:
            return False
    
    def get_working_proxies(self, max_proxies: int = 5) -> List[str]:
        """Get a list of working proxies."""
        current_time = time.time()
        
        # Fetch new proxies if needed
        if (current_time - self.last_fetch_time) > self.fetch_interval or not self.working_proxies:
            print("🔄 Refreshing proxy list...")
            fresh_proxies = self.fetch_free_proxies()
            
            # Test proxies in parallel (but limit to avoid overwhelming)
            working_proxies = []
            for proxy in fresh_proxies[:20]:  # Test max 20 proxies
                if self.test_proxy(proxy):
                    working_proxies.append(proxy)
                    print(f"   ✅ Working proxy: {proxy}")
                    if len(working_proxies) >= max_proxies:
                        break
                else:
                    self.failed_proxies.add(proxy)
            
            self.working_proxies = working_proxies
            self.current_proxy_index = 0
            self.last_fetch_time = current_time
            
            print(f"🎯 Found {len(self.working_proxies)} working proxies")
        
        return self.working_proxies
    
    def get_next_proxy(self) -> Optional[str]:
        """Get the next proxy in rotation."""
        if not self.working_proxies:
            self.get_working_proxies()
        
        if not self.working_proxies:
            return None
        
        proxy = self.working_proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.working_proxies)
        return proxy
    
    def mark_proxy_failed(self, proxy: str):
        """Mark a proxy as failed and remove it from working list."""
        self.failed_proxies.add(proxy)
        if proxy in self.working_proxies:
            self.working_proxies.remove(proxy)
            print(f"❌ Removed failed proxy: {proxy}")

class YouTubeBatchScraper:
    def __init__(self, delay_range=(1, 3), use_proxies=False):
        self.formatter = TextFormatter() if TRANSCRIPT_API_AVAILABLE else None
        self.delay_range = delay_range
        self.scraped_urls = set()
        self.use_proxies = use_proxies
        self.proxy_manager = ProxyManager() if use_proxies else None
        self.current_proxy = None
        
        # Initialize stats tracking
        self.stats = {
            'transcript_errors': 0,
            'proxy_switches': 0
        }
        
        if use_proxies:
            print("🔄 Proxy rotation enabled - fetching proxy list...")
            self.proxy_manager.get_working_proxies()
        
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
    def make_request_with_proxy(self, url: str, timeout: int = 10):
        """Make a request using current proxy with fallback."""
        if not requests:
            raise Exception("Requests library not available")
        
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                proxy = self.get_current_proxy()
                proxies = {'http': proxy, 'https': proxy} if proxy else None
                
                response = requests.get(url, proxies=proxies, headers=headers, timeout=timeout)
                return response
                
            except Exception as e:
                if proxy and attempt < max_retries - 1:
                    print(f"   🔄 Proxy failed, rotating...")
                    self.rotate_proxy()
                    time.sleep(1)
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e
        
        raise Exception("All proxy attempts failed")
        
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
                    output_dir: str = "../data", custom_filename: str = None, 
                    output_format: str = "both", proxy: str = None, require_transcript: bool = None) -> Dict:
        """Scrape a batch of YouTube videos automatically."""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine transcript requirement behavior
        if require_transcript is None:
            # Default: require transcripts if we're extracting them
            require_transcript = include_transcript
        
        print(f"🎯 Target: {target_count} videos")
        if include_transcript:
            if require_transcript:
                print("📝 Mode: Only videos with transcripts will be included")
            else:
                print("📝 Mode: Videos included regardless of transcript status")
        else:
            print("📝 Mode: Transcript extraction disabled")
        
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
        
        # Determine output filenames
        if custom_filename:
            base_name = custom_filename.replace('.json', '').replace('.csv', '')
            json_file = Path(output_dir) / f"{base_name}.json"
            csv_file = Path(output_dir) / f"{base_name}.csv"
            combined_file = Path(output_dir) / f"{base_name}_complete.json"
        else:
            json_file = Path(output_dir) / f"youtube_batch_{timestamp}.json"
            csv_file = Path(output_dir) / f"youtube_dataset_{timestamp}.csv"
            combined_file = Path(output_dir) / f"youtube_complete_{timestamp}.json"
        
        # Save based on output format
        if output_format in ["json", "both"]:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"   💾 JSON saved: {json_file}")
        
        if output_format in ["csv", "both"]:
            self.export_batch_to_csv(results, csv_file)
            print(f"   📊 CSV saved: {csv_file}")
        
        if output_format == "combined":
            # Create a comprehensive single file with everything
            combined_data = {
                'batch_info': results['batch_info'],
                'stats': results['stats'],
                'summary': {
                    'total_videos': len(results['videos']),
                    'successful_videos': results['stats']['successful_scrapes'],
                    'videos_with_transcripts': results['stats']['videos_with_transcripts'],
                    'average_views': sum(v['metadata']['view_count'] for v in results['videos'] if v['success'] and v['metadata']) / max(1, results['stats']['successful_scrapes']),
                    'average_likes': sum(v['metadata']['like_count'] for v in results['videos'] if v['success'] and v['metadata']) / max(1, results['stats']['successful_scrapes']),
                    'total_transcript_words': results['stats']['total_words']
                },
                'videos': []
            }
            
            # Add flattened video data for easy analysis
            for video in results['videos']:
                if video['success'] and video['metadata']:
                    metadata = video['metadata']
                    transcript = video.get('transcript', {})
                    
                    flattened_video = {
                        # Basic info
                        'video_id': metadata['video_id'],
                        'title': metadata['title'],
                        'description': metadata['description'],
                        'url': metadata['url'],
                        
                        # Channel info
                        'uploader': metadata['uploader'],
                        'upload_date': metadata['upload_date'],
                        
                        # Video metrics
                        'duration_seconds': metadata['duration'],
                        'view_count': metadata['view_count'],
                        'like_count': metadata['like_count'],
                        'comment_count': metadata['comment_count'],
                        
                        # Content
                        'tags': metadata['tags'],
                        'categories': metadata['categories'],
                        'thumbnail': metadata['thumbnail'],
                        
                        # Transcript info
                        'transcript_text': transcript.get('transcript_text', ''),
                        'transcript_language': transcript.get('language_used', ''),
                        'transcript_auto_generated': transcript.get('auto_generated', False),
                        'transcript_word_count': transcript.get('word_count', 0),
                        'has_transcript': bool(transcript.get('transcript_text')),
                        'transcript_error': 'error' in transcript,
                        
                        # Derived metrics
                        'duration_minutes': metadata['duration'] / 60 if metadata['duration'] else 0,
                        'like_rate_percent': (metadata['like_count'] / metadata['view_count'] * 100) if metadata['view_count'] > 0 else 0,
                        'title_length': len(metadata['title']),
                        'description_length': len(metadata['description']),
                        'tags_count': len(metadata['tags']) if isinstance(metadata['tags'], list) else len(str(metadata['tags']).split(',')),
                        
                        # Scraping info
                        'scraped_at': video['scraped_at'],
                        'source_batch': timestamp
                    }
                    
                    combined_data['videos'].append(flattened_video)
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"   📦 Combined file saved: {combined_file}")
        
        # Print summary with helpful tips
        stats = results['stats']
        success_rate = stats['successful_scrapes'] / (stats['successful_scrapes'] + stats['failed_scrapes']) * 100 if (stats['successful_scrapes'] + stats['failed_scrapes']) > 0 else 0
        
        print(f"\n📊 Batch Complete!")
        print(f"   ✅ Successful: {stats['successful_scrapes']}")
        print(f"   ❌ Failed: {stats['failed_scrapes']}")
        print(f"   📝 With transcripts: {stats['videos_with_transcripts']}")
        print(f"   📖 Total words: {stats['total_words']:,}")
        print(f"   🎯 Success rate: {success_rate:.1f}%")
        
        # Show output files based on format
        if output_format == "combined":
            print(f"   📦 All data in one file: {combined_file}")
        elif output_format == "json":
            print(f"   💾 JSON file: {json_file}")
        elif output_format == "csv":
            print(f"   📊 CSV file: {csv_file}")
        else:  # both
            print(f"   💾 JSON file: {json_file}")
            print(f"   📊 CSV file: {csv_file}")
        
        # Show proxy stats if using proxies
        if self.use_proxies:
            print(f"   🔄 Proxy switches: {self.stats.get('proxy_switches', 0)}")
            working_proxies = len(self.proxy_manager.working_proxies) if self.proxy_manager else 0
            print(f"   📡 Working proxies: {working_proxies}")
        
        # Give helpful tips based on success rate
        if success_rate < 50:
            print(f"\n💡 Low success rate detected! Try these solutions:")
            if not self.use_proxies:
                print(f"   • Try --use-proxies to rotate through free proxies")
            print(f"   • Use --safe-mode for slower, more reliable scraping")
            print(f"   • Try different search terms (avoid news/politics)")
            print(f"   • Use a VPN to change your IP address")
            print(f"   • Run smaller batches (-t 50) and combine results")
        elif success_rate < 80:
            print(f"\n💡 Consider using --safe-mode for better reliability")
            if not self.use_proxies:
                print(f"   • Try --use-proxies for even better success rates")
        
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
    parser.add_argument('--filename', help='Custom base filename for output files')
    
    # Single video options
    parser.add_argument('url', nargs='?', help='Single YouTube video URL')
    
    # Export options
    parser.add_argument('--output-csv', help='Custom CSV filename')
    parser.add_argument('--output-json', help='Custom JSON filename')
    parser.add_argument('--format', choices=['json', 'csv', 'both', 'combined'], default='both',
                        help='Output format: json, csv, both, or combined (default: both)')
    
    # General options
    parser.add_argument('--no-transcript', action='store_true', help='Skip transcript extraction')
    parser.add_argument('--require-transcript', action='store_true', 
                       help='Only include videos that have transcripts (default when using transcripts)')
    parser.add_argument('--allow-no-transcript', action='store_true',
                       help='Include videos even if transcript extraction fails')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Preferred transcript languages')
    parser.add_argument('--delay', nargs=2, type=float, default=[1, 3], 
                       help='Delay range between requests (min max)')
    parser.add_argument('--safe-mode', action='store_true', 
                       help='Use slower, safer scraping (5-10 sec delays)')
    parser.add_argument('--proxy', help='HTTP proxy (e.g., http://proxy:8080)')
    parser.add_argument('--use-proxies', action='store_true', 
                       help='Use rotating free proxies to avoid blocks')
    parser.add_argument('--max-proxies', type=int, default=5,
                       help='Maximum number of proxies to use (default: 5)')
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
    
    # Initialize scraper with proxy support
    use_proxies = args.use_proxies
    if use_proxies and not PROXY_SCRAPING_AVAILABLE:
        print("⚠️  Proxy scraping not available. Install requests and beautifulsoup4.")
        use_proxies = False
    
    scraper = YouTubeBatchScraper(delay_range=delay_range, use_proxies=use_proxies)
    
    if args.target:
        # Batch scraping mode
        
        # Handle legacy options
        custom_filename = args.filename
        if not custom_filename and (args.output_csv or args.output_json):
            custom_filename = args.output_csv or args.output_json
            custom_filename = custom_filename.replace('.csv', '').replace('.json', '')
        
        # Determine transcript requirement
        require_transcript = None
        if args.allow_no_transcript:
            require_transcript = False
        elif args.require_transcript:
            require_transcript = True
        # else: None (auto-determine based on transcript extraction)
        
        scraper.scrape_batch(
            target_count=args.target,
            search_terms=args.search_terms,
            include_transcript=not args.no_transcript,
            languages=args.languages,
            output_dir=args.output_dir,
            custom_filename=custom_filename,
            output_format=args.format,
            proxy=args.proxy,
            require_transcript=require_transcript
        )
    else:
        # Single video mode (legacy)
        print("Single video mode - use batch mode with -t for training data collection")

if __name__ == "__main__":
    main()

# Example usage for ML training data:
# python youtube_scraper.py -t 500                                    # Scrape 500 random videos (transcripts required by default)
# python youtube_scraper.py -t 100 --search-terms "python tutorial"  # 100 Python tutorials
# python youtube_scraper.py -t 250 --output-dir ./my_data             # Custom output directory
# python youtube_scraper.py -t 500 --filename my_dataset              # Custom filename (creates my_dataset.json/.csv)
# python youtube_scraper.py -t 100 --filename training_data --format combined  # Single comprehensive file
# python youtube_scraper.py -t 300 --filename tech_videos --format csv        # CSV only
# python youtube_scraper.py -t 500 --use-proxies                      # Use rotating free proxies to avoid blocks
# python youtube_scraper.py -t 200 --use-proxies --safe-mode          # Maximum reliability with proxies + slow mode
# python youtube_scraper.py -t 500 --allow-no-transcript              # Include videos even without transcripts
# python youtube_scraper.py -t 200 --no-transcript                    # Skip transcript extraction entirely
# python youtube_scraper.py -t 500 --safe-mode                        # Slower, safer scraping
# python youtube_scraper.py -t 500 --proxy http://proxy:8080          # Use specific HTTP proxy
# python youtube_scraper.py -t 1000 --delay 2 5                      # Custom delay range