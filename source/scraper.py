import time
import csv
import re
import json
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoYouTubeShortsCollector:
    def __init__(self, headless=True):
        """Initialize the auto-discovery YouTube Shorts collector."""
        self.setup_driver(headless)
        self.shorts_data = []
        self.processed_urls = set()  # Avoid duplicates
        self.current_batch = 0
        
    def setup_driver(self, headless=True):
        """Setup Chrome driver optimized for bulk scraping."""
        chrome_options = Options()
        
        # Performance optimizations for bulk scraping
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-images")  # Faster loading
        # chrome_options.add_argument("--disable-javascript")  # Enable JS for captions
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Memory optimizations
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")
        
        # Anti-detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Randomize user agent
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
        
        try:
            logger.info("Setting up ChromeDriver...")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("✓ Chrome driver setup successful!")
            
        except Exception as e:
            logger.error(f"Error setting up driver: {e}")
            raise
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 10)
    
    def discover_trending_shorts(self):
        """Navigate to trending Shorts page for automatic discovery."""
        try:
            logger.info("Navigating to YouTube Shorts trending page...")
            self.driver.get("https://www.youtube.com/shorts")
            time.sleep(5)
            
            # Accept cookies if prompted
            try:
                cookie_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree')]")
                cookie_button.click()
                time.sleep(2)
            except NoSuchElementException:
                pass
                
            logger.info("✓ Successfully loaded Shorts feed")
            return True
            
        except Exception as e:
            logger.error(f"Error accessing Shorts feed: {e}")
            return False
    
    def discover_shorts_by_search(self, search_terms):
        """Discover Shorts using search terms."""
        discovered_urls = []
        
        for term in search_terms:
            try:
                logger.info(f"Searching for Shorts with term: '{term}'")
                search_url = f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}&sp=EgIYAQ%253D%253D"  # Filter for Shorts
                self.driver.get(search_url)
                time.sleep(3)
                
                # Scroll to load more results
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                    time.sleep(2)
                
                # Find Shorts in search results
                shorts_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/shorts/']")
                
                for element in shorts_elements:
                    href = element.get_attribute('href')
                    if href and href not in self.processed_urls:
                        discovered_urls.append(href)
                        
                logger.info(f"Found {len(shorts_elements)} Shorts for term '{term}'")
                
            except Exception as e:
                logger.error(f"Error searching for term '{term}': {e}")
                
        return discovered_urls
    
    def extract_captions(self):
        """Extract captions from YouTube Short by playing video and collecting segments."""
        all_captions = []
        
        try:
            logger.info("Extracting captions by playing video...")
            
            # Step 1: Ensure captions are enabled
            self.enable_captions()
            
            # Step 2: Play the video and collect captions as they appear
            caption_text = self.collect_dynamic_captions()
            
            # Step 3: Fallback to transcript if available
            if not caption_text:
                caption_text = self.try_transcript_method()
            
            return caption_text
            
        except Exception as e:
            logger.debug(f"Error in caption extraction: {e}")
            return ""
    
    def extract_engagement_metrics(self):
        """Extract likes, comments, and reposts using exact selectors from user findings."""
        engagement = {'likes': '', 'comments': '', 'reposts': ''}
        
        try:
            # Your exact selector for engagement metrics
            engagement_selector = "span.yt-core-attributed-string.yt-core-attributed-string--white-space-pre-wrap.yt-core-attributed-string--text-alignment-center.yt-core-attributed-string--word-wrapping[role='text']"
            
            elements = self.driver.find_elements(By.CSS_SELECTOR, engagement_selector)
            
            logger.debug(f"Found {len(elements)} engagement elements")
            
            # The challenge is distinguishing between likes, comments, and reposts
            # We'll look at the parent elements or nearby elements for context
            for i, element in enumerate(elements):
                try:
                    text = element.text.strip()
                    if not text:
                        continue
                    
                    # Get parent element for context
                    parent = element.find_element(By.XPATH, "./..")
                    parent_html = parent.get_attribute('outerHTML').lower()
                    
                    # Check surrounding elements for context clues
                    surrounding_text = ""
                    try:
                        # Get text from nearby elements
                        siblings = parent.find_elements(By.XPATH, ".//preceding-sibling::*[1] | .//following-sibling::*[1]")
                        surrounding_text = " ".join([s.text.lower() for s in siblings if s.text])
                    except:
                        pass
                    
                    logger.debug(f"Element {i}: '{text}' - Context: '{surrounding_text}' - Parent: {parent_html[:100]}")
                    
                    # Determine type based on context and position
                    if 'like' in parent_html or 'like' in surrounding_text:
                        engagement['likes'] = text
                        logger.debug(f"✓ Found likes: {text}")
                    elif 'comment' in parent_html or 'comment' in surrounding_text:
                        engagement['comments'] = text
                        logger.debug(f"✓ Found comments: {text}")
                    elif 'share' in parent_html or 'repost' in parent_html or 'share' in surrounding_text:
                        engagement['reposts'] = text
                        logger.debug(f"✓ Found reposts: {text}")
                    else:
                        # Fallback: assign based on typical order (like, comment, share)
                        if not engagement['likes']:
                            engagement['likes'] = text
                            logger.debug(f"✓ Assigned to likes (position): {text}")
                        elif not engagement['comments']:
                            engagement['comments'] = text
                            logger.debug(f"✓ Assigned to comments (position): {text}")
                        elif not engagement['reposts']:
                            engagement['reposts'] = text
                            logger.debug(f"✓ Assigned to reposts (position): {text}")
                    
                except Exception as e:
                    logger.debug(f"Error processing engagement element {i}: {e}")
                    continue
            
            # Alternative method: look for aria-labels on buttons
            if not any(engagement.values()):
                self.extract_engagement_from_buttons(engagement)
                
        except Exception as e:
            logger.debug(f"Error extracting engagement metrics: {e}")
        
        return engagement
    
    def extract_engagement_from_buttons(self, engagement):
        """Fallback method to extract engagement from button aria-labels."""
        try:
            # Look for like buttons
            like_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[aria-label*='like' i]")
            for button in like_buttons:
                aria_label = button.get_attribute('aria-label') or ""
                if 'like' in aria_label.lower():
                    # Extract number from aria-label
                    number_match = re.search(r'([\d,]+(?:\.\d+)?[KMB]?)', aria_label)
                    if number_match:
                        engagement['likes'] = number_match.group(1)
                        break
            
            # Look for comment buttons/indicators
            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, "[aria-label*='comment' i], #comments-count")
            for element in comment_elements:
                aria_label = element.get_attribute('aria-label') or element.text
                if aria_label and ('comment' in aria_label.lower() or 'comment' in element.text.lower()):
                    number_match = re.search(r'([\d,]+(?:\.\d+)?[KMB]?)', aria_label)
                    if number_match:
                        engagement['comments'] = number_match.group(1)
                        break
            
            # Look for share/repost buttons
            share_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[aria-label*='share' i], button[aria-label*='repost' i]")
            for button in share_buttons:
                aria_label = button.get_attribute('aria-label') or ""
                if 'share' in aria_label.lower() or 'repost' in aria_label.lower():
                    number_match = re.search(r'([\d,]+(?:\.\d+)?[KMB]?)', aria_label)
                    if number_match:
                        engagement['reposts'] = number_match.group(1)
                        break
                        
        except Exception as e:
            logger.debug(f"Error in button engagement extraction: {e}")
    
    def extract_views(self):
        """Extract view count using multiple methods."""
        try:
            # Method 1: Look for view count elements
            views_selectors = [
                ".view-count",
                "span[class*='view']",
                ".ytd-video-view-count-renderer",
                ".style-scope.ytd-video-view-count-renderer",
                "#info-text"
            ]
            
            for selector in views_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    view_text = element.text.strip()
                    if 'view' in view_text.lower():
                        return view_text
                except NoSuchElementException:
                    continue
            
            # Method 2: Check if views are in the same attributed-string elements
            attributed_strings = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "span.yt-core-attributed-string[role='text']"
            )
            
            for element in attributed_strings:
                text = element.text.strip()
                if 'view' in text.lower() or re.search(r'\d+.*view', text.lower()):
                    return text
            
            # Method 3: Regex search in page source
            views_patterns = [
                r'([\d,]+(?:\.\d+)?[KMB]?)\s*views',
                r'([\d,]+)\s*views',
                r'([\d,]+(?:\.\d+)?[KMB]?)\s*visualizações',  # Portuguese
                r'"viewCountText":\s*{"simpleText":"([^"]+)"'
            ]
            
            page_text = self.driver.page_source
            for pattern in views_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return "views_not_found"
            
        except Exception as e:
            logger.debug(f"Error extracting views: {e}")
            return "views_extraction_error"
    
    def enable_captions(self):
        """Enable captions on the video."""
        try:
            # Look for caption/subtitle buttons
            caption_buttons = [
                "button[aria-label*='Captions' i]",
                "button[aria-label*='Subtitles' i]",
                ".ytp-subtitles-button",
                ".ytp-caption-button"
            ]
            
            for button_selector in caption_buttons:
                try:
                    caption_button = self.driver.find_element(By.CSS_SELECTOR, button_selector)
                    # Check if captions are already enabled
                    aria_pressed = caption_button.get_attribute("aria-pressed")
                    if not aria_pressed or aria_pressed == "false":
                        caption_button.click()
                        time.sleep(1)
                        logger.debug("✓ Captions enabled")
                    return True
                except NoSuchElementException:
                    continue
                    
        except Exception as e:
            logger.debug(f"Could not enable captions: {e}")
        return False
    
    def collect_dynamic_captions(self):
        """Collect captions as video plays using the exact selector you found."""
        all_captions = []
        seen_captions = set()
        
        try:
            # Start the video playing
            self.play_video()
            
            # Collect captions for up to 30 seconds (most Shorts are <60s)
            collection_time = 30
            check_interval = 0.5  # Check every 500ms
            checks = int(collection_time / check_interval)
            
            logger.debug(f"Collecting captions for {collection_time}s...")
            
            for i in range(checks):
                try:
                    # Find caption segments using your exact selector
                    caption_elements = self.driver.find_elements(
                        By.CSS_SELECTOR, 
                        "span.ytp-caption-segment"
                    )
                    
                    # Extract text from visible caption segments
                    for element in caption_elements:
                        if element.is_displayed():
                            caption_text = element.text.strip()
                            if caption_text and caption_text not in seen_captions:
                                all_captions.append(caption_text)
                                seen_captions.add(caption_text)
                                logger.debug(f"Found caption: {caption_text[:50]}...")
                
                except Exception as e:
                    logger.debug(f"Error checking captions: {e}")
                
                time.sleep(check_interval)
            
            # Combine all collected captions
            full_caption_text = " ".join(all_captions)
            logger.info(f"✓ Collected {len(all_captions)} caption segments")
            return full_caption_text
            
        except Exception as e:
            logger.debug(f"Error in dynamic caption collection: {e}")
            return ""
    
    def play_video(self):
        """Start video playback."""
        try:
            # Try to click play button or start video
            play_selectors = [
                ".ytp-play-button",
                "button[aria-label*='Play' i]",
                ".ytp-large-play-button"
            ]
            
            for selector in play_selectors:
                try:
                    play_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if play_button.is_displayed():
                        play_button.click()
                        time.sleep(1)
                        logger.debug("✓ Video started playing")
                        return True
                except NoSuchElementException:
                    continue
            
            # Alternative: Click on video player to start
            try:
                video_player = self.driver.find_element(By.CSS_SELECTOR, "video, .html5-video-player")
                video_player.click()
                time.sleep(1)
                return True
            except NoSuchElementException:
                pass
                
        except Exception as e:
            logger.debug(f"Could not start video: {e}")
        return False
    
    def try_transcript_method(self):
        """Fallback method to try extracting transcript."""
        try:
            # Look for more actions menu
            more_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[aria-label*='More' i]")
            for button in more_buttons:
                try:
                    button.click()
                    time.sleep(1)
                    
                    # Look for transcript option
                    transcript_selectors = [
                        "//yt-formatted-string[contains(text(), 'Show transcript')]",
                        "//yt-formatted-string[contains(text(), 'Transcript')]",
                        "//span[contains(text(), 'transcript')]"
                    ]
                    
                    for selector in transcript_selectors:
                        try:
                            transcript_button = self.driver.find_element(By.XPATH, selector)
                            transcript_button.click()
                            time.sleep(2)
                            
                            # Extract transcript text
                            transcript_elements = self.driver.find_elements(
                                By.CSS_SELECTOR, 
                                ".ytd-transcript-segment-renderer, .transcript-segment"
                            )
                            if transcript_elements:
                                transcript_text = " ".join([el.text.strip() for el in transcript_elements])
                                logger.info("✓ Extracted transcript")
                                return transcript_text
                        except NoSuchElementException:
                            continue
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.debug(f"Transcript method failed: {e}")
        
        return ""
    
    def debug_page_elements(self, short_url):
        """Debug method to inspect page elements - useful for development."""
        try:
            self.driver.get(short_url)
            time.sleep(5)
            
            logger.info(f"=== DEBUG INFO FOR: {short_url} ===")
            
            # Check for title elements
            title_candidates = self.driver.find_elements(By.CSS_SELECTOR, "span[class*='attributed-string'], h1, [role='text']")
            logger.info(f"Found {len(title_candidates)} potential title elements:")
            for i, elem in enumerate(title_candidates[:5]):  # Show first 5
                try:
                    logger.info(f"  {i+1}. '{elem.text[:100]}' - Classes: {elem.get_attribute('class')}")
                except:
                    continue
            
            # Check for engagement elements
            engagement_candidates = self.driver.find_elements(By.CSS_SELECTOR, "button[aria-label], span[aria-label], [class*='view'], [class*='like'], [class*='comment']")
            logger.info(f"Found {len(engagement_candidates)} potential engagement elements:")
            for i, elem in enumerate(engagement_candidates[:10]):  # Show first 10
                try:
                    aria_label = elem.get_attribute('aria-label') or elem.text
                    if aria_label:
                        logger.info(f"  {i+1}. '{aria_label[:100]}' - Tag: {elem.tag_name}, Classes: {elem.get_attribute('class')}")
                except:
                    continue
            
            # Check for caption/player elements
            caption_candidates = self.driver.find_elements(By.CSS_SELECTOR, "[class*='caption'], [class*='player'], [class*='transcript'], [class*='subtitle']")
            logger.info(f"Found {len(caption_candidates)} potential caption elements:")
            for i, elem in enumerate(caption_candidates[:5]):
                try:
                    logger.info(f"  {i+1}. Classes: {elem.get_attribute('class')}, Displayed: {elem.is_displayed()}")
                except:
                    continue
            
            logger.info("=== END DEBUG INFO ===")
            
        except Exception as e:
            logger.error(f"Error in debug method: {e}")
    
    def infinite_scroll_discovery(self, max_scrolls=50, target_count=100):
        """Automatically scroll through Shorts feed to discover videos."""
        discovered_urls = []
        scroll_count = 0
        
        try:
            logger.info(f"Starting infinite scroll discovery (max {max_scrolls} scrolls, target {target_count} videos)")
            
            while scroll_count < max_scrolls and len(discovered_urls) < target_count:
                # Find all current Shorts on page
                current_shorts = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/shorts/']")
                
                # Extract URLs
                for element in current_shorts:
                    try:
                        href = element.get_attribute('href')
                        if href and href not in self.processed_urls and href not in discovered_urls:
                            discovered_urls.append(href)
                            
                    except StaleElementReferenceException:
                        continue
                
                # Scroll down to load more content
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(2, 4))  # Random delay to appear human
                
                scroll_count += 1
                
                if scroll_count % 10 == 0:
                    logger.info(f"Scroll {scroll_count}/{max_scrolls} - Found {len(discovered_urls)} unique Shorts")
                    
        except Exception as e:
            logger.error(f"Error during infinite scroll: {e}")
            
        logger.info(f"✓ Infinite scroll complete. Discovered {len(discovered_urls)} Shorts")
        return discovered_urls
    
    def extract_enhanced_data(self, short_url):
        """Extract comprehensive data from a YouTube Short."""
        try:
            logger.info(f"Extracting data from: {short_url}")
            self.driver.get(short_url)
            time.sleep(5)  # Give more time for page to load
            
            # Initialize data dictionary
            data = {
                'url': short_url,
                'title': '',
                'script': '',
                'likes': '',
                'comments': '',
                'views': '',
                'reposts': '',  # Added reposts field
                'channel': '',
                'duration': '',
                'hashtags': [],
                'description': '',
                'upload_date': '',
                'video_id': ''
            }
            
            # Extract video ID from URL
            video_id_match = re.search(r'/shorts/([a-zA-Z0-9_-]+)', short_url)
            data['video_id'] = video_id_match.group(1) if video_id_match else ''
            
            # Extract title with updated selectors
            title_selectors = [
                "span.yt-core-attributed-string.yt-core-attributed-string--white-space-pre-wrap",
                "span.yt-core-attributed-string",
                "h1.ytd-watch-metadata yt-formatted-string",
                "h1 yt-formatted-string",
                ".title.style-scope.ytd-video-primary-info-renderer",
                "#title h1"
            ]
            
            for selector in title_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    data['title'] = element.text.strip()
                    logger.debug(f"✓ Title found: {data['title'][:50]}...")
                    break
                except NoSuchElementException:
                    continue
            
            # Extract channel name
            try:
                channel_selectors = [
                    "#channel-name a",
                    ".ytd-channel-name a",
                    "#owner-text a"
                ]
                
                for selector in channel_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        data['channel'] = element.text.strip()
                        break
                    except NoSuchElementException:
                        continue
                        
            except Exception as e:
                logger.debug(f"Error extracting channel: {e}")
            
            # Extract likes, comments, reposts with your exact selectors
            engagement_data = self.extract_engagement_metrics()
            data['likes'] = engagement_data.get('likes', '')
            data['comments'] = engagement_data.get('comments', '')
            data['reposts'] = engagement_data.get('reposts', '')
            
            # Extract views with enhanced selectors
            try:
                data['views'] = self.extract_views()
            except Exception as e:
                logger.debug(f"Error extracting views: {e}")
                data['views'] = "views_not_found"
            
            # Extract comments count
            try:
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                
                comments_selectors = [
                    "#count .count-text",
                    ".count-text",
                    "#comments-count"
                ]
                
                for selector in comments_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        data['comments'] = element.text.strip()
                        break
                    except NoSuchElementException:
                        continue
                        
            except Exception as e:
                logger.debug(f"Error extracting comments: {e}")
            
            # Extract captions/transcript
            caption_attempts = 0
            max_caption_attempts = 3
            
            while caption_attempts < max_caption_attempts and not data['script']:
                try:
                    caption_text = self.extract_captions()
                    if caption_text and len(caption_text.strip()) > 20:  # Ensure meaningful content
                        data['script'] = caption_text.strip()
                        logger.info(f"✓ Extracted captions: {len(caption_text)} characters")
                        break
                    else:
                        caption_attempts += 1
                        if caption_attempts < max_caption_attempts:
                            logger.debug(f"Caption attempt {caption_attempts} yielded minimal content, retrying...")
                            time.sleep(1)
                except Exception as e:
                    caption_attempts += 1
                    logger.debug(f"Caption attempt {caption_attempts} failed: {e}")
                    if caption_attempts < max_caption_attempts:
                        time.sleep(1)
            
            # Extract description and hashtags
            try:
                description_selectors = [
                    "#description-text",
                    ".description", 
                    "#meta-contents #description",
                    ".content.style-scope.ytd-video-secondary-info-renderer"
                ]
                
                for selector in description_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        full_description = element.text.strip()
                        data['description'] = full_description
                        
                        # Extract hashtags from description
                        hashtags = re.findall(r'#\w+', full_description)
                        data['hashtags'] = hashtags
                        
                        # Use description as script if no transcript available
                        if not data['script']:
                            data['script'] = full_description
                        break
                        
                    except NoSuchElementException:
                        continue
                        
            except Exception as e:
                logger.debug(f"Error extracting description: {e}")
            
            # Fill empty fields with default values
            for key, value in data.items():
                if not value and key != 'hashtags':
                    data[key] = f"{key}_not_found"
                elif key == 'hashtags' and not value:
                    data[key] = []
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data from {short_url}: {e}")
            return None
    
    def bulk_scrape_discovered_shorts(self, shorts_urls, batch_size=10):
        """Scrape data from discovered Shorts in batches."""
        total_urls = len(shorts_urls)
        logger.info(f"Starting bulk scrape of {total_urls} Shorts in batches of {batch_size}")
        
        for i in range(0, total_urls, batch_size):
            batch = shorts_urls[i:i + batch_size]
            self.current_batch += 1
            
            logger.info(f"\n=== BATCH {self.current_batch} ({i+1}-{min(i+batch_size, total_urls)}/{total_urls}) ===")
            
            for j, url in enumerate(batch, 1):
                try:
                    logger.info(f"Scraping {j}/{len(batch)}: {url}")
                    
                    data = self.extract_enhanced_data(url)
                    if data:
                        self.shorts_data.append(data)
                        self.processed_urls.add(url)
                        logger.info(f"✓ Success: {data['title'][:50]}... (Channel: {data['channel']})")
                    else:
                        logger.warning(f"✗ Failed: {url}")
                    
                    # Random delay between requests
                    time.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    continue
            
            # Save progress after each batch
            self.save_progress(f"batch_{self.current_batch}_data.csv")
            
            # Longer delay between batches
            if i + batch_size < total_urls:
                delay = random.uniform(10, 20)
                logger.info(f"Batch complete. Waiting {delay:.1f}s before next batch...")
                time.sleep(delay)
    
    def auto_collect_shorts(self, collection_method="trending", target_count=100, search_terms=None):
        """Main method for automatic Shorts collection."""
        discovered_urls = []
        
        if collection_method == "trending":
            logger.info("Using trending discovery method")
            if self.discover_trending_shorts():
                discovered_urls = self.infinite_scroll_discovery(target_count=target_count)
                
        elif collection_method == "search" and search_terms:
            logger.info(f"Using search discovery method with terms: {search_terms}")
            discovered_urls = self.discover_shorts_by_search(search_terms)
            
        elif collection_method == "mixed":
            logger.info("Using mixed discovery method")
            # First get trending
            if self.discover_trending_shorts():
                trending_urls = self.infinite_scroll_discovery(target_count=target_count//2)
                discovered_urls.extend(trending_urls)
            
            # Then search if terms provided
            if search_terms:
                search_urls = self.discover_shorts_by_search(search_terms)
                discovered_urls.extend(search_urls)
        
        # Remove duplicates
        unique_urls = list(set(discovered_urls))
        logger.info(f"Discovered {len(unique_urls)} unique Shorts URLs")
        
        if unique_urls:
            # Limit to target count
            if len(unique_urls) > target_count:
                unique_urls = unique_urls[:target_count]
                
            self.bulk_scrape_discovered_shorts(unique_urls)
        else:
            logger.warning("No Shorts discovered. Check your internet connection or try different methods.")
    
    def save_progress(self, filename):
        """Save current progress to CSV."""
        if not self.shorts_data:
            return
            
        fieldnames = ['url', 'title', 'script', 'likes', 'comments', 'views', 'reposts', 'channel', 'duration', 'hashtags', 'description', 'upload_date', 'video_id']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in self.shorts_data:
                # Convert hashtags list to string
                row_copy = row.copy()
                row_copy['hashtags'] = ', '.join(row['hashtags']) if row['hashtags'] else ''
                writer.writerow(row_copy)
        
        logger.info(f"Progress saved to {filename} ({len(self.shorts_data)} records)")
    
    def save_final_dataset(self, filename="youtube_shorts_dataset.csv"):
        """Save final dataset with metadata."""
        if not self.shorts_data:
            logger.warning("No data to save.")
            return
        
        # Save main dataset
        self.save_progress(filename)
        
        # Save metadata
        metadata = {
            'total_videos': len(self.shorts_data),
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'unique_channels': len(set(item['channel'] for item in self.shorts_data)),
            'total_batches': self.current_batch
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Final dataset saved to {filename}")
        logger.info(f"✓ Metadata saved to {metadata_filename}")
        logger.info(f"✓ Total videos collected: {len(self.shorts_data)}")
        logger.info(f"✓ Unique channels: {metadata['unique_channels']}")
    
    def close(self):
        """Close the browser driver."""
        self.driver.quit()

def main():
    """Main function for auto-discovery scraping."""
    # Configuration for deep learning data collection
    TARGET_COUNT = 10  # Start small for testing the new selectors
    COLLECTION_METHOD = "search"  # Start with search to get more reliable results
    DEBUG_MODE = False  # Set to True to debug element detection
    
    # Search terms for discovery (focusing on simpler content first)
    SEARCH_TERMS = [
        "funny shorts", "viral", "trending", "cooking", "music"
    ]
    
    collector = AutoYouTubeShortsCollector(headless=True)
    
    try:
        if DEBUG_MODE:
            # Debug mode - test on a single video
            test_url = input("Enter a YouTube Shorts URL to debug: ").strip()
            if test_url:
                logger.info("Running in DEBUG mode...")
                collector.debug_page_elements(test_url)
                data = collector.extract_enhanced_data(test_url)
                if data:
                    logger.info(f"Extracted data: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return
        
        logger.info("🚀 Starting YouTube Shorts auto-collection for deep learning dataset")
        logger.info(f"📊 Target: {TARGET_COUNT} videos using '{COLLECTION_METHOD}' method")
        logger.info(f"🔍 Search terms: {SEARCH_TERMS}")
        logger.info("=" * 60)
        
        # Auto-collect Shorts
        collector.auto_collect_shorts(
            collection_method=COLLECTION_METHOD,
            target_count=TARGET_COUNT,
            search_terms=SEARCH_TERMS
        )
        
        # Save final dataset
        dataset_filename = f"youtube_shorts_dataset_{int(time.time())}.csv"
        collector.save_final_dataset(dataset_filename)
        
        logger.info("✅ Auto-collection complete!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Collection interrupted by user. Saving current progress...")
        collector.save_final_dataset("interrupted_dataset.csv")
        
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        collector.save_final_dataset("error_recovery_dataset.csv")
        
    finally:
        collector.close()

if __name__ == "__main__":
    main()