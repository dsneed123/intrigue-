#!/usr/bin/env python3
"""
Quick test script to check if YouTube transcript API is working
"""

def test_transcript_api():
    print("🧪 Testing YouTube Transcript API...")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("✅ youtube-transcript-api imported successfully")
        
        # Test with a known video that should have transcripts
        test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
        
        print(f"\n🎬 Testing with video ID: {test_video_id}")
        
        # Method 1: Direct get_transcript
        try:
            transcript = YouTubeTranscriptApi.get_transcript(test_video_id)
            print("✅ Method 1 (get_transcript) works!")
            print(f"   Transcript length: {len(transcript)} segments")
            if transcript:
                print(f"   First segment: {transcript[0]}")
            method1_works = True
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
            method1_works = False
        
        # Method 2: list_transcripts (if available)
        try:
            if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                transcript_list = YouTubeTranscriptApi.list_transcripts(test_video_id)
                print("✅ Method 2 (list_transcripts) works!")
                
                available_transcripts = list(transcript_list)
                print(f"   Available transcripts: {len(available_transcripts)}")
                
                if available_transcripts:
                    first_transcript = available_transcripts[0]
                    print(f"   First available: {getattr(first_transcript, 'language_code', 'unknown')}")
                method2_works = True
            else:
                print("⚠️  Method 2 (list_transcripts) not available in this version")
                method2_works = False
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
            method2_works = False
        
        # Summary
        print(f"\n📊 Test Results:")
        print(f"   Direct get_transcript: {'✅ Works' if method1_works else '❌ Failed'}")
        print(f"   List transcripts: {'✅ Works' if method2_works else '❌ Failed'}")
        
        if method1_works or method2_works:
            print("🎉 At least one method works! Your scraper should be able to get transcripts.")
        else:
            print("🚨 No methods work. Check your youtube-transcript-api installation.")
            print("   Try: pip install --upgrade youtube-transcript-api")
        
        return method1_works or method2_works
        
    except ImportError as e:
        print(f"❌ Failed to import youtube-transcript-api: {e}")
        print("💡 Install with: pip install youtube-transcript-api")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_with_custom_video():
    """Test with a user-provided video URL"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        
        url = input("\n🎬 Enter a YouTube URL to test (or press Enter to skip): ").strip()
        if not url:
            return
        
        # Extract video ID
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/shorts\/([^&\n?#]+)',
        ]
        
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break
        
        if not video_id:
            print("❌ Could not extract video ID from URL")
            return
        
        print(f"🔍 Testing video ID: {video_id}")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            print(f"✅ Success! Found transcript with {len(transcript)} segments")
            
            # Show first few words
            if transcript:
                first_text = transcript[0].get('text', '')
                print(f"   First text: '{first_text[:100]}...'")
            
        except Exception as e:
            print(f"❌ Failed to get transcript: {e}")
            
            # Common reasons
            if "disabled" in str(e).lower():
                print("💡 Reason: Transcript disabled by video owner")
            elif "not found" in str(e).lower():
                print("💡 Reason: No transcript available for this video")
            elif "private" in str(e).lower():
                print("💡 Reason: Video is private")
            else:
                print("💡 This might be a temporary issue or the video doesn't have transcripts")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 YouTube Transcript API Tester")
    print("=" * 40)
    
    # Basic API test
    works = test_transcript_api()
    
    if works:
        # Optional custom video test
        test_with_custom_video()
        
        print("\n🎯 Recommendation:")
        print("   Your transcript API should work with the scraper!")
        print("   If you still get errors, they're likely video-specific (no transcripts available)")
    else:
        print("\n🔧 Next Steps:")
        print("   1. pip install --upgrade youtube-transcript-api")
        print("   2. pip install --upgrade pip")
        print("   3. Run this test again")
    
    print("\n✅ Test complete!")