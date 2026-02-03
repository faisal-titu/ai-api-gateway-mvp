import asyncio
import aiohttp
import random
import time
import glob
import os
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"  # Your FastAPI endpoint
TEST_DURATION_SECONDS = 900  # 15 minutes
IMAGE_DIR = "/home/iot/Desktop/ai_search_fastapi/ai-api-gateway-mvp/datalake/dataset/unsplash"  # Directory with test images
INDEX_NAME = "unsplash_batch_indexing_1"  # Your index name

# Test data
TEXT_QUERIES = [
    "a person walking on the beach",
    "sunset over mountains",
    "dog playing in park",
    "modern city skyline",
    "fruit on a plate",
    "children playing outside",
    "people working in an office",
    "flowers in a garden",
    "cars on a highway",
    "birds flying in the sky"
]

# Stats counters
stats = {
    "image_search_count": 0,
    "text_search_count": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "parallel_batches": 0,
    "sequential_batches": 0
}

# Get a list of test images
def get_test_images():
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(f"{IMAGE_DIR}/*.{ext}"))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}. Please check the directory path.")
        return []
    
    print(f"Found {len(image_files)} test images")
    return image_files

async def set_api_settings():
    """Set the index name and image directory for the API"""
    print("Setting API settings...")
    async with aiohttp.ClientSession() as session:
        settings_data = {
            "index_name": INDEX_NAME,
            "image_dir": IMAGE_DIR
        }
        
        try:
            async with session.post(f"{API_URL}/set-settings", json=settings_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Settings applied: {result}")
                    return True
                else:
                    print(f"Failed to set settings: {response.status}")
                    return False
        except Exception as e:
            print(f"Error setting API settings: {e}")
            return False

async def image_search_request(image_path, session, num_images=5):
    """Perform an image search request"""
    start_time = time.time()
    status = "failed"
    
    try:
        with open(image_path, 'rb') as f:
            form_data = aiohttp.FormData()
            form_data.add_field('file', f, filename=os.path.basename(image_path))
            form_data.add_field('num_images', str(num_images))
            
            async with session.post(f"{API_URL}/images/search", data=form_data) as response:
                elapsed = time.time() - start_time
                result = await response.text()
                
                if response.status == 200:
                    stats["successful_requests"] += 1
                    stats["image_search_count"] += 1
                    status = "success"
                else:
                    stats["failed_requests"] += 1
                    
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Image search: {status} - {elapsed:.2f}s - {os.path.basename(image_path)}")
                return response.status, elapsed, result
                
    except Exception as e:
        elapsed = time.time() - start_time
        stats["failed_requests"] += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Image search error: {str(e)[:100]}... - {elapsed:.2f}s")
        return 500, elapsed, str(e)

async def text_search_request(query, session, num_images=5):
    """Perform a text search request"""
    start_time = time.time()
    status = "failed"
    
    try:
        data = {
            "query": query,
            "num_images": num_images
        }
        
        async with session.post(f"{API_URL}/texts/search", json=data) as response:
            elapsed = time.time() - start_time
            result = await response.text()
            
            if response.status == 200:
                stats["successful_requests"] += 1
                stats["text_search_count"] += 1
                status = "success"
            else:
                stats["failed_requests"] += 1
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Text search: {status} - {elapsed:.2f}s - {query[:30]}...")
            return response.status, elapsed, result
            
    except Exception as e:
        elapsed = time.time() - start_time
        stats["failed_requests"] += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Text search error: {str(e)[:100]}... - {elapsed:.2f}s")
        return 500, elapsed, str(e)

async def parallel_requests(image_files, session):
    """Execute multiple requests in parallel"""
    batch_size = random.randint(2, 5)
    tasks = []
    
    # Add random mix of image and text searches
    for _ in range(batch_size):
        if random.random() < 0.6:  # 60% chance of image search
            image_path = random.choice(image_files)
            tasks.append(image_search_request(image_path, session))
        else:
            query = random.choice(TEXT_QUERIES)
            tasks.append(text_search_request(query, session))
    
    stats["parallel_batches"] += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel batch with {len(tasks)} requests")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def sequential_requests(image_files, session):
    """Execute requests one after another"""
    batch_size = random.randint(2, 5)
    results = []
    
    stats["sequential_batches"] += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting sequential batch with {batch_size} requests")
    
    for _ in range(batch_size):
        if random.random() < 0.6:  # 60% chance of image search
            image_path = random.choice(image_files)
            result = await image_search_request(image_path, session)
        else:
            query = random.choice(TEXT_QUERIES)
            result = await text_search_request(query, session)
        results.append(result)
    
    return results

async def run_tests():
    """Run the tests for the specified duration"""
    image_files = get_test_images()
    if not image_files:
        print("No test images available. Exiting...")
        return
    
    # First set the API settings
    settings_success = await set_api_settings()
    if not settings_success:
        print("Failed to initialize API settings. Continuing anyway...")
    
    start_time = time.time()
    end_time = start_time + TEST_DURATION_SECONDS
    
    print(f"Starting load test at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Test will run for {TEST_DURATION_SECONDS/60:.1f} minutes")
    
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            try:
                # Switch between parallel and sequential requests
                if random.random() < 0.7:  # 70% parallel, 30% sequential
                    await parallel_requests(image_files, session)
                else:
                    await sequential_requests(image_files, session)
                
                # Random pause between request batches (0.5 - 2 seconds)
                delay = random.uniform(0.5, 2.0)
                await asyncio.sleep(delay)
                
                # Print progress every ~30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0:
                    remaining = TEST_DURATION_SECONDS - elapsed
                    print(f"Test progress: {elapsed/TEST_DURATION_SECONDS*100:.1f}% complete, {remaining/60:.1f} minutes left")
                
            except Exception as e:
                print(f"Error during test execution: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    # Print final statistics
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Test completed after {total_time/60:.1f} minutes")
    print(f"Total requests: {stats['successful_requests'] + stats['failed_requests']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    print(f"Image searches: {stats['image_search_count']}")
    print(f"Text searches: {stats['text_search_count']}")
    print(f"Parallel batches: {stats['parallel_batches']}")
    print(f"Sequential batches: {stats['sequential_batches']}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_tests())