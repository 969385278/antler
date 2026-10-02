"""
Generate additional character frameworks for extending the character dataset.

The paper uses a single character (Elias Zhang) whose framework and stories are
already included. This script is provided for users who want to generate more
characters following the same pipeline. It uses the Elias Zhang framework as a
template and generates new character life timelines via the Gemini API.

This is NOT required to reproduce the paper's experiments -- it is an optional
extension tool.
"""

import json
import requests
import time
from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def gemini_api_call(prompt):
    """Makes a call to the Gemini API and returns the response text and usage metadata."""
    url = "https://go.apis.huit.harvard.edu/ais-google-gemini/v1beta/models/gemini-2.5-pro-preview-05-06:generateContent"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 100000,
        }
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': os.getenv("GEMINI_API_KEY", "")
    }
    
    max_tries = 5
    tries = 0
    while tries < max_tries:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API call attempt {tries + 1}/{max_tries}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt length: {len(prompt)} characters")
            
            start_time = time.time()
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            end_time = time.time()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request completed in {end_time - start_time:.2f} seconds")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status code: {response.status_code}")
            
            response.raise_for_status()
            response_dict = json.loads(response.text)
            
            if "candidates" in response_dict and response_dict["candidates"]:
                content = response_dict["candidates"][0].get("content", {})
                if "parts" in content and content["parts"]:
                    text = content["parts"][0].get("text", "")
                    usage = response_dict.get("usageMetadata", {})
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ API call successful! Response: {len(text)} chars")
                    return text, usage
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API response missing expected content")
            print(f"Response preview: {response.text[:200]}...")
            
        except requests.exceptions.Timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call timed out (attempt {tries + 1})")
        except requests.exceptions.HTTPError as http_err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ HTTP error: {http_err}")
            if 'response' in locals():
                print(f"Response: {response.text[:300]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response: {response.text[:300]}")
        
        tries += 1
        if tries < max_tries:
            wait_time = 3 * tries
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ All {max_tries} attempts failed")
    return None, {}

def load_elias_framework():
    """Load the Elias framework to use as a template."""
    framework_path = os.path.join(SCRIPT_DIR, '..', 'character_profiles', 'elias_zhang_framework.txt')
    try:
        with open(framework_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading Elias framework: {e}")
        return None

def generate_character_framework(character_concept, elias_template):
    """Generate a character framework using the Gemini API."""
    
    prompt = f"""
Based on the following example character framework for Elias Zhang, create a detailed, chronologically consistent life timeline framework for a new diverse character with the following concept:

{character_concept}

ELIAS ZHANG FRAMEWORK EXAMPLE:
{elias_template}

REQUIREMENTS:
1. Follow the exact same structure and format as the Elias framework
2. Cover a 50-year span (any 50-year period is fine, doesn't have to be ages 5-55) with authentic historical context for the chosen time period and location
3. Include specific historical events, cultural movements, and technological developments for each era
4. Create a compelling personal narrative that intersects with major historical moments
5. Include family context, education/career progression, personal relationships, and major life events
6. Ensure chronological consistency and authentic cultural details
7. Each year/age should have: Location, Family Context, Education/Career, Personal Life, Major Historical/Cultural Events, Technology, Pop Culture, Social Norms/Issues, and Potential Episodic Memory Themes

Make this character framework as detailed and authentic as the Elias example, with rich historical context and personal development across the 50-year timeline.
"""

    print(f"Generating framework for: {character_concept['name']}")
    response, usage = gemini_api_call(prompt)
    
    if response:
        return response
    else:
        print(f"Failed to generate framework for {character_concept['name']}")
        return None

def save_framework(character_name, framework_text):
    """Save the generated framework to a file in the same directory as Elias framework."""
    # Use the same directory as the Elias framework
    output_dir = os.path.join(SCRIPT_DIR, '..', 'character_profiles')
    
    # Generate filename
    filename = f"{character_name.lower().replace(' ', '_')}_framework.txt"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            f.write(framework_text)
        print(f"✓ Saved framework to: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error saving framework: {e}")
        return None

def main():
    """Generate 5 diverse character frameworks."""
    
    print("="*80)
    print("CHARACTER FRAMEWORK GENERATION")
    print("="*80)
    
    # Load Elias template
    elias_template = load_elias_framework()
    if not elias_template:
        print("Could not load Elias framework template. Exiting.")
        return
    
    print(f"Loaded Elias template: {len(elias_template)} characters")
    
    # Define 5 diverse character concepts with flexible age spans
    characters = [
        {
            "name": "Kenji Nakamura",
            "description": "Japanese man born in 1920, experiencing Imperial Japan, WWII, American occupation, and Japan's economic miracle (50-year span: 1925-1975)"
        },
        {
            "name": "Indira Sharma", 
            "description": "Indian woman born in 1920, living through British colonial period, independence struggle, partition, and early independent India (50-year span: 1925-1975)"
        },
        {
            "name": "Ahmed Al-Mansouri",
            "description": "Emirati man born in 1955, experiencing traditional desert life, oil discovery, rapid modernization, and UAE formation (50-year span: 1960-2010)"
        },
        {
            "name": "Elena Popović",
            "description": "Yugoslav/Serbian woman born in 1935, living through WWII, Tito's Yugoslavia, economic development, and the beginning of Yugoslav wars (50-year span: 1940-1990)"
        },
        {
            "name": "Kwame Asante",
            "description": "Ghanaian man born in 1945, experiencing late colonial period, independence, Nkrumah era, military coups, and economic adjustments (50-year span: 1950-2000)"
        }
    ]
    
    total_cost = 0.0
    successful_generations = 0
    
    # Generate each character framework independently
    for i, character in enumerate(characters, 1):
        print(f"\n{'='*60}")
        print(f"GENERATING CHARACTER {i}/5: {character['name']}")
        print(f"{'='*60}")
        print(f"Concept: {character['description']}")
        
        framework = generate_character_framework(character, elias_template)
        
        if framework:
            filepath = save_framework(character['name'], framework)
            if filepath:
                successful_generations += 1
                print(f"✓ Successfully generated and saved {character['name']}")
            else:
                print(f"✗ Generated but failed to save {character['name']}")
        else:
            print(f"✗ Failed to generate {character['name']}")
        
        # Wait between generations to avoid rate limiting
        if i < len(characters):
            print("Waiting 5 seconds before next generation...")
            time.sleep(5)
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully generated: {successful_generations}/{len(characters)} frameworks")
    print(f"Output directory: {os.path.join(SCRIPT_DIR, '..', 'character_profiles')}")

if __name__ == "__main__":
    main() 