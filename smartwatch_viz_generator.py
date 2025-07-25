import pandas as pd
import openai
import anthropic
import os
import json
import re
from pathlib import Path
import time
import cairosvg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
TEMPERATURE = 0.7
NUM_QUERIES = 5

# Validate API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize API clients
openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Global log file
log_file = None

def log_message(message):
    """Write message to both console and log file"""
    print(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

def setup_folders():
    """Create output folders for organized file storage"""
    base_folder = 'outputs'
    openai_folder = os.path.join(base_folder, 'openai')
    claude_folder = os.path.join(base_folder, 'claude')
    
    Path(base_folder).mkdir(exist_ok=True)
    Path(openai_folder).mkdir(exist_ok=True)
    Path(claude_folder).mkdir(exist_ok=True)
    
    # Create global log file
    global log_file
    log_file = open(os.path.join(base_folder, 'Note.txt'), 'w', encoding='utf-8')
    
    return base_folder, openai_folder, claude_folder

def create_safe_folder_name(query_text, max_length=50):
    """Create a safe folder name from query text"""
    # Remove special characters and limit length
    safe_name = re.sub(r'[^\w\s-]', '', query_text).strip()
    safe_name = re.sub(r'\s+', '_', safe_name)
    
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name

def load_data():
    """Load query and fitness data"""
    log_message("Loading data...")
    
    # Load query data from Excel
    query_df = pd.read_excel('query_data_205.xlsx')
    log_message(f"Loaded {len(query_df)} queries")
    
    # Load fitness data from CSV
    fitness_df = pd.read_csv('workout_fitness_tracker_data_10000.csv')
    log_message(f"Loaded {len(fitness_df)} fitness records")
    
    return query_df, fitness_df

def select_fitness_data(fitness_df, strategy='diverse', query_idx=None):
    """
    Select fitness data based on different strategies
    
    Strategies:
    - 'random': Random selection
    - 'sequential': Sequential selection based on query index
    - 'diverse': Try to get diverse data (different workout types, ages, etc.)
    """
    if strategy == 'random':
        return fitness_df.sample(n=1, random_state=42+query_idx).iloc[0]  # Fixed seed for reproducibility
    elif strategy == 'sequential':
        idx = query_idx % len(fitness_df)
        return fitness_df.iloc[idx]
    elif strategy == 'diverse':
        # Try to get diverse workout types
        workout_types = fitness_df['Workout Type'].unique()
        selected_type = workout_types[query_idx % len(workout_types)]
        subset = fitness_df[fitness_df['Workout Type'] == selected_type]
        return subset.sample(n=1, random_state=42+query_idx).iloc[0]  # Fixed seed for reproducibility
    else:
        return fitness_df.sample(n=1, random_state=42+query_idx).iloc[0]

def create_prompt_template():
    """Create the prompt template for SVG generation"""
    return """You are a data visualization expert specializing in smartwatch health interfaces. You will be given a single row of personal health data and a user query:

User Query: {query}

Personal Health Data:
{health_data}

Your task is to:
1. Determine the most appropriate visualization mode for the insight, choosing from:
   - Donut chart
   - Bar chart  
   - Icon-based chart
   - Text-only with color/icon emphasis

2. Generate a single compact SVG suitable for a smartwatch screen (approximately 360x360 pixels). The SVG must be:
   - Visually clear and glanceable
   - Optimized for small screens
   - Aesthetically pleasing, with components contrasting the background
   - Self-contained (no external styles or dependencies)
   - Labeled with meaningful titles, units, or icons where appropriate
   - In the style of a visual that would be seen on a smartwatch

Focus on the user's perspective: What is most relevant for them to know at a glance right now?

Please respond with only the SVG code, starting with <svg and ending with </svg>."""

def format_health_data(row):
    """Format a fitness data row for the prompt"""
    return f"""Age: {row['Age']} years
Gender: {row['Gender']}
Height: {row['Height (cm)']} cm
Weight: {row['Weight (kg)']} kg
Workout Type: {row['Workout Type']}
Workout Duration: {row['Workout Duration (mins)']} minutes
Calories Burned: {row['Calories Burned']}
Heart Rate: {row['Heart Rate (bpm)']} bpm
Steps Taken: {row['Steps Taken']:,}
Distance: {row['Distance (km)']} km
Workout Intensity: {row['Workout Intensity']}
Sleep Hours: {row['Sleep Hours']} hours
Water Intake: {row['Water Intake (liters)']} liters
Daily Calories Intake: {row['Daily Calories Intake']:,}
Resting Heart Rate: {row['Resting Heart Rate (bpm)']} bpm
VO2 Max: {row['VO2 Max']}
Body Fat: {row['Body Fat (%)']}%
Mood Before Workout: {row['Mood Before Workout']}
Mood After Workout: {row['Mood After Workout']}"""

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 API"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log_message(f"OpenAI error: {e}")
        return None

def call_claude_opus4(prompt):
    """Call Claude Opus 4 API"""
    try:
        response = anthropic_client.messages.create(
            model="claude-opus-4-20250514",  # Using Opus 4
            max_tokens=2000,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        log_message(f"Claude error: {e}")
        return None

def extract_svg(response_text):
    """Extract SVG code from model response"""
    if not response_text:
        return None
    
    # Look for SVG tags
    svg_match = re.search(r'<svg.*?</svg>', response_text, re.DOTALL | re.IGNORECASE)
    if svg_match:
        return svg_match.group(0)
    
    # If no complete SVG found, return the full response
    return response_text

def svg_to_png(svg_content, output_path):
    """Convert SVG to PNG using cairosvg"""
    try:
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), 
                        write_to=output_path,
                        output_width=360,
                        output_height=360)
        return True
    except Exception as e:
        log_message(f"Error converting SVG to PNG: {e}")
        return False

def save_model_results(query_data, health_data, fitness_row, prompt, svg_content, 
                      response, model_folder):
    """Save SVG, PNG, and prompt for a single model"""
    if not svg_content:
        return False
    
    success_count = 0
    
    # Save SVG
    svg_path = os.path.join(model_folder, 'result.svg')
    try:
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        success_count += 1
    except Exception as e:
        log_message(f"Error saving SVG: {e}")
    
    # Save PNG
    png_path = os.path.join(model_folder, 'result.png')
    if svg_to_png(svg_content, png_path):
        success_count += 1
    
    # Save prompt with all context
    prompt_path = os.path.join(model_folder, 'prompt.txt')
    try:
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write("=== USER QUERY ===\n")
            f.write(f"Query: {query_data['query']}\n")
            f.write(f"Activity: {query_data['activity']}\n")
            f.write(f"Related to Activity: {query_data['relatedToActivity']}\n")
            if 'time' in query_data:
                f.write(f"Time: {query_data['time']}\n")
            f.write("\n=== SELECTED FITNESS DATA ===\n")
            f.write(f"User ID: {fitness_row['User ID']}\n")
            f.write(health_data + '\n\n')
            f.write("=== FULL PROMPT SENT TO MODEL ===\n")
            f.write(prompt + '\n\n')
            f.write("=== MODEL RESPONSE ===\n")
            f.write(response)
        success_count += 1
    except Exception as e:
        log_message(f"Error saving prompt: {e}")
    
    return success_count == 3

def main():
    """Main execution function"""
    log_message("Starting Smartwatch Visualization Generator")
    log_message("=" * 50)
    log_message(f"Configuration:")
    log_message(f"  Models: OpenAI GPT-4 & Claude Opus 4")
    log_message(f"  Temperature: {TEMPERATURE}")
    log_message(f"  Queries to process: {NUM_QUERIES}")
    
    # Setup
    base_folder, openai_folder, claude_folder = setup_folders()
    
    # Load data
    query_df, fitness_df = load_data()
    
    # Get prompt template
    prompt_template = create_prompt_template()
    
    # Process first NUM_QUERIES queries
    queries_to_process = query_df.head(NUM_QUERIES)
    
    log_message(f"\nProcessing {len(queries_to_process)} queries...")
    log_message("Data selection strategy: 'diverse' (different workout types)")
    log_message("=" * 50)
    
    results = {
        'successful_openai': 0,
        'successful_claude': 0,
        'failed_openai': 0,
        'failed_claude': 0,
        'processed_queries': []
    }
    
    for idx, query_row in queries_to_process.iterrows():
        log_message(f"\nProcessing query {idx + 1}/{len(queries_to_process)}")
        log_message(f"Query: {query_row['query']}")
        log_message(f"Activity: {query_row['activity']}")
        
        # Create query-specific folder name
        safe_query_name = create_safe_folder_name(query_row['query'])
        query_folder_name = f"{idx+1:03d}_{safe_query_name}"
        
        # Create folders for both models with same query
        openai_query_folder = os.path.join(openai_folder, query_folder_name)
        claude_query_folder = os.path.join(claude_folder, query_folder_name)
        Path(openai_query_folder).mkdir(exist_ok=True)
        Path(claude_query_folder).mkdir(exist_ok=True)
        
        # IMPORTANT: Select fitness data ONCE and use for BOTH models
        fitness_row = select_fitness_data(fitness_df, strategy='diverse', query_idx=idx)
        health_data = format_health_data(fitness_row)
        
        log_message(f"Selected fitness data: User {fitness_row['User ID']} - {fitness_row['Workout Type']} (SAME for both models)")
        
        # Create full prompt (SAME for both models)
        full_prompt = prompt_template.format(
            query=query_row['query'],
            health_data=health_data
        )
        
        query_result = {
            'query_id': idx + 1,
            'query_text': query_row['query'],
            'folder_name': query_folder_name,
            'fitness_user_id': int(fitness_row['User ID']),
            'workout_type': fitness_row['Workout Type']
        }
        
        # Call OpenAI GPT-4
        log_message("  Calling OpenAI GPT-4...")
        openai_response = call_openai_gpt4(full_prompt)
        openai_svg = extract_svg(openai_response)
        
        if save_model_results(query_row, health_data, fitness_row, full_prompt, 
                             openai_svg, openai_response, openai_query_folder):
            results['successful_openai'] += 1
            query_result['openai_success'] = True
            log_message("  ✓ OpenAI files saved (SVG, PNG, prompt)")
        else:
            results['failed_openai'] += 1
            query_result['openai_success'] = False
            log_message("  ✗ OpenAI generation failed")
        
        # Small delay between API calls
        time.sleep(1)
        
        # Call Claude Opus 4 (with SAME prompt and data)
        log_message("  Calling Claude Opus 4...")
        claude_response = call_claude_opus4(full_prompt)
        claude_svg = extract_svg(claude_response)
        
        if save_model_results(query_row, health_data, fitness_row, full_prompt, 
                             claude_svg, claude_response, claude_query_folder):
            results['successful_claude'] += 1
            query_result['claude_success'] = True
            log_message("  ✓ Claude files saved (SVG, PNG, prompt)")
        else:
            results['failed_claude'] += 1
            query_result['claude_success'] = False
            log_message("  ✗ Claude generation failed")
        
        results['processed_queries'].append(query_result)
        
        # Small delay between queries
        time.sleep(1)
    
    # Print results summary
    log_message("\n" + "=" * 50)
    log_message("RESULTS SUMMARY")
    log_message("=" * 50)
    log_message(f"OpenAI GPT-4: {results['successful_openai']} successful, {results['failed_openai']} failed")
    log_message(f"Claude Opus 4: {results['successful_claude']} successful, {results['failed_claude']} failed")
    log_message(f"\nGenerated folder structure:")
    log_message(f"  outputs/")
    log_message(f"    openai/")
    log_message(f"    claude/")
    for query_result in results['processed_queries']:
        log_message(f"      {query_result['folder_name']}/")
    
    # Save results summary
    final_log_path = os.path.join(base_folder, 'generation_summary.json')
    with open(final_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'num_queries': NUM_QUERIES,
                'temperature': TEMPERATURE,
                'data_selection_strategy': 'diverse',
                'models': ['openai-gpt4', 'claude-opus4']
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    log_message(f"\nGeneration summary saved to: {final_log_path}")
    log_message("All logs saved to: outputs/Note.txt")
    log_message("Done!")
    
    # Close log file
    if log_file:
        log_file.close()

if __name__ == "__main__":
    main()