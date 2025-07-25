# URV-glanceablevis

Generate and compare smartwatch health visualizations using OpenAI GPT-4 and Claude Opus 4.

## Setup

1. **Install dependencies**
   ```bash
   pip install pandas openai anthropic openpyxl cairosvg python-dotenv
   ```
   - If you have 'OSError: no library called "cairo-2" was found' issue, please try ```conda install cairo pango```

2. **Configure API keys**
   Create `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

3. **Add data files**
   - `query_data_205.xlsx` - User queries
   - `workout_fitness_tracker_data_10000.csv` - Fitness data

## Usage

```bash
python smartwatch_viz_generator.py
```

- Processes 30 queries by default
- Each query uses the same fitness data for both models
- Temperature: 0.7

## Output Structure

```
outputs/
├── Note.txt                           # Execution logs
├── generation_summary.json            # Results summary
├── openai/
│   ├── 001_how_much_dancing_do_i_need/
│   │   ├── result.svg                 # Generated SVG
│   │   ├── result.png                 # PNG conversion
│   │   └── prompt.txt                 # Full prompt + context
│   └── 002_summary_of_my_sleep_cycles/
│       └── ...
└── claude/
    ├── 001_how_much_dancing_do_i_need/
    │   ├── result.svg
    │   ├── result.png
    │   └── prompt.txt
    └── ...
```
