# Goobusters: Ultrasound Fluid Detection System

## Overview
Goobusters is an advanced system for detecting and tracking fluid in ultrasound videos using computer vision and machine learning techniques. The system employs optical flow tracking and multi-frame analysis to provide accurate fluid detection and tracking capabilities.

## Features
- Automated fluid detection in ultrasound videos
- Multi-frame tracking using optical flow
- Ground truth dataset creation and management
- Feedback loop system for continuous improvement
- Integration with MD.ai for annotation management
- Comprehensive debugging and visualization tools

## Setup

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Pandas
- MD.ai Python client

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/goobusters.git
   cd goobusters
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Environment Variables
Required environment variables:
- `DATA_DIR`: Path to data directory
- `LABEL_ID_FREE_FLUID`: MD.ai label ID for free fluid
- `LABEL_ID_NO_FLUID`: MD.ai label ID for no fluid
- `LABEL_ID_MACHINE_GROUP`: MD.ai label ID for machine annotations

## Usage

### Ground Truth Creation
Create ground truth datasets for training and evaluation:
```bash
python src/consolidated_tracking.py --create-ground-truth
```

For detailed options and usage, see [Ground Truth Documentation](docs/ground_truth_feedback_loop.md)

### Feedback Loop
Run the feedback loop for continuous improvement:
```bash
python src/consolidated_tracking.py --feedback-loop
```

### Common Use Cases
1. Process single exam:
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186
   ```

2. Run feedback loop with learning:
   ```bash
   python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 5
   ```

## Command Line Options

### Core Functionality Flags
```bash
--create-ground-truth        # Create ground truth dataset
--feedback-loop             # Run ground truth feedback loop
--debug                     # Enable debug mode
--upload                    # Upload results to MD.ai
--no-upload                 # Skip uploading annotations to MD.ai
```

### Ground Truth Creation Options
```bash
# Process specific exam/study/series
--ground-truth-single-exam EXAM      # Create ground truth for single exam number
--ground-truth-single-study STUDY    # Create ground truth for single StudyInstanceUID
--ground-truth-single-series SERIES  # Create ground truth for single SeriesInstanceUID

# Other ground truth options
--ground-truth-videos N              # Number of videos per issue type (default: 15)
--all-issues                        # Process all issue types
```

### Feedback Loop Options
```bash
--iterations N              # Number of feedback loop iterations (default: 3)
--exam-id EXAM             # Run feedback loop on specific exam ID
--learning-mode            # Enable learning mode in feedback loop
--params-file FILE         # Path to parameters JSON file
--genuine-evaluation       # Use genuine evaluation with sparse annotations
--sampling-rate N          # Sampling rate for sparse annotations (default: 10)
```

### Video Selection Options
```bash
--study STUDY_UID          # Process specific StudyInstanceUID
--series SERIES_UID        # Process specific SeriesInstanceUID
--issue TYPE              # Process specific issue type
                         # Choices: disappear_reappear, branching_fluid, 
                         #          multiple_distinct, no_fluid
```

### Directory and Path Options
```bash
--images-dir PATH          # Specific path to the images directory
```

### Example Commands

1. Create ground truth for a single exam:
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --upload
   ```

2. Run feedback loop with learning mode:
   ```bash
   python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 5 --exam-id 64
   ```

3. Process specific study/series:
   ```bash
   python src/consolidated_tracking.py --study "1.2.826..." --series "1.2.826..." --upload
   ```

4. Create ground truth for specific issue type:
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --issue multiple_distinct --ground-truth-videos 10
   ```

5. Run feedback loop with custom parameters:
   ```bash
   python src/consolidated_tracking.py --feedback-loop --params-file params.json --genuine-evaluation --sampling-rate 5
   ```

### Common Option Combinations

1. **Quick Testing Setup**
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --debug --no-upload
   ```

2. **Full Production Run**
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --all-issues --upload
   ```

3. **Optimized Learning**
   ```bash
   python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 10 --genuine-evaluation --sampling-rate 20
   ```

4. **Single Video Analysis**
   ```bash
   python src/consolidated_tracking.py --study STUDY_UID --series SERIES_UID --debug
   ```

### Advanced Options Explained

#### Learning Mode (`--learning-mode`)
Learning mode enables the feedback loop to automatically adjust tracking parameters based on performance metrics. When enabled:

- The system analyzes the performance metrics (IoU and Dice scores) after each iteration
- Parameters are automatically adjusted based on performance trends
- Adjustable parameters include:
  - Flow quality threshold
  - Tracking strategy weights
  - Confidence thresholds
  - Motion detection sensitivity

Example with learning mode:
```bash
python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 5 --exam-id 64
```

You can also provide initial parameters:
```bash
python src/consolidated_tracking.py --feedback-loop --learning-mode --params-file initial_params.json
```

#### Sampling Rate (`--sampling-rate`)
The sampling rate determines how many frames are used in the evaluation process. A sampling rate of N means the system will process every Nth frame.

- Default: 10 (processes every 10th frame)
- Lower values (e.g., 5) provide more detailed evaluation but take longer
- Higher values (e.g., 20) are faster but might miss some details
- Minimum recommended: 5
- Maximum recommended: 30

Examples of different sampling rates:
```bash
# Detailed evaluation (every 5th frame)
python src/consolidated_tracking.py --feedback-loop --sampling-rate 5

# Balanced evaluation (every 10th frame)
python src/consolidated_tracking.py --feedback-loop --sampling-rate 10

# Quick evaluation (every 20th frame)
python src/consolidated_tracking.py --feedback-loop --sampling-rate 20
```

#### Genuine Evaluation (`--genuine-evaluation`)
Genuine evaluation mode uses a more rigorous evaluation approach that better reflects real-world performance:

- Creates sparse ground truth annotations for validation
- Uses independent frame sets for training and evaluation
- Implements cross-validation techniques
- Provides more reliable performance metrics

Features:
- Prevents overfitting to specific frame patterns
- Gives more realistic performance estimates
- Better for production deployment decisions
- More computationally intensive

Example combinations:
```bash
# Full evaluation setup
python src/consolidated_tracking.py --feedback-loop \
    --genuine-evaluation \
    --sampling-rate 10 \
    --learning-mode \
    --iterations 5

# Quick evaluation for testing
python src/consolidated_tracking.py --feedback-loop \
    --genuine-evaluation \
    --sampling-rate 20 \
    --iterations 2
```

#### Recommended Combinations

1. **Development Testing**
   ```bash
   python src/consolidated_tracking.py --feedback-loop \
       --learning-mode \
       --sampling-rate 20 \
       --iterations 3
   ```
   Best for: Quick iterations during development

2. **Production Validation**
   ```bash
   python src/consolidated_tracking.py --feedback-loop \
       --genuine-evaluation \
       --sampling-rate 5 \
       --learning-mode \
       --iterations 10
   ```
   Best for: Final validation before deployment

3. **Balanced Approach**
   ```bash
   python src/consolidated_tracking.py --feedback-loop \
       --genuine-evaluation \
       --sampling-rate 10 \
       --learning-mode \
       --iterations 5
   ```
   Best for: Regular testing and validation

#### Performance Impact

| Option | Processing Time | Accuracy | Memory Usage |
|--------|----------------|----------|--------------|
| Learning Mode | ↑↑ | ↑↑ | ↑ |
| Sampling Rate 5 | ↑↑↑ | ↑↑↑ | ↑ |
| Sampling Rate 20 | ↓ | ↓ | ↓ |
| Genuine Evaluation | ↑↑ | ↑↑ | ↑↑ |

## Project Structure
```
goobusters/
├── src/
│   ├── consolidated_tracking.py    # Main tracking system
│   ├── multi_frame_tracking/       # Multi-frame tracking components
│   ├── utils/                      # Utility functions
│   └── visualization/              # Visualization tools
├── docs/                           # Documentation
├── data/                          # Data directory (gitignored)
└── tests/                         # Test suite
```

## Documentation
- [Ground Truth and Feedback Loop](docs/ground_truth_feedback_loop.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
This project follows PEP 8 guidelines. Run linting with:
```bash
flake8 src/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Insert License Information]

## Authors
[Your Name/Team]

## Acknowledgments
- [List any acknowledgments, libraries, or tools used] 