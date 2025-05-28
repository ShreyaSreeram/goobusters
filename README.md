# Goobusters: Free Fluid Tracking System in Trauma Ultrasounds

## Overview
Goobusters is a semi-automated pipeline for tracking free fluid in ultrasound videos using computer vision and machine learning techniques. The system employs optical flow tracking and multi-frame analysis to provide accurate tracking capabilities by propagating annotations throughout ultrasoun exams utilising a few initial expert annotations.

## Objective of the project
- To create densely annotated ultrasound examinations using optical flow algorithms. 
- To act as a training and validation dataset in the bigger picture goal of eventually creating a model that would detect and track free fluid in ultrasound exams.  

## Features
- Semi-automated fluid tracking in ultrasound videos
- Multi-frame tracking using optical flow
- Ground truth dataset creation and management
- Feedback loop system for continuous improvement
- Integration with MD.ai for annotation management
- Comprehensive debugging and visualisation tools

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
   git clone https://github.com/ShreyaSreeram/goobusters.git
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

3. **Optimised Learning**
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

- The system analyses the performance metrics (IoU and Dice scores) after each iteration
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
- [Feedback Loop Optimization Guide](docs/feedback_loop_optimization.md)

## Development

### Running Tests

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



## Authors


## Acknowledgments


### Ground Truth vs Feedback Loop Processing

#### Ground Truth Creation (`--create-ground-truth`)

Purpose:
- Creates a verified dataset of fluid annotations
- Establishes "source of truth" for algorithm evaluation
- Used for training and validation purposes

Processing Flow:
1. Loads expert-annotated frames from MD.ai
2. Processes each video frame-by-frame
3. For each frame:
   - Converts expert annotations to binary masks
   - Applies optical flow tracking between frames
   - Validates tracking results
   - Creates intermediate masks for non-annotated frames
4. Saves results:
   - Binary mask files for each frame
   - Metadata about processing
   - Uploads annotations back to MD.ai (if --upload is set)

Use Cases:
- Initial dataset creation
- Adding new examples to training set
- Validating algorithm performance
- Creating benchmarks for testing
- Establishing baseline performance metrics

Example Workflow:
```bash
# 1. Create initial ground truth for specific exam
python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186

# 2. Verify results and then upload to MD.ai
python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --upload

# 3. Create ground truth for multiple issue types
python src/consolidated_tracking.py --create-ground-truth --all-issues --ground-truth-videos 15
```

#### Feedback Loop (`--feedback-loop`)

Purpose:
- Iteratively improves algorithm performance
- Tests different parameter combinations
- Validates algorithm against ground truth
- Optimizes tracking parameters automatically

Processing Flow:
1. Initial Setup:
   - Loads ground truth dataset
   - Sets up initial parameters
   - Prepares evaluation metrics

2. For each iteration:
   - Runs fluid detection algorithm
   - Compares results with ground truth
   - Calculates performance metrics (IoU, Dice)
   - Adjusts parameters based on performance
   - Validates on independent test set
   
3. Final Output:
   - Performance metrics for each iteration
   - Optimized parameter sets
   - Visualization of improvements
   - Evaluation reports

Use Cases:
- Algorithm optimization
- Parameter tuning
- Performance validation
- Testing new tracking strategies
- Pre-deployment validation

Example Workflow:
```bash
# 1. Quick parameter optimization
python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 3

# 2. Thorough validation with genuine evaluation
python src/consolidated_tracking.py --feedback-loop --genuine-evaluation --sampling-rate 5

# 3. Production-ready optimization
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \
    --genuine-evaluation \
    --sampling-rate 10 \
    --iterations 10
```

#### Key Differences

| Aspect | Ground Truth Creation | Feedback Loop |
|--------|---------------------|---------------|
| Input | Expert annotations | Ground truth dataset |
| Output | Verified masks & annotations | Optimized parameters & metrics |
| Purpose | Dataset creation | Algorithm optimization |
| Processing | Single-pass | Iterative |
| Validation | Manual verification | Automated metrics |
| MD.ai Integration | Creates annotations | Uses existing annotations |
| Resource Usage | Linear with video length | Depends on iterations |
| Typical Duration | Longer per video | Shorter but multiple passes |

#### When to Use Which

Use Ground Truth Creation when:
- Starting a new project
- Adding new examples to dataset
- Creating validation sets
- Establishing benchmarks
- Needing verified annotations

Use Feedback Loop when:
- Optimizing algorithm parameters
- Testing performance improvements
- Validating changes
- Preparing for deployment
- Fine-tuning tracking behavior

#### Common Workflow Combining Both

1. Initial Setup:
   ```bash
   # Create initial ground truth dataset
   python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --upload
   ```

2. Parameter Optimization:
   ```bash
   # Run feedback loop with learning
   python src/consolidated_tracking.py --feedback-loop --learning-mode --exam-id 186
   ```

3. Validation:
   ```bash
   # Validate with genuine evaluation
   python src/consolidated_tracking.py --feedback-loop \
       --genuine-evaluation \
       --exam-id 186 \
       --sampling-rate 5
   ```

4. Expand Dataset:
   ```bash
   # Add more ground truth data
   python src/consolidated_tracking.py --create-ground-truth --all-issues
   ``` 