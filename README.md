# TokenLense: Advanced Token Attribution Analysis for LLMs

TokenLense is a comprehensive framework for analyzing, visualizing, and interpreting token relationships and attribution metrics in Large Language Model outputs. This project provides deep insights into how LLMs generate text through gradient-based attribution analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-blue.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15+-blue.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

🔗 **Documentation**: [Extractor](extractor/ReadMe.md) | [Visualizer](visualizer/README.md) | [Analyzer](analyzer/README.md) | [Custom Datasets](extractor/docs/custom_dataset_guide.md)

## Overview

TokenLense consists of three main components:

1. **Extractor**: Runs attribution analysis on language models and generates data
2. **Analyzer**: API service that processes and serves attribution data
3. **Visualizer**: Web interface for exploring and visualizing token relationships

## Features

- **Token Attribution Analysis**: Track how input tokens influence output tokens
- **Multiple Attribution Methods**: Support for various attribution techniques:
  - Attention weights
  - Input × Gradient
  - Integrated Gradients
  - Layer Gradient × Activation
  - LIME
  - Saliency maps
  - [...](https://inseq.org/en/v0.6.0/main_classes/feature_attribution.html#inseq.attr.FeatureAttribution)
- **Interactive Visualization**: Explore token relationships through intuitive UI
- **Performance Analysis**: Compare attribution methods across models and hardware
- **Comparison View**: Side-by-side analysis of different models and methods
- **Docker Support**: Easy deployment with containerization

## Getting Started

### Prerequisites

- Docker and Docker Compose (for containerized setup)
- Python 3.10+ (for local development)
- Node.js 18+ (for local frontend development)

### Quick Start with Docker (Recommended)

The easiest way to run TokenLense is using Docker Compose:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TokenLense.git
   cd TokenLense
   ```

2. Make sure you have attribution data in the `analyzer/data` directory. If not, either:
   - Run the extractor to generate data (see [Extractor Usage](#extractor-usage))
   - download [sample data](https://tumde-my.sharepoint.com/:f:/g/personal/ruben_kaiser_tum_de/Et5_y-L6FJJPkaswFl-VfYkB_t_C2xPPhsJzvfufynZtOQ?e=peIRzA) (only available for [TUM](https://tum.de) members)

   Notice that the data must be organized in the following structure:
   ```
   analyzer/data/
   ├── MODEL_NAME/
   │   ├── method_ATTRIBUTION_METHOD/
   │   │   └── data/
   │   │       ├── <attribution_data>_inseq.json
   │   │       └── ...
   ```

   where `MODEL_NAME` is the name of the model (e.g., `gpt2`), `ATTRIBUTION_METHOD` is the name of the attribution method (e.g., `input_times_gradient`), and `<attribution_data>` is the name of the generated attribution data file ending with `_inseq.json`.

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at: http://localhost:3000

### Manual Setup

#### 1. Extractor Setup

```bash
cd extractor
poetry install
# Run attribution analysis (see Extractor Usage below)
```

#### 2. Analyzer Setup

```bash
cd analyzer
poetry install
poetry run uvicorn main:app --reload
```

#### 3. Visualizer Setup

```bash
cd visualizer
npm install
npm run dev
```

## Extractor Usage

The extractor component generates attribution data by analyzing LLM token relationships:

1. Install dependencies:
   ```bash
   cd extractor
   poetry install
   ```

2. Run attribution analysis with a benchmark script:
   ```bash
   poetry run python scripts/run_facts_attribution.py
   ```

3. Output will be saved to `extractor/scripts/output/data/`

4. Copy the generated data to the analyzer:
   ```bash
   # Create model and method directories if they don't exist 
   mkdir -p analyzer/data/MODEL_NAME/method_ATTRIBUTION_METHOD/data
   
   # Copy the attribution data files
   cp extractor/scripts/output/data/*.json analyzer/data/MODEL_NAME/method_ATTRIBUTION_METHOD/data/
   ```

5. You can explore the generated data using Jupyter notebooks in the `extractor/notebooks/` directory.

### Using Custom Datasets

To run attribution analysis on your own datasets, see the detailed [Custom Dataset Guide](extractor/docs/custom_dataset_guide.md). This guide explains how to:

- Create a custom attribution script
- Configure models and attribution methods
- Format your data for optimal results
- Process your own prompts with various attribution techniques

## Analyzer Usage

The analyzer serves attribution data through a REST API:

1. Start the server:
   ```bash
   cd analyzer
   poetry run uvicorn main:app --reload
   ```

2. The API will be available at http://localhost:8000
3. View API documentation at http://localhost:8000/docs

The analyzer automatically detects and serves any attribution data placed in the `analyzer/data` directory, organized by model and attribution method.

## Visualizer Usage

The visualizer provides an interactive web interface for exploring token attributions:

1. Start the development server:
   ```bash
   cd visualizer
   npm install
   npm run dev
   ```

2. Access the web interface at http://localhost:3000

3. Use the interface to:
   - Browse models and attribution methods
   - Visualize token relationships
   - Analyze performance metrics
   - Compare different models and methods

For detailed documentation on all visualizer features, see the [Visualizer Documentation](visualizer/docs/).

## Project Structure

```
TokenLense/
├── docker-compose.yml            # Docker Compose configuration
├── extractor/                    # Attribution extractor component
│   ├── notebooks/                # Jupyter notebooks for data exploration
│   ├── scripts/                  # Attribution analysis scripts
│   │   └── output/data/          # Generated attribution data
│   └── src/                      # Core attribution analysis code
├── analyzer/                     # API service component
│   ├── data/                     # Attribution data organized by model/method
│   │   ├── MODEL_NAME/           # Each model has its own directory
│   │   │   └── method_NAME/      # Each attribution method has a subdirectory
│   │   │       └── data/         # JSON attribution data files
│   ├── routers/                  # API endpoint implementations
│   └── main.py                   # API server entry point
└── visualizer/                   # Web visualization frontend
    ├── components/               # React components
    │   ├── attribution/          # Token visualization components
    │   ├── charts/               # Performance visualization components
    │   └── comparison/           # Comparison view components
    ├── app/                      # Next.js pages and routes
    └── docs/                     # Visualizer documentation
```

## Data Flow

1. **Extractor**: Runs attribution analysis on models and generates JSON data files
2. **Manual Transfer**: Copy JSON files from extractor output to the analyzer data directory
3. **Analyzer**: Serves attribution data through REST API endpoints
4. **Visualizer**: Fetches data from the analyzer API and renders visualizations

## Docker Setup

The project includes Docker configurations for easy deployment:

- `docker-compose.yml`: Orchestrates the analyzer and visualizer services
- `analyzer/Dockerfile`: Containerizes the Python API service
- `visualizer/Dockerfile`: Containerizes the Next.js frontend