# SEA Assignment - Optimization Algorithms

This project implements and compares different optimization algorithms including Hill Climbing, Random Search, and Genetic Algorithms for solving a 4D coffee brewing optimization problem.

## How to Install the Project

### Prerequisites
- Python 3.11 or higher
- uv package manager

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/einfachhuba/SEA_Assignment.git
   cd SEA_Assignment
   ```

2. **Create and start a uv virtual environment:**
    ```bash
        uv venv
    ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

## How to Build Docker

### Build the Docker Image

1. **Build the Docker image:**
   ```bash
   docker build -t sea-assignment .
   ```

2. **Alternatively, use docker-compose to build:**
   ```bash
   docker-compose build
   ```

### Docker Configuration Files

The project includes several Docker configuration files:
- `docker-compose.yml` - Main development configuration
- `docker-compose.prod.yml` - Production configuration
- `docker-compose.override.yml` - Local overrides
- `docker/dockerfile` - Docker image definition
- `docker/start.sh` - Container startup script

## How to Start the Project

### Local Development

1. **Start the Streamlit application:**
   ```bash
   uv run streamlit run app/Home.py
   ```

2. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`

### Using Docker

1. **Start with docker-compose (recommended):**
   ```bash
   docker-compose up
   ```

2. **Start with docker-compose in detached mode:**
   ```bash
   docker-compose up -d
   ```

3. **Start production environment:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
   ```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`

### Available Pages

The application includes the following pages:
- **Home** - Project overview and navigation
- **Chat Interface** - Interactive chat functionality  
- **Random Search** - Demo Page for random search algorithm implementation
- **Hill Climbing** - Hill climbing algorithm with visualization
- **Genetic Algorithms** - Comprehensive genetic algorithm implementation

### Stopping the Application

**Local development:**
- Press `Ctrl+C` in the terminal

**Docker:**
```bash
docker-compose down
```

## Project Structure

```
SEA_Assignment/
├── app/
│   ├── Home.py                 # Main entry point
│   ├── pages/                  # Streamlit pages
│   └── utils/                  # Algorithm implementations
├── docker/                     # Docker configuration
├── Assignment_Sheets/          # Assignment documentation
├── docker-compose*.yml         # Docker compose files
└── uv.lock                     # Python dependencies lock file
```