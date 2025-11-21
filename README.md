# AI Vision Platform - Image Analysis and Q&A

An intelligent image analysis platform that combines state-of-the-art object detection with AI-powered question answering capabilities. Upload images for automatic object detection using YOLOv11, then query the results using natural language through Google's Gemini AI.

## Features

- 🔍 **Object Detection**: Real-time object detection using YOLOv11 with bounding boxes and confidence scores
- 💬 **Intelligent Q&A**: Ask natural language questions about detected objects using RAG (Retrieval-Augmented Generation)
- 🔐 **Secure Authentication**: JWT-based user authentication and authorization
- 🚀 **Modern Stack**: Built with FastAPI, leveraging async capabilities for high performance
- 🐳 **Docker Ready**: Fully containerized with Docker Compose for easy deployment

## Architecture Overview

### Technology Stack

**Backend Framework:**
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Uvicorn**: ASGI server for high-performance async operations

**AI/ML Components:**
- **Ultralytics YOLOv11**: State-of-the-art object detection model
- **Google Gemini**: Advanced LLM for natural language understanding and generation
- **Pydantic-AI**: Framework for building production-grade AI applications

**Database & Authentication:**
- **SQLite**: Lightweight database for user management
- **SQLAlchemy**: ORM for database operations
- **JWT (python-jose)**: Token-based authentication
- **bcrypt**: Secure password hashing

**Package Management:**
- **uv**: Ultra-fast Python package installer and resolver


### Key Technical Choices

1. **FastAPI over Flask/Django**:
   - Native async/await support for handling multiple image processing requests
   - Automatic OpenAPI documentation
   - Built-in data validation with Pydantic
   - Superior performance for I/O-bound operations

2. **YOLOv11 for Object Detection**:
   - Latest version with improved accuracy and speed
   - Pre-trained weights (yolo11m.pt) for 80+ object classes
   - Real-time inference capabilities
   - Comprehensive bounding box and confidence information

3. **Google Gemini for Q&A**:
   - Advanced reasoning capabilities
   - Efficient token usage with temperature control (0.3)
   - Fast response times for user queries
   - Natural language understanding

4. **JWT Authentication**:
   - Stateless authentication suitable for distributed systems
   - Secure token-based approach
   - Easy to scale horizontally

5. **uv Package Manager**:
   - 10-100x faster than pip
   - Reliable dependency resolution
   - Perfect for Docker builds with caching

6. **SQLite Database**:
   - Zero configuration required
   - Sufficient for user management
   - Easy to backup and migrate
   - Can be upgraded to PostgreSQL if needed

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your system
- Google API key for Gemini AI (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))
- YOLO model weights file (`yolo11m.pt`)

### Local Setup (Without Docker)

If you prefer to run the application without Docker:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd image-analysis-and-qa
   ```

2. **Install Python 3.12+ and uv:**
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create a virtual environment and install dependencies:**
   ```bash
   uv sync
   ```

4. **Download YOLO model weights:**
   ```bash
   # The yolo11m.pt file should be in the project root
   # It will be automatically downloaded on first use if not present
   ```

5. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SECRET_KEY=your_secret_key_here_min_32_chars
   ALGORITHM=HS256
   ```

6. **Run the application:**
   ```bash
   uv run fastapi dev --host 0.0.0.0 --port 8000
   ```

7. **Access the application:**
   - Open your browser and navigate to `http://localhost:8000`
   - API documentation available at `http://localhost:8000/docs`

## Running with Docker

Docker is the recommended way to run this application for consistent behavior across environments.

### Quick Start with Docker Compose

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd image-analysis-and-qa
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SECRET_KEY=your_secret_key_here_min_32_chars
   ALGORITHM=HS256
   ```

3. **Build and run the application:**
   ```bash
   docker compose up --build
   ```

   The application will be available at `http://localhost:8000`

4. **Run in detached mode (background):**
   ```bash
   docker compose up -d
   ```

5. **View logs:**
   ```bash
   docker compose logs -f
   ```

6. **Stop the application:**
   ```bash
   docker compose down
   ```

### Development with Hot Reload

The Docker Compose configuration includes a `watch` feature for development:

```bash
docker compose watch
```

This will:
- Automatically sync code changes to the container
- Reload the FastAPI server when files change
- Rebuild the image if `uv.lock` changes

### Docker Commands Reference

```bash
# Build the image
docker compose build

# Start services
docker compose up

# Start in background
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f web

# Rebuild and start
docker compose up --build

# Remove volumes (resets database)
docker compose down -v

# Execute commands in container
docker compose exec web bash
```

## API Endpoints

### Authentication Endpoints

- `POST /auth/signup` - Create a new user account
- `POST /auth/login` - Login and receive JWT token
- `GET /auth/me` - Get current user information (requires JWT)

### Image Analysis Endpoints

- `POST /detect` - Upload an image for object detection
  - Returns: Annotated image with bounding boxes + detection results
  
- `POST /query` - Ask questions about the last detected image
  - Body: `{"query": "How many people are in the image?"}`
  - Returns: AI-generated answer based on detection data

### Frontend Endpoints

- `GET /` - Serves the login page
- `GET /application` - Serves the main application interface
- `GET /static/*` - Serves static frontend assets

## Project Structure

```
image-analysis-and-qa/
├── backend/                # Backend application code
│   ├── __init__.py
│   ├── app.py             # FastAPI application setup
│   ├── routes.py          # Image detection and RAG endpoints
│   ├── auth_routes.py     # Authentication endpoints
│   ├── auth.py            # JWT token handling
│   ├── database.py        # SQLAlchemy models and DB operations
│   └── models.py          # Pydantic models for request/response
├── frontend/              # Frontend HTML/JS files
│   ├── login_page.html    # User authentication interface
│   └── application.html   # Main application interface
├── test/                  # Test utilities
│   ├── fix_database.py
│   └── test_auth.py
├── main.py               # Application entry point
├── Dockerfile            # Docker image definition
├── compose.yml           # Docker Compose configuration
├── pyproject.toml        # Project dependencies and metadata
├── uv.lock              # Locked dependency versions
├── .env                 # Environment variables (create this)
├── .dockerignore        # Files to exclude from Docker build
├── yolo11m.pt           # YOLO model weights
└── users.db             # SQLite database (auto-created)
```

## Usage Examples

### 1. Create an Account and Login

Navigate to `http://localhost:8000` and:
1. Click "Sign Up" to create an account
2. Enter your email, name, and password
3. Login with your credentials
4. You'll receive a JWT token for API access

### 2. Detect Objects in an Image

1. After logging in, you'll be redirected to the application page
2. Click "Choose File" and select an image
3. Click "Analyze Image"
4. View the annotated image with detected objects

### 3. Ask Questions About the Image

After detection, use the query box:
- "How many people are in the image?"
- "What objects have the highest confidence?"
- "List all detected objects"
- "What are the bounding box coordinates?"

## License

This project is provided as-is for educational and development purposes.

## Support

For issues, questions, or contributions, please open an issue in the repository.
