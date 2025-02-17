# Dota 2 Match Predictor

## Overview

The Dota 2 Match Predictor is a web application built using FastAPI that predicts the outcome of Dota 2 matches based on selected heroes for both the Radiant and Dire teams. This application utilizes a pre-trained Random Forest model to make predictions and provides a user-friendly web interface for interaction.

## Technologies Used

- **FastAPI**: A modern web framework for building APIs quickly and efficiently.
- **Uvicorn**: An ASGI server used to run FastAPI applications.
- **Scikit-learn**: A machine learning library for Python that provides tools for model training and evaluation.
- **Joblib**: A library for saving and loading Python objects, specifically used to manage the trained model.
- **Numpy**: A library for numerical operations, used to handle data efficiently.
- **Streamlit**: A framework for building interactive web applications, providing an easy way to create a user interface for machine learning models and visualizations.

## Getting Started

### Prerequisites

- **Python 3.9** or higher
- **pip** (Python package manager)
- **Docker** (optional, for containerization)

### Setting Up the Environment

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment (Optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:

   Navigate to the `code/deployment/api/` directory and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Running Locally

1. **Start the FastAPI Application**:

   Navigate to the `code/deployment/api/` directory and run the application with:

   ```bash
   python main.py
   ```

2. **Start the Streamlit Application**:

   Navigate to the `code/deployment/app/` directory and run the application with:

   ```bash
   streamlit run frontend.py
   ```
   
3. **Access the Application**:

   Open your web browser and go to `http://localhost:8501/` to view the application.

#### Option 2: Running with Docker

1. **Build the Docker Image**:

   In the deployment directory, run:

   ```bash
   docker compose up --build
   ```

2. **Access the Application**:

   Open your web browser and go to `http://localhost:8501/`.

## API Endpoints

- **POST /predict/**: Predicts the match outcome based on selected heroes.
  - **Request Body**:
    
    • List of 5 unique hero IDs for Radiant team
    
    • List of 5 unique hero IDs for Dire team
    
    ```json
    {
      "radiant_heroes": [1, 2, 3, 4, 5],
      "dire_heroes": [6, 7, 8, 9, 10]
    }
    ```
  - **Response**:
    
    • "Radiant" or "Dire"
    
    ```json
    {
      "winner": "Radiant"
    }
    ```

- **GET /heroes/**: Returns a list of available hero IDs and names.

## Frontend

The frontend consists of a HTML page generated by Streamlit. This page allows users to select heroes and submit them for prediction.

