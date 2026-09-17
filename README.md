<div align="center">

# CCTV-Object-Identification
</div>
## The Strategic "Why"

> The manual process of tracking objects across multiple CCTV feeds is an incredibly time-consuming, error-prone, and resource-intensive task for security personnel. In critical situations, the delay in identifying and locating specific individuals or items can have severe consequences, leading to operational inefficiencies, increased security risks, and missed opportunities for rapid intervention. Traditional video analysis often lacks the precision and automation required to effectively manage vast amounts of visual data, leaving organizations vulnerable.

This project addresses these critical challenges by providing an intelligent, automated object re-identification system. By leveraging cutting-edge deep learning models like YOLO for robust object detection and DinoV2 for generating highly discriminative image embeddings, we enable rapid and accurate tracking of objects across diverse camera views. This significantly reduces manual effort, enhances situational awareness, and empowers security teams with the tools for proactive and efficient incident response, transforming raw video data into actionable intelligence.

---

## Key Features

*   🎯 **Precise Object Detection**: Utilizes YOLOv8 for state-of-the-art real-time object detection, ensuring high accuracy in identifying targets within complex environments.
*   🔍 **Intelligent Re-Identification**: Leverages DinoV2 image embeddings to accurately match and track objects across different camera views, even with variations in pose or lighting.
*   ⚡ **Rapid Incident Response**: Drastically cuts down the time required to locate and track specific objects or individuals, enabling faster and more effective security interventions.
*   🌐 **Scalable Microservices Architecture**: Built with a Flask backend and React frontend, providing a modular, scalable, and responsive platform suitable for various deployment scenarios.
*   📊 **Intuitive User Interface**: Offers a user-friendly web interface for uploading footage, querying objects, and visualizing tracking results, simplifying complex analytical tasks.
*   🛡️ **Enhanced Security Posture**: Transforms passive surveillance into an active, intelligent monitoring system, significantly improving overall security and operational efficiency.

---

## Technical Architecture

This project is engineered with a modern, decoupled architecture, separating the backend API from the frontend user interface for enhanced scalability and maintainability.

### Tech Stack

| Technology   | Purpose                           | Key Benefit                                      |
| :----------- | :-------------------------------- | :----------------------------------------------- |
| **Python**   | Primary Language, Backend Logic   | Versatile, extensive ML ecosystem, rapid development |
| **YOLOv8**   | Object Detection                  | High-performance, real-time, accurate detection  |
| **DinoV2**   | Image Embeddings, Re-ID           | State-of-the-art visual feature extraction, robust re-identification |
| **Flask**    | Backend API Framework             | Lightweight, flexible, easy to integrate ML models |
| **React**    | Frontend User Interface           | Component-based, dynamic, responsive UI development |

### Directory Structure

```
.
├── .gitignore
├── flask-backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── models/
│   │   └── yolov8s.pt
│   └── .env.example
├── react-client/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   └── App.js
│   ├── package.json
│   ├── package-lock.json
│   └── .env.example
└── README.md
```

---

## Operational Setup

Follow these steps to get the CCTV-Object-Identification system up and running on your local machine.

### Prerequisites

Ensure you have the following installed:

*   **Python 3.8+**
*   **Node.js 16+** and **npm**
*   **Git**

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-org/CCTV-Object-Identification.git
    cd CCTV-Object-Identification
    ```

2.  **Backend Setup (Flask API):**

    Navigate to the `flask-backend` directory, create a virtual environment, install dependencies, and download the YOLO model.

    ```bash
    cd flask-backend
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

    *Note*: The `yolov8s.pt` model should be placed in `flask-backend/models/`. If it's not present, you may need to download it:
    ```bash
    mkdir -p models
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt -P models/
    ```

    Run the Flask application:

    ```bash
    flask run
    ```
    The backend API will typically run on `http://127.0.0.1:5000`.

3.  **Frontend Setup (React Client):**

    Open a new terminal, navigate to the `react-client` directory, and install its dependencies.

    ```bash
    cd ../react-client
    npm install
    ```

    Start the React development server:

    ```bash
    npm start
    ```
    The frontend application will typically open in your browser at `http://localhost:3000`.

### Environment Configuration

Both the backend and frontend require environment variables for proper operation.

#### `flask-backend/.env`

Create a file named `.env` in the `flask-backend/` directory based on `.env.example`.

```ini
# flask-backend/.env
FLASK_APP=app.py
FLASK_ENV=development
# Add any other backend-specific configurations, e.g., model paths or API keys
```

#### `react-client/.env`

Create a file named `.env` in the `react-client/` directory based on `.env.example`.

```ini
# react-client/.env
REACT_APP_API_BASE_URL=http://127.0.0.1:5000
# Add any other frontend-specific configurations
```

---

## Community & Governance

We welcome contributions from the community to make this project even better!

### Contributing

We encourage you to contribute to the CCTV-Object-Identification project. To get started:

1.  **Fork** the repository to your own GitHub account.
2.  **Clone** your forked repository to your local machine.
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/issue-description`.
4.  **Make your changes** and commit them with clear, concise messages.
5.  **Push** your branch to your forked repository.
6.  **Open a Pull Request** against the `main` branch of the original repository. Please provide a detailed description of your changes and why they are valuable.

Please ensure your code adheres to the existing style and conventions.

### License

This project is licensed under the **MIT License**.

A copy of the license can be found in the `LICENSE` file at the root of the repository.

**Summary of Permissions:**
*   **Commercial Use**: Allowed
*   **Modification**: Allowed
*   **Distribution**: Allowed
*   **Private Use**: Allowed

**Conditions:**
*   **License and Copyright Notice**: Must be included with the software.

**Limitations:**
*   **Liability**: Software is provided "as is," without warranty of any kind. The authors or copyright holders shall not be liable for any claim, damages, or other liability arising from or in connection with the software.
