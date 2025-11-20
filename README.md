# MLOps Major Project

A complete end‑to‑end implementation of an MLOps pipeline, built to demonstrate practical skills across model training, CI/CD automation, Dockerization, and Kubernetes deployment. This project was designed not only to work—but to be clear, readable, and production‑ready. The goal is to simulate how real-world ML applications are developed, packaged, tested, and deployed to scalable environments. – End-to-End ML Deployment on Docker & Kubernetes

 Project Overview

This project implements a complete **MLOps pipeline** for training, testing, containerizing, automating, and deploying a Machine Learning model using:

* **Python (PyTorch)** for training/testing
* **Docker** for containerization
* **GitHub Actions** for CI & Docker Image Build/Push
* **Kubernetes (Docker Desktop)** for deployment using Deployment + NodePort Service

The model is trained on the **Olivetti Faces dataset**, and the deployed app provides a simple UI to upload a **flattened 4096‑dim face (.npy)** and get the predicted class.

---

## 🚀 Features Implemented

### ✔ Model Development

* `train.py` trains a simple neural network.
* `test.py` evaluates accuracy.
* Model saved to: `models/savedmodel.pth`.

### ✔ Dockerization

* Dockerfile created using Python 3.9 slim.
* Local build + test successful.
* Image pushed to Docker Hub:

  * **`saurabh0112/mlops-major:latest`**

### ✔ CI/CD Pipeline via GitHub Actions

Two separate workflows:

1. **Train & Test CI** (dev branch)
2. **Docker Build & Push CI/CD** (docker_cicd branch)

All workflows passed successfully.

### ✔ Kubernetes Deployment

Using Docker Desktop Kubernetes:

* **Deployment** with 3 replicas
* **NodePort Service** exposing port **30080**
* App accessible at:
  **[http://localhost:30080](http://localhost:30080)**

---

## 📂 Project Structure

```
mlops-major/
│
├── train.py
├── test.py
├── requirements.txt
├── Dockerfile
├── models/
│   └── savedmodel.pth
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/workflows/
│   ├── ci.yml
│   └── docker-ci.yml
│
└── README.md
```

---

## 🐳 Docker Instructions

### **Build the image:**

```
docker build -t mlops-major-local:latest .
```

### **Run the container locally:**

```
docker run --rm -p 5000:5000 mlops-major-local:latest
```

### **Tag & Push to Docker Hub:**

```
docker tag mlops-major-local:latest saurabh0112/mlops-major:latest

docker push saurabh0112/mlops-major:latest
```

---

## 🔄 GitHub Actions Workflows

### **1. CI – Train & Test (dev branch)**

Automatically runs:

* Install deps
* Train model
* Run tests

### **2. Docker CI/CD (docker_cicd branch)**

Automatically:

* Builds Docker image
* Pushes to Docker Hub

Uses GitHub Secret:

* `DOCKERHUB_USERNAME`
* `DOCKERHUB_TOKEN`

---

## ☸ Kubernetes Deployment

### **Apply Deployment & Service:**

```
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### **Check resources:**

```
kubectl get pods -l app=mlops-major
kubectl get deployments
kubectl get svc mlops-major-service
```

### **Access Application:**

Open browser:

```
http://localhost:30080
```

---

## 🖼 Screenshots (Add in Report)

* Docker image build
* CI/CD workflow success
* Kubernetes pods & services
* Running app UI on NodePort

---

## 📙 Final Notes

This completes the full MLOps lifecycle:

1. Code → 2. CI → 3. Docker → 4. Registry → 5. Kubernetes Deployment

You can now proceed to write the **final project report PDF** using this README as the reference.

---

## ✨ Author

**Saurabh Sharma**
(M.Tech – Data & Computational Science, IIT Jodhpur)
