#  Brain Tumor Detection using Deep Learning

##  Overview

This project is a web-based application that detects the presence of brain tumors from MRI images using a trained deep learning model. The model is integrated into a Flask web app where users can upload an MRI scan and receive a prediction.

---

##  Features

* Upload MRI images through a web interface
* Predict whether a tumor is present or not
* Simple and clean UI using HTML/CSS
* Backend powered by Flask

---

##  Model Details

* Convolutional Neural Network (CNN)
* Binary Classification:

  * Tumor
  * No Tumor
* Trained on MRI image dataset

---

##  Tech Stack

* Python
* TensorFlow / Keras
* Flask
* HTML / CSS

---

##  Project Structure

```
MRI-detection/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   └── (images / uploads)
```

---

##  How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/Shivansh-Joshii/MRI-detection.git
cd MRI-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

##  Important Note

The trained model file is not included in this repository due to size limitations.

You can retrain the model using:

```
python train.py
```

---

##  Demo

https://github.com/Shivansh-Joshii/MRI-detection/blob/main/demo/Screenshot%202026-05-01%20181445.png

---

##  Future Improvements

* Improve model accuracy
* Add better UI/UX
* Deploy the app online
* Support multiple image formats

---

##  Author

**Shivansh Joshi**
