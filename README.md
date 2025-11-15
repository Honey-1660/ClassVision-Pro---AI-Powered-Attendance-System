# ClassVision-Pro---AI-Powered-Attendance-System
# ClassVision Pro - AI-Powered Attendance System

An intelligent classroom attendance management system using facial recognition technology. ClassVision Pro automates attendance tracking with face detection, student registration, and schedule management.

## Features

- **🎥 Real-time Facial Recognition** - Automatic student identification using LBPH face recognition
- **📋 Schedule Management** - Create and manage class schedules with specific days and times
- **👥 Student Registration** - Easy student enrollment with captured facial data
- **✅ Automated Attendance** - Mark attendance in seconds with face detection
- **📊 Attendance Records** - View and export attendance data by date and subject
- **🔊 Audio Feedback** - Text-to-speech notifications for attendance confirmation
- **🖥️ User-Friendly GUI** - Intuitive Tkinter interface with live camera feed

## Tech Stack

- **Python 3.13**
- OpenCV (cv2) - Face detection & recognition
- NumPy - Numerical operations
- PIL/Pillow - Image processing
- Tkinter - GUI framework
- pyttsx3 - Text-to-speech
- JSON - Data storage

## Installation

bash
pip install opencv-contrib-python numpy pillow pyttsx

## Usage
python class_vision.py

## project strecture
├── class_vision.py       # Main application
├── dataset/              # Student facial images
├── attendance/           # Attendance records (CSV)
├── trainer/              # Trained model & labels
├── schedule.json         # Class schedule configuration
└── README.md

 ## How It Works
 
Register Students - Capture 5 face images per student

Train Model - Build recognition model from collected images

Add Classes - Define class schedules

Mark Attendance - System scans and marks attendance automatically

Requirements

Webcam/Camera device

Python 3.x

Windows/Linux/Mac

 ## License

Open Source

Author
[ MADHUSUDAN sarkar]
