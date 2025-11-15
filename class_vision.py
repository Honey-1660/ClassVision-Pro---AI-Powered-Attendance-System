import os
import cv2
import csv
import time
import json
import threading
import numpy as np
import pyttsx3
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import shutil
import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(cv2.__version__)
print(hasattr(cv2.face, 'LBPHFaceRecognizer_create'))

# ========================
# CONFIGURATION
# ========================
BG_COLOR = "#85ECCD"          # Dark blue background
BUTTON_COLOR = "#CFCD60"       # Bright blue buttons
HOVER_COLOR = "#00E5FF"        # Cyan hover effect
ACCENT_COLOR = "#BC00DD"       # Purple accents
TEXT_COLOR = "#F30303"         # White text
STATUS_BAR_COLOR = "#787DD8"   # Dark status bar

# ========================
# MAIN APPLICATION CLASS
# ========================
class ClassVision:
    def __init__(self, root):
        self.root = root
        self.root.title("Class Vision Pro")
        self.root.geometry("1400x800")
        self.root.configure(bg=BG_COLOR)
        
        try:
            # Initialize voice engine
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            self.voice_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Could not initialize TTS engine: {str(e)}")
            self.voice_engine = None
        
        # Data stores configuration
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.base_dir, "dataset")
        self.attendance_dir = os.path.join(self.base_dir, "attendance")
        self.training_dir = os.path.join(self.base_dir, "trainer")
        self.schedule_file = os.path.join(self.base_dir, "schedule.json")
        self.trainer_path = os.path.join(self.training_dir, "trainer.yml")
        self.label_map_path = os.path.join(self.training_dir, "labels.json")
        
        # Create directories if they don't exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        
        # Initialize face recognition
        try:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.trained = os.path.exists(self.trainer_path)
            
            # Try to load existing model
            if self.trained:
                try:
                    self.recognizer.read(self.trainer_path)
                except Exception as e:
                    print(f"Error loading trained model: {str(e)}")
                    self.trained = False
        except AssertionError:
            messagebox.showerror("Fatal Error", f"Could not initialize face recognition: {str(e)}")
            self.root.destroy()
            return
            
        self.label_map = self.load_label_map()
        self.schedule = self.load_schedule()
        self.current_classes = {}
        self.current_user = ""
        self.attendance_log = []
        self.camera = None
        self.is_running = False
        self.attendance_thread = None
        self.register_thread = None
        self.train_thread = None
        
        # Create UI
        self.create_ui()
        
        # Start camera and schedule checker
        self.start_camera()
        self.update_current_classes()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # ========================
    # UI COMPONENTS
    # ========================
    def create_ui(self):
        # Main container
        self.main_frame = Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Left panel (camera)
        self.left_panel = Frame(self.main_frame, bg=BG_COLOR)
        self.left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=20, pady=20)

        # Camera view
        self.setup_camera_view()

        # Right panel (controls)
        self.right_panel = Frame(self.main_frame, bg=BG_COLOR)
        self.right_panel.pack(side=RIGHT, fill=Y, padx=20, pady=20)

        # Schedule management
        self.setup_schedule_controls()

        # Student management
        self.setup_student_controls()

        # Attendance controls
        self.setup_attendance_controls()

        # Status bar
        self.setup_status_bar()

    def setup_camera_view(self):
        camera_frame = LabelFrame(self.left_panel, text="Live Camera Feed", 
                                bg="#000000", fg=TEXT_COLOR, 
                                font=("Arial", 10, "bold"))
        camera_frame.pack(fill=BOTH, expand=True)

        # Camera placeholder with aspect ratio 4:3
        self.camera_label = Label(camera_frame, bg="#000000", text="Initializing camera...",
                                font=("Arial", 12), fg=TEXT_COLOR)
        self.camera_label.pack(fill=BOTH, expand=True)

    def setup_schedule_controls(self):
        frame = LabelFrame(self.right_panel, text="CLASS SCHEDULE", 
                          bg=BG_COLOR, fg=TEXT_COLOR, 
                          font=("Arial", 10, "bold"))
        frame.pack(fill=X, pady=10)

        # Current classes display
        current_class_frame = Frame(frame, bg=BG_COLOR)
        current_class_frame.pack(fill=X, pady=5)
        
        Label(current_class_frame, text="Current Classes:", 
             bg=BG_COLOR, fg=TEXT_COLOR, 
             font=("Arial", 9, "bold")).pack(anchor=W)
             
        self.current_class_label = Label(current_class_frame, 
                                       text="No classes scheduled now", 
                                       bg=BG_COLOR, fg=TEXT_COLOR, 
                                       font=("Arial", 9), wraplength=250,
                                       justify=LEFT)
        self.current_class_label.pack(fill=X)

        # Schedule treeview with scrollbar
        tree_frame = Frame(frame, bg=BG_COLOR)
        tree_frame.pack(fill=BOTH, expand=True, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.schedule_tree = ttk.Treeview(tree_frame, columns=("Subject", "Days", "Time"), 
                                         show="headings", yscrollcommand=scrollbar.set,
                                         height=5)
        scrollbar.config(command=self.schedule_tree.yview)
        
        self.schedule_tree.heading("Subject", text="Subject", anchor=W)
        self.schedule_tree.heading("Days", text="Days", anchor=W)
        self.schedule_tree.heading("Time", text="Time", anchor=W)
        
        self.schedule_tree.column("Subject", width=120, stretch=False)
        self.schedule_tree.column("Days", width=80, stretch=False)
        self.schedule_tree.column("Time", width=50, stretch=False)
        
        self.schedule_tree.pack(fill=BOTH, expand=True)

        # Schedule buttons
        btn_frame = Frame(frame, bg=BG_COLOR)
        btn_frame.pack(fill=X, pady=5)

        self.add_class_btn = ttk.Button(btn_frame, text="Add Class", command=self.add_class)
        self.add_class_btn.pack(side=LEFT, padx=2)
        
        self.remove_class_btn = ttk.Button(btn_frame, text="Remove Class", command=self.remove_class)
        self.remove_class_btn.pack(side=LEFT, padx=2)
        
        self.refresh_btn = ttk.Button(btn_frame, text="Refresh", command=self.refresh_schedule)
        self.refresh_btn.pack(side=RIGHT, padx=2)

        # Initialize the schedule display
        self.refresh_schedule()

    def setup_student_controls(self):
        frame = LabelFrame(self.right_panel, text="STUDENT MANAGEMENT", 
                          bg=BG_COLOR, fg=TEXT_COLOR, 
                          font=("Arial", 10, "bold"))
        frame.pack(fill=X, pady=10)

        btn_style = ttk.Style()
        btn_style.configure('Student.TButton', padding=5)

        self.register_btn = ttk.Button(frame, text="Register New Student", 
                                     command=self.register_user, style='Student.TButton')
        self.register_btn.pack(fill=X, pady=2)
        
        self.delete_btn = ttk.Button(frame, text="Delete Student", 
                                   command=self.delete_student, style='Student.TButton')
        self.delete_btn.pack(fill=X, pady=2)
        
        self.view_btn = ttk.Button(frame, text="View Registered Students", 
                                 command=self.view_registered_students, style='Student.TButton')
        self.view_btn.pack(fill=X, pady=2)
        
        self.train_btn = ttk.Button(frame, text="★ Train Recognition Model ★", 
                                  command=self.train_model, style='Student.TButton')
        self.train_btn.pack(fill=X, pady=2)
        
        # Disable train button if no training data exists
        if not os.listdir(self.dataset_dir):
            self.train_btn.config(state=DISABLED)

    def setup_attendance_controls(self):
        frame = LabelFrame(self.right_panel, text="ATTENDANCE SYSTEM", 
                          bg=BG_COLOR, fg=TEXT_COLOR, 
                          font=("Arial", 10, "bold"))
        frame.pack(fill=X, pady=10)

        self.attend_btn = ttk.Button(frame, text="Take Attendance Now", 
                                   command=self.mark_attendance)
        self.attend_btn.pack(fill=X, pady=2)
        
        self.view_attend_btn = ttk.Button(frame, text="View Attendance Records", 
                                        command=self.view_attendance)
        self.view_attend_btn.pack(fill=X, pady=2)
        
        # Disable attendance button if model not trained
        if not self.trained:
            self.attend_btn.config(state=DISABLED)

    def setup_status_bar(self):
        frame = Frame(self.left_panel, bg=STATUS_BAR_COLOR)
        frame.pack(fill=X, pady=(10, 0))

        self.status_text = Text(frame, bg="#1a1a1a", fg=TEXT_COLOR, 
                              font=("Consolas", 9), state=DISABLED, 
                              wrap=WORD, height=5)
        scrollbar = ttk.Scrollbar(frame, command=self.status_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.status_text.config(yscrollcommand=scrollbar.set)
        self.status_text.pack(fill=BOTH, expand=True)

        # Initial status message
        self.add_to_log(f"System initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================
    # CORE FUNCTIONS
    # ========================
    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera device")
                
            self.is_running = True
            self.add_to_log("Camera initialized successfully")
            self.show_camera_feed()
        except Exception as e:
            self.add_to_log(f"Camera error: {str(e)}")
            self.camera_label.config(text="Camera not available")
            self.is_running = False

    def show_camera_feed(self):
        if self.is_running:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Convert to RGB and display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    if not hasattr(self.camera_label, 'imgtk'):
                        self.camera_label.imgtk = imgtk
                        self.camera_label.configure(image=imgtk)
                    else:
                        self.camera_label.imgtk = imgtk
                        self.camera_label.configure(image=imgtk)
            except Exception as e:
                self.add_to_log(f"Camera feed error: {str(e)}")
                self.is_running = False
                self.camera_label.config(text="Camera feed unavailable")
                
        self.root.after(15, self.show_camera_feed)

    def register_user(self):
        self.current_user = simpledialog.askstring("Register Student", 
                                                  "Enter student name or ID:", 
                                                  parent=self.root)
        if not self.current_user:
            return
            
        # Check if user already exists
        if os.path.exists(os.path.join(self.dataset_dir, self.current_user)):
            if not messagebox.askyesno("Confirm", 
                                     f"User '{self.current_user}' already exists. Overwrite?"):
                return
            else:
                # Delete existing directory
                try:
                    shutil.rmtree(os.path.join(self.dataset_dir, self.current_user))
                except Exception as e:
                    self.add_to_log(f"Error clearing existing user: {str(e)}")
                    return

        self.add_to_log(f"Starting registration for: {self.current_user}")
        self.speak(f"Please look straight at the camera. Registration in progress for {self.current_user}")
        
        self.register_thread = threading.Thread(target=self.capture_images, daemon=True)
        self.register_thread.start()

    def capture_images(self):
        self.register_btn.config(state=DISABLED)
        user_dir = os.path.join(self.dataset_dir, str(self.current_user))
        os.makedirs(user_dir, exist_ok=True)

        count = 0
        MAX_IMAGES = 5
        start_time = time.time()
        timeout = 30  # 1 minute timeout
        
        while count < MAX_IMAGES and time.time() - start_time < timeout:
            if not self.is_running:
                break
                
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                
                img_path = os.path.join(user_dir, f"{self.current_user}_{count}.jpg")
                try:
                    cv2.imwrite(img_path, face_img)
                    self.add_to_log(f"Captured image {count}/{MAX_IMAGES} for {self.current_user}")
                    self.speak(f"Image {count} captured")
                    
                    # Show countdown to user
                    remaining = MAX_IMAGES - count
                    self.root.after(0, lambda r=remaining: 
                                  self.current_class_label.config(
                                      text=f"Registration in progress...\n{remaining} more images needed"))
                    time.sleep(1)  # Short pause between captures
                except Exception as e:
                    self.add_to_log(f"Error saving image: {str(e)}")
                    count -= 1  # Retry this image

        self.register_btn.config(state=NORMAL)
        
        if count == MAX_IMAGES:
            self.add_to_log(f"Registration complete for {self.current_user}")
            self.speak(f"Registration complete for {self.current_user}")
            self.root.after(0, lambda: self.current_class_label.config(
                text=f"Registration complete for {self.current_user}"))
            self.train_btn.config(state=NORMAL)
        else:
            self.add_to_log(f"Registration timed out or failed for {self.current_user}")
            self.speak("Registration failed. Please try again.")
            try:
                shutil.rmtree(user_dir)  # Clean up incomplete registration
            except:
                pass

    def train_model(self):
        self.train_thread = threading.Thread(target=self.train_model_thread, daemon=True)
        self.train_thread.start()

    def train_model_thread(self):
        self.train_btn.config(state=DISABLED)
        self.register_btn.config(state=DISABLED)
        self.attend_btn.config(state=DISABLED)

        self.add_to_log("Starting model training...")
        self.speak("Training process started. Please wait.")
        self.root.after(0, lambda: self.current_class_label.config(
            text="Training in progress..."))

        faces = []
        labels = []  # Using 'labels' instead of 'ids' for clarity
        self.label_map = {}
        current_id = 0

        try:
            for root_dir, dirs, _ in os.walk(self.dataset_dir):
                for dir_name in dirs:
                    if dir_name not in self.label_map:
                        self.label_map[dir_name] = current_id
                        current_id += 1
                    
                    user_id = self.label_map[dir_name]
                    folder_path = os.path.join(root_dir, dir_name)
                    
                    for file in os.listdir(folder_path):
                        if file.endswith('.jpg') or file.endswith('.png'):
                            img_path = os.path.join(folder_path, file)
                            try:
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    faces.append(img)
                                    labels.append(user_id)
                                else:
                                    self.add_to_log(f"Could not read image: {img_path}")
                            except Exception as e:
                                self.add_to_log(f"Error processing {img_path}: {str(e)}")

            if len(faces) > 0 and len(labels) > 0:
                self.recognizer.train(faces, np.array(labels, dtype=np.int32))
                self.recognizer.save(self.trainer_path)
                self.save_label_map()
                
                self.add_to_log("Training completed successfully! Model saved.")
                self.speak("Training completed successfully!")
                self.root.after(0, lambda: self.current_class_label.config(
                    text="Training complete.\nModel ready for attendance."))
                self.trained = True
                self.attend_btn.config(state=NORMAL)
            else:
                self.add_to_log("Training failed - no valid training images found!")
                self.speak("Training failed. No valid training images found.")
                self.root.after(0, lambda: self.current_class_label.config(
                    text="Training failed.\nNo valid images found."))
                
        except Exception as e:
            self.add_to_log(f"Training error: {str(e)}")
            self.speak("Training error occurred.")
            self.root.after(0, lambda: self.current_class_label.config(
                text="Training error occurred."))
                
        finally:
            self.train_btn.config(state=NORMAL)
            self.register_btn.config(state=NORMAL)

    def mark_attendance(self):
        if not self.trained:
            messagebox.showerror("Error", "Model not trained yet!", parent=self.root)
            return
            
        if not self.current_classes:
            messagebox.showwarning("No Classes", 
                                 "No classes are scheduled right now", 
                                 parent=self.root)
            return

        attendance_window = Toplevel(self.root)
        attendance_window.title("Attendance System - In Progress")
        attendance_window.geometry("600x400")
        attendance_window.protocol("WM_DELETE_WINDOW", 
                                 lambda: self.stop_attendance(attendance_window))
        
        # Header with current time
        header_frame = Frame(attendance_window)
        header_frame.pack(fill=X)
        
        Label(header_frame, text="Taking Attendance For:", 
             font=("Arial", 12, "bold")).pack(pady=5, anchor=W)
        
        # Show current classes
        for subject in self.current_classes:
            Label(header_frame, text=f"- {subject} at {self.current_classes[subject]}",
                 font=("Arial", 10)).pack(anchor=W)
        
        # Timer and instructions
        Label(header_frame, text="System will scan for 30 seconds", 
             font=("Arial", 9)).pack(pady=10, anchor=W)
        
        # Attendance frame
        attendance_frame = Frame(attendance_window)
        attendance_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        # List to display recognized students
        tree_scroll = ttk.Scrollbar(attendance_frame)
        tree_scroll.pack(side=RIGHT, fill=Y)
        
        self.attendance_list = ttk.Treeview(attendance_frame, 
                                          columns=("Student", "Time"), 
                                          show="headings",
                                          yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=self.attendance_list.yview)
        
        self.attendance_list.heading("Student", text="Student")
        self.attendance_list.heading("Time", text="Time")
        self.attendance_list.column("Student", width=120)
        self.attendance_list.column("Time", width=80)
        
        self.attendance_list.pack(fill=BOTH, expand=True)
        
        # Stop button
        ttk.Button(attendance_window, text="Stop Attendance", 
                  command=lambda: self.stop_attendance(attendance_window)
                  ).pack(pady=5)
        
        # Start recognition thread
        self.attendance_window = attendance_window
        self.attendance_thread = threading.Thread(
            target=self.recognize_faces, 
            args=(attendance_window,),
            daemon=True)
        self.attendance_thread.start()

    def stop_attendance(self, window):
        window.running = False
        window.destroy()
        if self.attendance_thread and self.attendance_thread.is_alive():
            self.attendance_thread.join(timeout=1)
        self.save_attendance()
        self.add_to_log("Attendance process stopped manually")

    def recognize_faces(self, window):
        window.running = True
        attendance_marked = []
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while time.time() - start_time < timeout and getattr(window, 'running', True):
            if not self.is_running:
                break
                
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                
                id, confidence = self.recognizer.predict(face_img)
                
                if confidence < 50:  # Lower confidence is better (0 is perfect match)
                    user_name = next((name for name, id_map in self.label_map.items() 
                                    if id_map == id), None)
                    
                    if user_name and user_name not in attendance_marked:
                        attendance_marked.append(user_name)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Record attendance for each current class
                        for subject in self.current_classes:
                            self.attendance_log.append({
                                "student": user_name,
                                "subject": subject,
                                "timestamp": timestamp
                            })
                        
                        # Update UI
                        self.root.after(0, lambda u=user_name, t=timestamp: 
                                      self.attendance_list.insert("", "end", 
                                                                values=(u, t)))
                        self.add_to_log(f"Attendance marked for: {user_name}")
                        self.speak(f"Recognized {user_name}")
            
            # Update timer in UI
            remaining = int(timeout - (time.time() - start_time))
            self.root.after(0, lambda: window.title(
                f"Attendance - Scanning... ({remaining}s remaining)"))
            
            time.sleep(0.5)
        
        # Finalize attendance
        window.running = False
        self.root.after(0, lambda: window.title("Attendance Complete"))
        self.save_attendance()
        self.add_to_log("Attendance marking completed!")
        self.speak("Attendance process completed.")
        self.root.after(0, lambda: messagebox.showinfo(
            "Complete", 
            f"Attendance marked for {len(attendance_marked)} students",
            parent=window))

    # ========================
    # SCHEDULE MANAGEMENT
    # ========================
    def refresh_schedule(self):
        for i in self.schedule_tree.get_children():
            self.schedule_tree.delete(i)
            
        for subject, details in sorted(self.schedule.items()):
            self.schedule_tree.insert("", "end", values=(
                subject, 
                ", ".join(details["days"]), 
                details["time"]
            ))

    def add_class(self):
        add_window = Toplevel(self.root)
        add_window.title("Add New Class")
        add_window.resizable(False, False)
        
        # Subject input
        Label(add_window, text="Subject Name:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        subject_entry = Entry(add_window, width=25)
        subject_entry.grid(row=0, column=1, padx=5, pady=5, sticky=EW)
        subject_entry.focus_set()
        
        # Time input
        Label(add_window, text="Time (HH:MM):").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        time_entry = Entry(add_window, width=25)
        time_entry.grid(row=1, column=1, padx=5, pady=5, sticky=EW)
        time_entry.insert(0, datetime.now().strftime("%H:%M"))
        
        # Days selection
        days_frame = LabelFrame(add_window, text="Days of Week", padx=5, pady=5)
        days_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=EW)
        
        days_vars = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for i, day in enumerate(days):
            days_vars[day] = IntVar()
            Checkbutton(days_frame, text=day, variable=days_vars[day],
                      anchor=W).grid(row=i, column=0, sticky=W, padx=5, pady=2)
        
        # Button frame
        btn_frame = Frame(add_window)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        def save_class():
            subject = subject_entry.get().strip()
            time_str = time_entry.get().strip()
            days_selected = [day for day, var in days_vars.items() if var.get()]
            
            if not subject:
                messagebox.showerror("Error", "Subject name cannot be empty", parent=add_window)
                return
                
            if not time_str:
                messagebox.showerror("Error", "Time cannot be empty", parent=add_window)
                return
                
            try:
                # Validate time format
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                messagebox.showerror("Error", "Invalid time format. Please use HH:MM", parent=add_window)
                return
                
            if not days_selected:
                messagebox.showerror("Error", "Please select at least one day", parent=add_window)
                return
                
            if subject in self.schedule:
                if not messagebox.askyesno("Confirm", 
                                         f"Subject '{subject}' already exists. Overwrite?", 
                                         parent=add_window):
                    return
            
            self.schedule[subject] = {
                "time": time_str,
                "days": days_selected
            }
            
            self.save_schedule()
            self.refresh_schedule()
            add_window.destroy()
            self.update_current_classes()
            messagebox.showinfo("Success", "Class added successfully!", parent=self.root)
        
        Button(btn_frame, text="Cancel", command=add_window.destroy).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Save Class", command=save_class).pack(side=RIGHT, padx=5)

    def remove_class(self):
        selected = self.schedule_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a class to remove", parent=self.root)
            return
            
        selected_item = selected[0]
        subject = self.schedule_tree.item(selected_item, "values")[0]
        
        confirm = messagebox.askyesno("Confirm", 
                                    f"Remove class: {subject}?\nThis cannot be undone.", 
                                    parent=self.root)
        if confirm:
            del self.schedule[subject]
            self.save_schedule()
            self.refresh_schedule()
            self.update_current_classes()
            self.add_to_log(f"Removed class: {subject}")

    def update_current_classes(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A")  # Full day name
        
        self.current_classes = {}
        for subject, details in self.schedule.items():
            if current_day in details["days"]:
                # Compare times as strings (HH:MM format allows this)
                if details["time"] <= current_time:
                    self.current_classes[subject] = details["time"]
        
        if self.current_classes:
            classes_text = "Current Classes:\n" + "\n".join(
                f"• {subj} ({time})" for subj, time in self.current_classes.items())
        else:
            classes_text = "No classes scheduled right now"
        
        self.current_class_label.config(text=classes_text)
        
        # Check every minute
        self.root.after(60000, self.update_current_classes)

    # ========================
    # DATA MANAGEMENT
    # ========================
    def load_label_map(self):
        if os.path.exists(self.label_map_path):
            try:
                with open(self.label_map_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.add_to_log(f"Error loading label map: {str(e)}")
                return {}
        return {}

    def save_label_map(self):
        try:
            with open(self.label_map_path, "w") as f:
                json.dump(self.label_map, f, indent=4)
        except Exception as e:
            self.add_to_log(f"Error saving label map: {str(e)}")

    def load_schedule(self):
        if os.path.exists(self.schedule_file):
            try:
                with open(self.schedule_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.add_to_log(f"Error loading schedule: {str(e)}")
                # Return default schedule
                return {
                    "Mathematics": {
                        "time": "10:00",
                        "days": ["Monday", "Wednesday", "Friday"]
                    },
                    "Computer Science": {
                        "time": "14:00",
                        "days": ["Tuesday", "Thursday"]
                    }
                }
        else:
            return {
                "Mathematics": {
                    "time": "10:00",
                    "days": ["Monday", "Wednesday", "Friday"]
                },
                "Computer Science": {
                    "time": "14:00",
                    "days": ["Tuesday", "Thursday"]
                }
            }

    def save_schedule(self):
        try:
            with open(self.schedule_file, "w") as f:
                json.dump(self.schedule, f, indent=4)
        except Exception as e:
            self.add_to_log(f"Error saving schedule: {str(e)}")

    def save_attendance(self):
        if not self.attendance_log:
            return
            
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Group by subject
        attendance_by_subject = {}
        for entry in self.attendance_log:
            if entry["subject"] not in attendance_by_subject:
                attendance_by_subject[entry["subject"]] = []
            attendance_by_subject[entry["subject"]].append(entry)
        
        # Save separate CSV for each subject
        errors = []
        for subject, records in attendance_by_subject.items():
            filename = os.path.join(self.attendance_dir, f"attendance_{date_str}_{subject}.csv")
            try:
                with open(filename, "w", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Student Name", "Timestamp"])
                    for record in records:
                        writer.writerow([record["student"], record["timestamp"]])
            except Exception as e:
                errors.append(f"{subject}: {str(e)}")
        
        if errors:
            self.add_to_log(f"Error saving some attendance: {'; '.join(errors)}")
        else:
            self.add_to_log(f"Attendance records saved for {date_str}")

    def view_attendance(self):
        date = simpledialog.askstring("View Attendance", 
                                     "Enter date (YYYYMMDD) or leave empty for today:",
                                     parent=self.root)
        if date is None:  # User cancelled
            return
            
        if not date.strip():
            date = datetime.now().strftime("%Y%m%d")
            
        # Validate date format
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYYMMDD", parent=self.root)
            return
            
        # Find all attendance files for this date
        try:
            files = [f for f in os.listdir(self.attendance_dir) 
                    if f.startswith(f"attendance_{date}") and f.endswith('.csv')]
        except Exception as e:
            messagebox.showerror("Error", f"Could not read attendance: {str(e)}", parent=self.root)
            return
            
        if not files:
            messagebox.showinfo("No Records", 
                              f"No attendance records found for {date}", 
                              parent=self.root)
            return
            
        # Create display window
        view_window = Toplevel(self.root)
        view_window.title(f"Attendance Records - {date}")
        view_window.geometry("800x600")
        
        notebook = ttk.Notebook(view_window)
        notebook.pack(fill=BOTH, expand=True)
        
        for file in files:
            # Extract subject name from filename
            subject = file[len(f"attendance_{date}_"):-4].replace('_', ' ')
            
            # Create tab for this subject
            tab_frame = Frame(notebook)
            notebook.add(tab_frame, text=subject)
            
            # Create treeview with scrollbars
            tree_frame = Frame(tab_frame)
            tree_frame.pack(fill=BOTH, expand=True)
            
            scroll_y = ttk.Scrollbar(tree_frame)
            scroll_y.pack(side=RIGHT, fill=Y)
            
            scroll_x = ttk.Scrollbar(tree_frame, orient=HORIZONTAL)
            scroll_x.pack(side=BOTTOM, fill=X)
            
            tree = ttk.Treeview(tree_frame, 
                              columns=("Student", "Time"), 
                              show="headings",
                              yscrollcommand=scroll_y.set,
                              xscrollcommand=scroll_x.set)
            
            scroll_y.config(command=tree.yview)
            scroll_x.config(command=tree.xview)
            
            tree.heading("Student", text="Student")
            tree.heading("Time", text="Time")
            tree.column("Student", width=150, anchor=W)
            tree.column("Time", width=80, anchor=CENTER)
            
            tree.pack(fill=BOTH, expand=True)
            
            # Load and display data
            try:
                with open(os.path.join(self.attendance_dir, file), "r", encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        tree.insert("", "end", values=row)
            except Exception as e:
                tree.insert("", "end", values=[f"Error loading data: {str(e)}", ""])

    def delete_student(self):
        # Get list of registered students
        try:
            students = os.listdir(self.dataset_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read students: {str(e)}", parent=self.root)
            return
            
        if not students:
            messagebox.showinfo("No Students", "No students registered yet", parent=self.root)
            return
            
        # Create selection dialog
        delete_window = Toplevel(self.root)
        delete_window.title("Delete Student")
        delete_window.resizable(False, False)
        
        Label(delete_window, text="Select student to delete:").pack(pady=5)
        
        student_var = StringVar(delete_window)
        student_var.set(students[0])  # Default value
        
        OptionMenu(delete_window, student_var, *students).pack(pady=5)
        
        def confirm_delete():
            student_name = student_var.get()
            if not student_name:
                return
                
            if messagebox.askyesno("Confirm", 
                                 f"Permanently delete {student_name}?\nThis cannot be undone.", 
                                 parent=delete_window):
                user_dir = os.path.join(self.dataset_dir, student_name)
                if os.path.exists(user_dir):
                    try:
                        shutil.rmtree(user_dir)
                        self.add_to_log(f"Deleted student: {student_name}")
                        
                        # Update label map if it contains this student
                        if student_name in self.label_map:
                            del self.label_map[student_name]
                            self.save_label_map()
                            self.add_to_log(f"Updated label map after deletion")
                        
                        # Offer to retrain model if it was trained
                        if self.trained:
                            if messagebox.askyesno("Retrain Model", 
                                                 "Student deleted. Retrain model now?", 
                                                 parent=delete_window):
                                self.train_model()
                    except Exception as e:
                        messagebox.showerror("Error", f"Could not delete student: {str(e)}", parent=delete_window)
                else:
                    messagebox.showerror("Error", f"Student not found: {student_name}", parent=delete_window)
                    
                delete_window.destroy()
        
        Button(delete_window, text="Cancel", command=delete_window.destroy).pack(side=LEFT, padx=5, pady=5)
        Button(delete_window, text="Delete", command=confirm_delete).pack(side=RIGHT, padx=5, pady=5)

    def view_registered_students(self):
        try:
            students = os.listdir(self.dataset_dir)
            student_count = len(students)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read students: {str(e)}", parent=self.root)
            return
            
        if students:
            # Create a window with a list of students
            view_window = Toplevel(self.root)
            view_window.title("Registered Students")
            view_window.geometry("400x500")
            
            Label(view_window, text=f"Registered Students ({student_count}):",
                 font=("Arial", 12, "bold")).pack(pady=10)
            
            # Create scrollable list
            frame = Frame(view_window)
            frame.pack(fill=BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side=RIGHT, fill=Y)
            
            tree = ttk.Treeview(frame, columns=("Name",), show="headings",
                              yscrollcommand=scrollbar.set)
            scrollbar.config(command=tree.yview)
            
            tree.heading("Name", text="Student Name")
            tree.column("Name", width=350)
            tree.pack(fill=BOTH, expand=True)
            
            for student in sorted(students):
                tree.insert("", "end", values=(student,))
        else:
            messagebox.showinfo("No Students", "No students registered yet", parent=self.root)

    # ========================
    # UTILITY METHODS
    # ========================
    def speak(self, text):
        if hasattr(self, 'voice_engine') and self.voice_engine:
            try:
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
            except Exception as e:
                self.add_to_log(f"TTS error: {str(e)}")

    def add_to_log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        try:
            self.status_text.config(state=NORMAL)
            self.status_text.insert(END, f"{timestamp} - {message}\n")
            self.status_text.see(END)
            self.status_text.config(state=DISABLED)
        except Exception as e:
            print(f"Error updating log: {str(e)}")

    def close_app(self):
        self.is_running = False
        
        # Stop any running threads
        if hasattr(self, 'attendance_window') and hasattr(self.attendance_window, 'running'):
            self.attendance_window.running = False
            
        if self.camera is not None:
            self.camera.release()
            self.add_to_log("Camera released")
        
        if hasattr(self, 'voice_engine') and self.voice_engine:
            self.voice_engine.stop()
        
        # Save any unsaved data
        self.save_schedule()
        self.save_label_map()
        
        self.root.destroy()

# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    root = Tk()
    
    try:
        app = ClassVision(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", f"The application encountered an error:\n{str(e)}")
        raise
