import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
from queue import Queue
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VideoSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("VISTA - Traffic Analysis System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables first
        self.selected_file = None
        self.cap = None
        self.is_playing = False
        self.stop_thread = False
        self.detection_thread = None
        self.display_thread = None
        self.current_frame = None
        self.frame_queue = Queue(maxsize=32)
        self.last_frame_time = 0
        self.total_frames = 0  # Add frame counter
        self.fps = 0
        
        # Tracking variables
        self.next_object_id = 0
        self.tracked_objects = {}  # Format: {id: {'bbox': [...], 'class': class_id, 'last_seen': time}}
        self.crossed_vehicles = set()
        self.vehicle_counts = defaultdict(int)
        self.IOU_THRESHOLD = 0.3
        
        # Traffic flow analysis
        self.flow_history = []  # List to store vehicle counts per time window
        self.last_analysis_time = time.time()
        self.analysis_interval = 60  # Analyze traffic every 60 seconds
        self.peak_threshold = 20  # Threshold for high traffic
        
        # Traffic analysis variables
        self.flow_data = {
            'timestamps': [],
            'vehicle_counts': [],
            'vehicle_types': [],
            'speed_estimates': [],
            'congestion_levels': []
        }
        self.analysis_start_time = None
        self.last_analysis_time = time.time()
        self.analysis_interval = 10  # Analyze every 10 seconds
        self.peak_threshold = 20
        self.congestion_threshold = 0.7
        
        # Speed estimation
        self.frame_height = None
        self.meters_per_pixel = 0.1  # Approximate conversion
        
        # Report generation
        self.report_data = {
            'peak_hours': [],
            'congestion_events': [],
            'vehicle_distribution': defaultdict(int),
            'average_speeds': [],
            'recommendations': set()
        }
        
        # Create output directory for reports
        self.output_dir = "traffic_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Class names mapping
        self.class_names = {
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Line crossing parameters
        self.counting_line = None
        self.line_position = 0.5  # Line at 50% of frame height
        
        # Traffic analysis thresholds
        self.congestion_levels = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        self.vehicle_ratio_thresholds = {
            'cars': 0.6,      # High car ratio threshold
            'trucks': 0.25,   # High truck ratio threshold
            'buses': 0.15,    # High bus ratio threshold
            'bikes': 0.1      # High bicycle/motorcycle ratio threshold
        }
        
        # Create main container
        self.container = ttk.Frame(self.root)
        self.container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create left panel for video
        self.left_panel = ttk.Frame(self.container)
        self.left_panel.pack(side='left', fill='both', expand=True)
        
        # Create right panel for analytics
        self.right_panel = ttk.Frame(self.container)
        self.right_panel.pack(side='right', fill='both', padx=(20,0))
        
        # Load YOLOv8 model
        try:
            self.model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            messagebox.showerror("Error", "Failed to load YOLOv8 model")
        
        # Setup UI panels
        self.setup_video_panel()
        self.setup_analytics_panel()
        self.setup_insights_panel()
        
        # Initialize previous positions dictionary
        self.previous_positions = {}
        
    def setup_video_panel(self):
        # Title
        self.title_label = ttk.Label(
            self.left_panel,
            text="VISTA - Traffic Analysis System",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=10)
        
        # Video frame
        self.video_frame = ttk.Label(self.left_panel)
        self.video_frame.pack(pady=10)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.left_panel)
        self.control_frame.pack(pady=10)
        
        # Style for buttons
        style = ttk.Style()
        style.configure('Action.TButton', padding=10)
        
        self.select_button = ttk.Button(
            self.control_frame,
            text="Select Video",
            command=self.select_video,
            style='Action.TButton'
        )
        self.select_button.pack(side='left', padx=5)
        
        self.process_button = ttk.Button(
            self.control_frame,
            text="Process Video",
            command=self.process_video,
            state='disabled',
            style='Action.TButton'
        )
        self.process_button.pack(side='left', padx=5)
        
        self.pause_button = ttk.Button(
            self.control_frame,
            text="Pause",
            command=self.toggle_pause,
            state='disabled',
            style='Action.TButton'
        )
        self.pause_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(
            self.control_frame,
            text="Stop",
            command=self.stop_video,
            state='disabled',
            style='Action.TButton'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # File info
        self.file_label = ttk.Label(
            self.left_panel,
            text="No file selected",
            wraplength=500
        )
        self.file_label.pack(pady=5)
        
        # Status bar
        self.status_frame = ttk.Frame(self.left_panel)
        self.status_frame.pack(fill='x', pady=5)
        
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)

    def setup_analytics_panel(self):
        # Analytics title
        self.analytics_title = ttk.Label(
            self.right_panel,
            text="Traffic Analytics",
            font=("Arial", 14, "bold")
        )
        self.analytics_title.pack(pady=10)
        
        # Vehicle counts frame
        self.counts_frame = ttk.LabelFrame(self.right_panel, text="Vehicle Distribution")
        self.counts_frame.pack(fill='x', pady=10, padx=5)
        
        # Create labels for each vehicle type
        self.count_labels = {}
        for class_id, vehicle_type in self.class_names.items():
            frame = ttk.Frame(self.counts_frame)
            frame.pack(fill='x', pady=2)
            label = ttk.Label(frame, text=f"{vehicle_type}:")
            label.pack(side='left', padx=5)
            count = ttk.Label(frame, text="0")
            count.pack(side='right', padx=5)
            self.count_labels[class_id] = count
        
        # Traffic flow metrics
        self.metrics_frame = ttk.LabelFrame(self.right_panel, text="Traffic Metrics")
        self.metrics_frame.pack(fill='x', pady=10, padx=5)
        
        # Total vehicles
        self.total_frame = ttk.Frame(self.metrics_frame)
        self.total_frame.pack(fill='x', pady=2)
        ttk.Label(self.total_frame, text="Total Vehicles:").pack(side='left', padx=5)
        self.total_vehicles_label = ttk.Label(self.total_frame, text="0")
        self.total_vehicles_label.pack(side='right', padx=5)
        
        # Vehicles per minute
        self.vpm_frame = ttk.Frame(self.metrics_frame)
        self.vpm_frame.pack(fill='x', pady=2)
        ttk.Label(self.vpm_frame, text="Vehicles/Minute:").pack(side='left', padx=5)
        self.vpm_label = ttk.Label(self.vpm_frame, text="0")
        self.vpm_label.pack(side='right', padx=5)
        
        # Create matplotlib figure for real-time plot
        self.fig = Figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # Initial plot setup
        self.update_plot()

    def setup_insights_panel(self):
        """Setup panel for real-time traffic insights"""
        insights_frame = ttk.LabelFrame(self.right_panel, text="Traffic Insights", padding="10")
        insights_frame.pack(fill='x', pady=10, padx=5)
        
        # Traffic status
        self.status_label = ttk.Label(
            insights_frame, 
            text="Analyzing traffic...", 
            font=('Arial', 11, 'bold')
        )
        self.status_label.pack(anchor='w', pady=5)
        
        # Current conditions
        self.conditions_label = ttk.Label(
            insights_frame,
            text="",
            font=('Arial', 10),
            wraplength=400
        )
        self.conditions_label.pack(anchor='w', pady=5)
        
        # Concerns
        concerns_frame = ttk.LabelFrame(insights_frame, text="Concerns", padding="5")
        concerns_frame.pack(fill='x', pady=5)
        self.concerns_text = tk.Text(concerns_frame, height=3, width=50, wrap=tk.WORD)
        self.concerns_text.pack(fill='x')
        
        # Recommendations
        recommendations_frame = ttk.LabelFrame(insights_frame, text="Recommendations", padding="5")
        recommendations_frame.pack(fill='x', pady=5)
        self.recommendations_text = tk.Text(recommendations_frame, height=4, width=50, wrap=tk.WORD)
        self.recommendations_text.pack(fill='x')

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def update_tracker(self, detections, detection_classes):
        """Update tracker with new detections using IoU matching"""
        current_time = time.time()
        
        # Remove old tracks
        self.tracked_objects = {
            id: info for id, info in self.tracked_objects.items()
            if current_time - info['last_seen'] < 1.0  # Remove tracks not seen for 1 second
        }
        
        # Match detections to existing tracks using IoU
        matched_tracks = {}
        unmatched_detections = list(range(len(detections)))
        
        for track_id, track_info in self.tracked_objects.items():
            best_iou = self.IOU_THRESHOLD
            best_detection = None
            
            for i in unmatched_detections:
                iou = self.calculate_iou(track_info['bbox'], detections[i][:4])
                if iou > best_iou:
                    best_iou = iou
                    best_detection = i
            
            if best_detection is not None:
                matched_tracks[track_id] = {
                    'bbox': detections[best_detection][:4],
                    'class': detection_classes[best_detection],
                    'last_seen': current_time
                }
                unmatched_detections.remove(best_detection)
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            new_id = self.next_object_id
            self.next_object_id += 1
            matched_tracks[new_id] = {
                'bbox': detections[i][:4],
                'class': detection_classes[i],
                'last_seen': current_time
            }
        
        self.tracked_objects = matched_tracks
        return self.tracked_objects

    def process_frames(self):
        """Process video frames with vehicle tracking and line crossing detection"""
        self.analysis_start_time = time.time()
        frame_count = 0
        last_time = time.time()
        
        while self.is_playing and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.generate_traffic_report()
                break
            
            # Set up counting line if not already done
            if self.counting_line is None:
                height, width = frame.shape[:2]
                self.counting_line = [(0, int(height * self.line_position)), 
                                    (width, int(height * self.line_position))]
            
            # Store frame height for speed estimation
            if self.frame_height is None:
                self.frame_height = frame.shape[0]
            
            # Perform detection
            results = self.model(frame, classes=[1, 2, 3, 5, 7])
            
            # Convert detections to format for tracker
            detections = []
            detection_classes = {}
            
            # Process all detections from the model
            for i, r in enumerate(results[0].boxes.data):
                x1, y1, x2, y2, conf, cls = r
                if conf > 0.5:  # Only track confident detections
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                    detection_classes[i] = int(cls)
            
            # Process detections and update tracking
            annotated_frame = frame.copy()
            
            if len(detections) > 0:
                # Update tracker
                tracked_objects = self.update_tracker(detections, detection_classes)
                
                # Process tracked objects
                for track_id, track_info in tracked_objects.items():
                    bbox = track_info['bbox']
                    class_id = track_info['class']
                    
                    # Check line crossing
                    if track_id not in self.crossed_vehicles:
                        self.check_line_crossing(track_id, bbox, class_id)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) if track_id in self.crossed_vehicles else (0, 0, 255)
                    thickness = 3 if track_id in self.crossed_vehicles else 2
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label with ID and class
                    class_name = self.class_names.get(class_id, 'Unknown')
                    label = f"ID:{track_id} {class_name}"
                    if track_id in self.crossed_vehicles:
                        label += " "
                    
                    # Draw label background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
            
            # Draw counting line
            if self.counting_line:
                cv2.line(
                    annotated_frame,
                    self.counting_line[0],
                    self.counting_line[1],
                    (0, 255, 0),
                    2
                )
            
            # Add frame to queue
            if not self.frame_queue.full():
                self.frame_queue.put((annotated_frame, len(self.crossed_vehicles)))
            
            # Update analytics and insights
            self.analyze_traffic_patterns()
            
            frame_count += 1
        
        self.cap.release()

    def check_line_crossing(self, track_id, bbox, class_id):
        """Check if a vehicle has crossed the counting line"""
        if self.counting_line is None:
            return

        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        line_y = self.counting_line[0][1]

        # Get previous position
        prev_center_y = self.previous_positions.get(track_id, center_y)
        
        # Check if vehicle has crossed the line from top to bottom
        if prev_center_y <= line_y and center_y > line_y:
            if track_id not in self.crossed_vehicles:
                self.crossed_vehicles.add(track_id)
                # Only increment count when vehicle crosses the line
                if class_id in self.class_names:
                    self.vehicle_counts[class_id] += 1
                    self.update_analytics_display()

        # Update previous position
        self.previous_positions[track_id] = center_y

    def select_video(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=filetypes
        )
        
        if filename:
            self.selected_file = filename
            self.file_label.config(text=f"Selected: {os.path.basename(filename)}")
            self.process_button.config(state='normal')
            self.stop_video()
            
            # Reset analytics
            self.vehicle_counts.clear()
            self.crossed_vehicles.clear()
            self.tracked_objects.clear()
            self.next_object_id = 0
            self.update_plot()

    def process_video(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a video file first")
            return
            
        try:
            self.stop_video()
            self.cap = cv2.VideoCapture(self.selected_file)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open the video file")
                return
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = int(1000 / self.fps)
            
            # Enable control buttons
            self.pause_button.config(state='normal')
            self.stop_button.config(state='normal')
            
            self.is_playing = True
            self.paused = False
            
            # Start processing thread
            self.detection_thread = threading.Thread(target=self.process_frames)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Start display update
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing video: {str(e)}")

    def update_frame(self):
        if self.is_playing:
            current_time = time.time()
            
            if current_time - self.last_frame_time >= 1.0/self.fps:
                if not self.frame_queue.empty() and not self.paused:
                    frame, vehicle_count = self.frame_queue.get()
                    
                    # Calculate FPS
                    fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time > 0 else 0
                    self.last_frame_time = current_time
                    
                    # Update status
                    self.status_label.config(
                        text=f"Processing: {fps:.1f} FPS | Current Frame: {self.total_frames}"
                    )
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    self.video_frame.config(image=photo)
                    self.video_frame.image = photo
            
            # Schedule next update
            self.root.after(max(1, self.frame_delay), self.update_frame)
        else:
            if not self.paused:
                self.stop_video()
                messagebox.showinfo("Info", "Video playback completed")
                
                # Show final analytics
                total = sum(self.vehicle_counts.values())
                analysis = "Traffic Analysis Summary:\n\n"
                for class_id, vehicle_type in self.class_names.items():
                    count = self.vehicle_counts[class_id]
                    percentage = (count / total * 100) if total > 0 else 0
                    analysis += f"{vehicle_type}: {count} ({percentage:.1f}%)\n"
                
                if self.fps and self.total_frames:
                    minutes = self.total_frames / (self.fps * 60)
                    vpm = total / minutes if minutes > 0 else 0
                    analysis += f"\nAverage Traffic Flow: {vpm:.1f} vehicles/minute"
                
                messagebox.showinfo("Analysis Results", analysis)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def stop_video(self):
        self.is_playing = False
        self.paused = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        self.video_frame.config(image='')
        self.pause_button.config(state='disabled', text="Pause")
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Ready")
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join()

    def setup_counting_line(self, height, width):
        self.counting_line = [(0, int(height * self.line_position)), (width, int(height * self.line_position))]

    def update_analytics_display(self):
        """Update analytics display with current vehicle counts"""
        # Update total vehicles count (only crossed vehicles)
        total_count = len(self.crossed_vehicles)
        self.total_vehicles_label.config(text=str(total_count))
        
        # Update individual vehicle type counts
        for class_id in self.class_names:
            count = self.vehicle_counts.get(class_id, 0)
            if class_id in self.count_labels:
                self.count_labels[class_id].config(text=str(count))
        
        # Update plot
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        
        # Create bar chart of vehicle distribution
        vehicle_types = []
        counts = []
        for class_id, vehicle_type in self.class_names.items():
            vehicle_types.append(vehicle_type)
            counts.append(self.vehicle_counts[class_id])
        
        self.ax.bar(vehicle_types, counts)
        self.ax.set_title('Vehicle Distribution')
        self.ax.tick_params(axis='x', rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()

    def analyze_traffic_patterns(self):
        """Analyze current traffic patterns and generate insights"""
        if not self.vehicle_counts:
            return
        
        total_vehicles = sum(self.vehicle_counts.values())
        if total_vehicles == 0:
            return
        
        # Calculate vehicle type ratios
        ratios = {
            'cars': self.vehicle_counts[2] / total_vehicles,
            'trucks': self.vehicle_counts[7] / total_vehicles,
            'buses': self.vehicle_counts[5] / total_vehicles,
            'bikes': (self.vehicle_counts[1] + self.vehicle_counts[3]) / total_vehicles  # bicycles + motorcycles
        }
        
        # Calculate average congestion level from recent history
        recent_congestion = self.flow_data['congestion_levels'][-5:] if self.flow_data['congestion_levels'] else [0]
        avg_congestion = sum(recent_congestion) / len(recent_congestion)
        
        # Determine traffic status
        if avg_congestion >= self.congestion_levels['high']:
            status = "SEVERE CONGESTION"
            status_color = "red"
        elif avg_congestion >= self.congestion_levels['medium']:
            status = "MODERATE CONGESTION"
            status_color = "orange"
        elif avg_congestion >= self.congestion_levels['low']:
            status = "LIGHT CONGESTION"
            status_color = "yellow"
        else:
            status = "NORMAL FLOW"
            status_color = "green"
        
        # Update status display
        self.status_label.config(
            text=f"Traffic Status: {status}",
            foreground=status_color
        )
        
        # Generate current conditions text
        conditions = []
        if ratios['cars'] > self.vehicle_ratio_thresholds['cars']:
            conditions.append("High passenger vehicle volume")
        if ratios['trucks'] > self.vehicle_ratio_thresholds['trucks']:
            conditions.append("Significant heavy vehicle presence")
        if ratios['buses'] > self.vehicle_ratio_thresholds['buses']:
            conditions.append("High public transport activity")
        if ratios['bikes'] > self.vehicle_ratio_thresholds['bikes']:
            conditions.append("Notable bicycle/motorcycle presence")
        
        self.conditions_label.config(
            text="Current Conditions:\n• " + "\n• ".join(conditions) if conditions else "Normal traffic conditions"
        )
        
        # Generate concerns
        concerns = []
        if avg_congestion >= self.congestion_levels['high']:
            concerns.append("⚠️ Severe congestion may lead to significant delays")
        if ratios['trucks'] > self.vehicle_ratio_thresholds['trucks']:
            concerns.append("🚛 High heavy vehicle volume may impact road wear")
            concerns.append("🚦 Reduced visibility for other vehicles due to truck density")
        if ratios['bikes'] > self.vehicle_ratio_thresholds['bikes'] and avg_congestion > self.congestion_levels['medium']:
            concerns.append("🚲 Cyclist safety at risk during high congestion")
            concerns.append("🚸 Increased risk of accidents in mixed traffic")
        if ratios['buses'] < self.vehicle_ratio_thresholds['buses'] and avg_congestion > self.congestion_levels['medium']:
            concerns.append("🚌 Limited public transport during high congestion")
        
        # Vehicle-specific concerns
        if ratios['cars'] > 0.7:  # More than 70% cars
            concerns.append("🚗 High private vehicle concentration may lead to parking issues")
        if ratios['trucks'] + ratios['buses'] > 0.4:  # More than 40% heavy vehicles
            concerns.append("⚠️ High proportion of heavy vehicles affecting traffic speed")
        if ratios['bikes'] > 0.2:  # More than 20% bikes
            concerns.append("🚲 High vulnerability risk for cyclists in mixed traffic")
        
        self.concerns_text.delete(1.0, tk.END)
        self.concerns_text.insert(tk.END, "\n".join(concerns) if concerns else "No immediate concerns")
        
        # Generate recommendations
        recommendations = []
        if avg_congestion >= self.congestion_levels['high']:
            recommendations.append("⚡ Implement immediate congestion management measures")
            recommendations.append("🚥 Optimize traffic signal timing")
            if ratios['cars'] > self.vehicle_ratio_thresholds['cars']:
                recommendations.append("🚗 Encourage use of alternative routes")
                recommendations.append("🅿️ Consider park-and-ride facilities")
        
        if ratios['trucks'] > self.vehicle_ratio_thresholds['trucks']:
            recommendations.append("🛣️ Consider dedicated heavy vehicle lanes")
            recommendations.append("⏰ Implement time-based restrictions for heavy vehicles")
            recommendations.append("📍 Designate specific routes for heavy vehicles")
            recommendations.append("🛑 Install weight monitoring systems")
        
        if ratios['bikes'] > self.vehicle_ratio_thresholds['bikes']:
            recommendations.append("🚲 Implement protected bicycle lanes")
            recommendations.append("🚦 Install bicycle-specific traffic signals")
            if avg_congestion > self.congestion_levels['medium']:
                recommendations.append("⚠️ Add additional cyclist safety measures")
                recommendations.append("🚸 Install bike boxes at intersections")
        
        if ratios['buses'] < self.vehicle_ratio_thresholds['buses'] and avg_congestion > self.congestion_levels['medium']:
            recommendations.append("🚌 Increase public transport frequency")
            recommendations.append("🛑 Add dedicated bus lanes")
            recommendations.append("🚏 Optimize bus stop locations")
            recommendations.append("📱 Implement real-time bus tracking")

        # Vehicle mix-based recommendations
        if ratios['cars'] > 0.7:
            recommendations.append("🅿️ Expand parking facilities")
            recommendations.append("🚗 Promote carpooling initiatives")
        if ratios['trucks'] + ratios['buses'] > 0.4:
            recommendations.append("🛣️ Strengthen road infrastructure")
            recommendations.append("🚦 Adjust signal timing for heavy vehicles")
        if ratios['bikes'] > 0.2:
            recommendations.append("🚲 Create bike-sharing stations")
            recommendations.append("🛣️ Implement traffic calming measures")
        
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, "\n".join(recommendations) if recommendations else "No specific recommendations at this time")

    def update_traffic_analysis(self, frame_count):
        """Update traffic analysis data"""
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return
            
        # ... (existing analysis code) ...
        
        # Update insights
        self.analyze_traffic_patterns()
        
        self.last_analysis_time = current_time

    def estimate_vehicle_speeds(self):
        """Estimate vehicle speeds based on frame-to-frame movement"""
        speeds = []
        current_time = time.time()
        
        for track_id, track_info in self.tracked_objects.items():
            if 'prev_pos' in track_info and 'prev_time' in track_info:
                # Calculate distance moved in pixels
                prev_y = track_info['prev_pos'][1]
                curr_y = track_info['bbox'][1]
                pixels_moved = abs(curr_y - prev_y)
                
                # Convert to meters and calculate speed
                meters_moved = pixels_moved * self.meters_per_pixel
                time_diff = current_time - track_info['prev_time']
                
                if time_diff > 0:
                    speed = (meters_moved / time_diff) * 3.6  # Convert to km/h
                    speeds.append(speed)
        
        return speeds

    def update_recommendations(self, flow_rate, congestion_level, avg_speed):
        """Update traffic recommendations based on current conditions"""
        if congestion_level > self.congestion_threshold:
            self.report_data['recommendations'].add(
                "⚠️ Consider implementing traffic management measures during peak hours"
            )
        
        if avg_speed < 20:  # km/h
            self.report_data['recommendations'].add(
                "🚗 Low average speed detected. Consider optimizing traffic signal timing"
            )
        
        heavy_vehicles = self.vehicle_counts[5] + self.vehicle_counts[7]  # buses + trucks
        total_vehicles = sum(self.vehicle_counts.values())
        if total_vehicles > 0 and heavy_vehicles / total_vehicles > 0.3:
            self.report_data['recommendations'].add(
                "🚛 High proportion of heavy vehicles. Consider dedicated lanes or time restrictions"
            )

    def generate_traffic_report(self):
        """Generate comprehensive traffic analysis report"""
        if not self.analysis_start_time:
            return
        
        # Create report filename with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.output_dir, f'traffic_report_{timestamp}.html')
        
        # Calculate overall statistics
        total_duration = (time.time() - self.analysis_start_time) / 3600  # hours
        total_vehicles = sum(self.flow_data['vehicle_counts'])
        avg_flow_rate = total_vehicles / total_duration if total_duration > 0 else 0
        peak_congestion = max(self.flow_data['congestion_levels']) if self.flow_data['congestion_levels'] else 0
        avg_speed = sum(self.flow_data['speed_estimates']) / len(self.flow_data['speed_estimates']) if self.flow_data['speed_estimates'] else 0
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Traffic Analysis Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
                .highlight {{ color: #d63031; }}
            </style>
        </head>
        <body>
            <h1>Traffic Analysis Report</h1>
            <div class="section">
                <h2>Overview</h2>
                <p>Analysis Duration: {total_duration:.1f} hours</p>
                <p>Total Vehicles: {total_vehicles}</p>
                <p>Average Flow Rate: {avg_flow_rate:.1f} vehicles/hour</p>
                <p>Peak Congestion Level: {peak_congestion:.2f}</p>
                <p>Average Speed: {avg_speed:.1f} km/h</p>
            </div>
            
            <div class="section">
                <h2>Peak Hours</h2>
                <ul>
                    {''.join(f'<li>{time}</li>' for time in self.report_data['peak_hours'][-5:])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Vehicle Distribution</h2>
                <ul>
                    {''.join(f'<li>{self.class_names[type]}: {count}</li>' for type, count in self.report_data['vehicle_distribution'].items())}
                </ul>
            </div>
            
            <div class="section">
                <h2>Congestion Events</h2>
                <ul>
                    {''.join(f'<li>{event["time"]} - Level: {event["level"]:.2f}</li>' for event in self.report_data['congestion_events'][-5:])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in self.report_data['recommendations'])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Show report path in UI
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, f"Traffic report generated: {report_file}\n\n")
        self.recommendations_text.insert(tk.END, "Key Findings:\n")
        self.recommendations_text.insert(tk.END, f"• Total Vehicles: {total_vehicles}\n")
        self.recommendations_text.insert(tk.END, f"• Average Flow: {avg_flow_rate:.1f} vehicles/hour\n")
        self.recommendations_text.insert(tk.END, f"• Peak Congestion: {peak_congestion:.2f}\n")

def main():
    root = tk.Tk()
    app = VideoSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
