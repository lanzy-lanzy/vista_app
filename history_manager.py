import os
import json
import datetime
from collections import defaultdict

class HistoryManager:
    def __init__(self, history_dir="detection_history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
        self.current_session = None
        self.session_data = defaultdict(list)
        
    def start_session(self):
        """Start a new detection session"""
        self.current_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data.clear()
        return self.current_session
    
    def add_detection(self, timestamp, vehicle_counts, vehicle_speeds=None, congestion_level=None):
        """Add a detection record to the current session"""
        if not self.current_session:
            self.start_session()
            
        self.session_data["timestamps"].append(timestamp)
        self.session_data["vehicle_counts"].append(vehicle_counts)
        if vehicle_speeds:
            self.session_data["vehicle_speeds"].append(vehicle_speeds)
        if congestion_level is not None:
            self.session_data["congestion_levels"].append(congestion_level)
    
    def end_session(self, metadata=None):
        """End the current session and save data"""
        if not self.current_session:
            return
            
        session_file = os.path.join(self.history_dir, f"session_{self.current_session}.json")
        
        data = {
            "session_id": self.current_session,
            "metadata": metadata or {},
            "data": dict(self.session_data)
        }
        
        with open(session_file, "w") as f:
            json.dump(data, f, indent=4)
            
        self.current_session = None
        self.session_data.clear()
        
    def get_session_history(self, limit=None):
        """Get list of all detection sessions"""
        sessions = []
        for filename in os.listdir(self.history_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.history_dir, filename)
                with open(file_path, "r") as f:
                    session_data = json.load(f)
                sessions.append(session_data)
                
        # Sort by session_id (timestamp) in descending order
        sessions.sort(key=lambda x: x["session_id"], reverse=True)
        
        if limit:
            sessions = sessions[:limit]
            
        return sessions
    
    def get_session_data(self, session_id):
        """Get data for a specific session"""
        file_path = os.path.join(self.history_dir, f"session_{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
    
    def get_historical_patterns(self):
        """Analyze historical patterns across all sessions"""
        sessions = self.get_session_history()
        
        patterns = {
            "peak_hours": defaultdict(int),
            "vehicle_distribution": defaultdict(int),
            "average_congestion": [],
            "busiest_days": defaultdict(int)
        }
        
        for session in sessions:
            session_date = session["session_id"][:8]  # YYYYMMDD
            patterns["busiest_days"][session_date] += sum(session["data"]["vehicle_counts"])
            
            for timestamp, counts in zip(session["data"]["timestamps"], session["data"]["vehicle_counts"]):
                hour = datetime.datetime.fromisoformat(timestamp).hour
                patterns["peak_hours"][hour] += sum(counts.values())
                
                for vehicle_type, count in counts.items():
                    patterns["vehicle_distribution"][vehicle_type] += count
            
            if "congestion_levels" in session["data"]:
                patterns["average_congestion"].extend(session["data"]["congestion_levels"])
        
        return patterns
