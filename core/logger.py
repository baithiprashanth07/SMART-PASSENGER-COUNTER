import csv
import json
import os
from datetime import datetime
from pathlib import Path


class Logger:
    """
    CSV/JSON/database logger for tracking events and analytics.
    """

    def __init__(self, log_file="logs/passenger_log.csv", log_format="csv"):
        """
        Args:
            log_file: Path to log file
            log_format: "csv" or "json"
        """
        self.log_file = log_file
        self.log_format = log_format
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file with headers
        if log_format == "csv" and not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_count",
                    "detections",
                    "in_count",
                    "out_count",
                    "total_count"
                ])
        
        self.frame_count = 0

    def log(self, data):
        """
        Log data entry.
        
        Args:
            data: Dictionary containing:
                - detections: List of detection boxes
                - IN: Number of entries
                - OUT: Number of exits
                - total: Current occupancy
        """
        self.frame_count += 1
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "frame_count": self.frame_count,
            "detections": len(data.get("detections", [])),
            "in_count": data.get("IN", 0),
            "out_count": data.get("OUT", 0),
            "total_count": data.get("total", 0)
        }
        
        if self.log_format == "csv":
            self._log_csv(entry)
        elif self.log_format == "json":
            self._log_json(entry)

    def _log_csv(self, entry):
        """Append entry to CSV file."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                entry["timestamp"],
                entry["frame_count"],
                entry["detections"],
                entry["in_count"],
                entry["out_count"],
                entry["total_count"]
            ])

    def _log_json(self, entry):
        """Append entry to JSON file."""
        # Read existing data
        data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        
        # Append new entry
        data.append(entry)
        
        # Write back
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def log_event(self, event_type, details):
        """
        Log a specific event.
        
        Args:
            event_type: Type of event (e.g., "entry", "exit", "alert")
            details: Event details dictionary
        """
        timestamp = datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details
        }
        
        # Log to separate events file
        events_file = self.log_file.replace(".csv", "_events.json").replace(".json", "_events.json")
        
        events = []
        if os.path.exists(events_file):
            try:
                with open(events_file, 'r') as f:
                    events = json.load(f)
            except json.JSONDecodeError:
                events = []
        
        events.append(event)
        
        with open(events_file, 'w') as f:
            json.dump(events, f, indent=2)

    def get_summary(self):
        """
        Get summary statistics from logs.
        
        Returns:
            Dictionary with summary statistics
        """
        if not os.path.exists(self.log_file):
            return {}
        
        if self.log_format == "csv":
            return self._get_csv_summary()
        elif self.log_format == "json":
            return self._get_json_summary()

    def _get_csv_summary(self):
        """Get summary from CSV logs."""
        total_in = 0
        total_out = 0
        max_occupancy = 0
        
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_in += int(row.get("in_count", 0))
                total_out += int(row.get("out_count", 0))
                max_occupancy = max(max_occupancy, int(row.get("total_count", 0)))
        
        return {
            "total_entries": total_in,
            "total_exits": total_out,
            "max_occupancy": max_occupancy,
            "current_occupancy": total_in - total_out
        }

    def _get_json_summary(self):
        """Get summary from JSON logs."""
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return {}
        
        total_in = sum(entry.get("in_count", 0) for entry in data)
        total_out = sum(entry.get("out_count", 0) for entry in data)
        max_occupancy = max(entry.get("total_count", 0) for entry in data)
        
        return {
            "total_entries": total_in,
            "total_exits": total_out,
            "max_occupancy": max_occupancy,
            "current_occupancy": total_in - total_out
        }