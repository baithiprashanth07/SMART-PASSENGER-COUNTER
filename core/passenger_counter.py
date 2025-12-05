import numpy as np
from collections import defaultdict


class PassengerCounter:
    """
    Simple line-crossing counter for IN/OUT passenger tracking.
    Supports multiple counting lines with directional logic.
    """

    def __init__(self, lines=None):
        """
        Args:
            lines: List of line definitions, each as:
                   {"name": "entry", "coords": [x1, y1, x2, y2], "direction": "vertical"}
        """
        if lines is None:
            # Default: single horizontal line in middle of frame
            self.lines = [
                {
                    "name": "entry",
                    "coords": [0, 360, 1280, 360],  # Horizontal line
                    "direction": "vertical"  # Crossing vertically (up/down)
                }
            ]
        else:
            self.lines = lines
        
        # Track object positions
        self.track_positions = {}  # track_id -> last_position
        self.track_states = {}     # track_id -> {"crossed": bool, "direction": str}
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        self.total_crossed = set()

    def update(self, boxes, track_ids=None):
        """
        Update counter with new detections.
        
        Args:
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            track_ids: Optional list of track IDs corresponding to boxes
        
        Returns:
            (in_count, out_count) - number of new crossings this frame
        """
        new_in = 0
        new_out = 0
        
        # If no track IDs provided, use box indices
        if track_ids is None:
            track_ids = list(range(len(boxes)))
        
        for track_id, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check each counting line
            for line in self.lines:
                lx1, ly1, lx2, ly2 = line["coords"]
                direction = line.get("direction", "vertical")
                
                # Get previous position
                prev_pos = self.track_positions.get(track_id)
                
                if prev_pos is not None:
                    prev_x, prev_y = prev_pos
                    
                    # Check for line crossing
                    if direction == "vertical":
                        # Horizontal line, vertical crossing
                        line_y = ly1  # Assume horizontal line
                        
                        # Check if crossed from top to bottom (IN)
                        if prev_y < line_y and center_y >= line_y:
                            if track_id not in self.total_crossed:
                                self.in_count += 1
                                new_in += 1
                                self.total_crossed.add(track_id)
                                self.track_states[track_id] = {"crossed": True, "direction": "in"}
                        
                        # Check if crossed from bottom to top (OUT)
                        elif prev_y > line_y and center_y <= line_y:
                            if track_id not in self.total_crossed:
                                self.out_count += 1
                                new_out += 1
                                self.total_crossed.add(track_id)
                                self.track_states[track_id] = {"crossed": True, "direction": "out"}
                    
                    elif direction == "horizontal":
                        # Vertical line, horizontal crossing
                        line_x = lx1  # Assume vertical line
                        
                        # Check if crossed from left to right (IN)
                        if prev_x < line_x and center_x >= line_x:
                            if track_id not in self.total_crossed:
                                self.in_count += 1
                                new_in += 1
                                self.total_crossed.add(track_id)
                                self.track_states[track_id] = {"crossed": True, "direction": "in"}
                        
                        # Check if crossed from right to left (OUT)
                        elif prev_x > line_x and center_x <= line_x:
                            if track_id not in self.total_crossed:
                                self.out_count += 1
                                new_out += 1
                                self.total_crossed.add(track_id)
                                self.track_states[track_id] = {"crossed": True, "direction": "out"}
                
                # Update position
                self.track_positions[track_id] = (center_x, center_y)
        
        return new_in, new_out

    def get_counts(self):
        """Return current counts."""
        return {
            "in": self.in_count,
            "out": self.out_count,
            "total": self.in_count - self.out_count
        }

    def reset(self):
        """Reset all counters."""
        self.in_count = 0
        self.out_count = 0
        self.total_crossed.clear()
        self.track_positions.clear()
        self.track_states.clear()
