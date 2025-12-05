import time
from collections import deque


class LineCrossState:
    """
    Track a person's state for each door:
        - side_A_history
        - side_B_history
        - last_side
        - has_counted_entry
        - has_counted_exit
    """
    def __init__(self):
        self.side_A_history = deque(maxlen=7)
        self.side_B_history = deque(maxlen=7)
        self.last_side = None
        self.has_counted_entry = False
        self.has_counted_exit = False


class MultiDoorCounter:
    """
    Multi-door passenger counter.
    Handles:
        - enter
        - exit
        - per-door occupancy
        - line crossing logic
    """

    def __init__(self, doors_config):
        """
        doors_config example:
        {
            "door1": {
                "line_A": [200,300, 800,300],
                "line_B": [200,500, 800,500]
            },
            "door2": {
                ...
            }
        }
        """

        self.doors = doors_config
        self.track_states = {}     # track_id → {door_name → LineCrossState}
        self.counts = {}           # door_name → {enter, exit, occupancy}

        # Initialize counters
        for d in doors_config:
            self.counts[d] = {
                "enter": 0,
                "exit": 0,
                "occupancy": 0
            }

    # -------------------------------------------------------------
    # Helper: Check which side of the line the point is on
    # -------------------------------------------------------------
    @staticmethod
    def point_side(x, y, x1, y1, x2, y2):
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    # -------------------------------------------------------------
    # Process each tracked person for each door
    # -------------------------------------------------------------
    def update(self, tracked_objects):
        """
        tracked_objects = {id → TrackedObject}
        Returns updated door counters.
        """
        for tid, obj in tracked_objects.items():

            if tid not in self.track_states:
                # Create state for each door
                self.track_states[tid] = {
                    door: LineCrossState() for door in self.doors
                }

            center_x = (obj.bbox[0] + obj.bbox[2]) / 2
            center_y = (obj.bbox[1] + obj.bbox[3]) / 2

            # Evaluate each door
            for door_name, dcfg in self.doors.items():
                state = self.track_states[tid][door_name]

                # Extract door lines
                Ax1, Ay1, Ax2, Ay2 = dcfg["line_A"]
                Bx1, By1, Bx2, By2 = dcfg["line_B"]

                # Compute sides
                sideA = self.point_side(center_x, center_y, Ax1, Ay1, Ax2, Ay2)
                sideB = self.point_side(center_x, center_y, Bx1, By1, Bx2, By2)

                state.side_A_history.append(sideA)
                state.side_B_history.append(sideB)

                # Smooth the signal
                avgA = sum(1 for v in state.side_A_history if v >= 0) / len(state.side_A_history)
                avgB = sum(1 for v in state.side_B_history if v >= 0) / len(state.side_B_history)

                # Determine logical side
                logical_A = "pos" if avgA > 0.5 else "neg"
                logical_B = "pos" if avgB > 0.5 else "neg"

                current_side = f"A-{logical_A}-B-{logical_B}"

                # ----------------------------------------
                # Detect crossing
                # ----------------------------------------
                if state.last_side is None:
                    state.last_side = current_side
                    continue

                prev = state.last_side
                now = current_side

                # Detect Enter (A → B)
                if "A-pos" in prev and "A-neg" in now and not state.has_counted_entry:
                    self.counts[door_name]["enter"] += 1
                    self.counts[door_name]["occupancy"] += 1
                    state.has_counted_entry = True
                    obj.door_event = "enter"

                # Detect Exit (B-pos → B-neg)
                if "B-pos" in prev and "B-neg" in now and not state.has_counted_exit:
                    self.counts[door_name]["exit"] += 1
                    self.counts[door_name]["occupancy"] -= 1
                    state.has_counted_exit = True
                    obj.door_event = "exit"

                state.last_side = now

        return self.counts
