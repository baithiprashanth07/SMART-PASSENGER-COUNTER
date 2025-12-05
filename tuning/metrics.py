class MetricsEvaluator:
    def __init__(self):
        self.track_history = {}
        self.id_switches = 0

    def update(self, tracked_objects):
        for tid, obj in tracked_objects.items():
            if tid not in self.track_history:
                self.track_history[tid] = obj.bbox
            else:
                prev = self.track_history[tid]
                curr = obj.bbox

                # Detect ID switches
                if abs(prev[0] - curr[0]) > 150 or abs(prev[1] - curr[1]) > 150:
                    self.id_switches += 1

                self.track_history[tid] = curr

    def get_score(self):
        if len(self.track_history) == 0:
            return 0
        score = 1 - (self.id_switches / (len(self.track_history) + 1))
        return max(score, 0)
