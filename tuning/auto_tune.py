import yaml
import time
import cv2
import numpy as np
from tuning.metrics import MetricsEvaluator
from core.detection_tracker import DetectionTracker
from core.directional_counter_multi import MultiDoorCounter
from core.input_reader import InputReader


class AutoTuner:
    def __init__(self, base_config="config/config.yaml", sample_video="sample.mp4"):
        self.base_config = yaml.safe_load(open(base_config, "r"))
        self.sample_video = sample_video
        self.metrics = MetricsEvaluator()

        # Parameter search space
        self.params = {
            "sort": {
                "max_age": [10, 15, 20, 30],
                "min_hits": [1, 2, 3],
                "iou_threshold": [0.1, 0.2, 0.3]
            },
            "yolo": {
                "conf": [0.3, 0.4, 0.5],
                "iou": [0.45, 0.5, 0.6]
            },
            "reid": {
                "similarity": [0.45, 0.55, 0.65],
                "every_n_frames": [3, 5, 7]
            }
        }

    # -----------------------------------------------------
    # Test run
    # -----------------------------------------------------
    def run_test(self, cfg):
        reader = InputReader({"source": self.sample_video})
        detector = DetectionTracker(cfg)
        counter = MultiDoorCounter(cfg["counting"]["doors"])

        frame_count = 0
        start = time.time()

        while True:
            frame = reader.get_frame()
            if frame is None:
                break

            tracked = detector.update(frame)
            counter.update(tracked)

            # Tracking metrics
            self.metrics.update(tracked)

            frame_count += 1
            if frame_count > 600:   # Limit ~20 seconds
                break

        fps = frame_count / (time.time() - start)
        score = self.metrics.get_score()

        return score, fps, counter.counts

    # -----------------------------------------------------
    # Tuning loop
    # -----------------------------------------------------
    def tune(self):
        print("\n🔧 Starting Auto-Tuning…\n")

        best_config = None
        best_score = -1
        best_fps = 0

        for max_age in self.params["sort"]["max_age"]:
            for min_hits in self.params["sort"]["min_hits"]:
                for iou_t in self.params["sort"]["iou_threshold"]:
                    for yc in self.params["yolo"]["conf"]:
                        for yi in self.params["yolo"]["iou"]:
                            for rs in self.params["reid"]["similarity"]:
                                for rn in self.params["reid"]["every_n_frames"]:

                                    cfg = self._apply_params(
                                        max_age, min_hits, iou_t,
                                        yc, yi,
                                        rs, rn
                                    )

                                    print(f"Testing: SORT({max_age},{min_hits},{iou_t}) "
                                          f"YOLO({yc},{yi}) ReID({rs},{rn})")

                                    score, fps, counts = self.run_test(cfg)
                                    fitness = score * 0.7 + fps * 0.3

                                    if fitness > best_score:
                                        best_score = fitness
                                        best_config = cfg
                                        best_fps = fps

                                        print(f"🌟 NEW BEST → Score:{score:.2f}  FPS:{fps:.1f}")

        # Save result
        yaml.dump(best_config, open("config/config_optimized.yaml", "w"))
        print("\n🎉 AUTO-TUNING COMPLETE!")
        print(f"Best FPS: {best_fps:.1f}")
        print("Saved: config/config_optimized.yaml")

        return best_config

    # -----------------------------------------------------
    def _apply_params(self, max_age, min_hits, iou_t, yc, yi, rs, rn):
        cfg = self.base_config.copy()

        cfg["tracking"]["max_age"] = max_age
        cfg["tracking"]["min_hits"] = min_hits
        cfg["tracking"]["iou_threshold"] = iou_t

        cfg["yolo"]["conf"] = yc
        cfg["yolo"]["iou"] = yi

        cfg["reid"]["similarity"] = rs
        cfg["reid"]["every_n_frames"] = rn

        return cfg


if __name__ == "__main__":
    tuner = AutoTuner()
    tuner.tune()
