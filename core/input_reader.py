import cv2
import threading
import queue
import time


class InputReader:
    """
    Threaded video reader for RTSP/Webcam/File with buffering.
    Prevents frame drops and improves performance.
    """

    def __init__(self, config):
        """
        Args:
            config: Dictionary with:
                - source: Video source (RTSP URL, webcam index, or file path)
                - buffer_size: Frame buffer size (default: 1)
                - reconnect: Auto-reconnect for RTSP streams (default: True)
        """
        self.source = config.get("source", 0)
        self.buffer_size = config.get("buffer_size", 1)
        self.reconnect = config.get("reconnect", True)
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.thread = None
        
        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Initialize capture
        self._init_capture()
        
        # Start reading thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _init_capture(self):
        """Initialize video capture."""
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # RTSP stream
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Webcam or file
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
        
        # Get FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Default FPS

    def _reader(self):
        """Background thread for reading frames."""
        last_time = time.time()
        
        while not self.stopped:
            if not self.cap.isOpened():
                if self.reconnect and isinstance(self.source, str):
                    print(f"⚠️  Reconnecting to {self.source}...")
                    time.sleep(1)
                    self._init_capture()
                    continue
                else:
                    break
            
            ret, frame = self.cap.read()
            
            if not ret:
                if self.reconnect and isinstance(self.source, str):
                    print(f"⚠️  Lost connection, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self._init_capture()
                    continue
                else:
                    self.stopped = True
                    break
            
            self.frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_time >= 1.0:
                self.fps = self.frame_count / (current_time - last_time)
                self.frame_count = 0
                last_time = current_time
            
            # Add to queue (drop old frames if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                self.dropped_frames += 1

    def get_frame(self):
        """
        Get the latest frame.
        
        Returns:
            Frame or None if stopped
        """
        if self.stopped and self.frame_queue.empty():
            return None
        
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def get_fps(self):
        """Get current FPS."""
        return self.fps

    def get_stats(self):
        """Get reader statistics."""
        return {
            "fps": self.fps,
            "dropped_frames": self.dropped_frames,
            "queue_size": self.frame_queue.qsize()
        }

    def stop(self):
        """Stop the reader thread."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()