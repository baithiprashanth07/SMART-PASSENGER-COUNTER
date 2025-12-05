from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Smart Passenger Counter - Project Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'{num}. {label}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

def create_report():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Abstract
    pdf.chapter_title('1', 'Abstract')
    abstract_text = (
        "This project implements a Smart Passenger Counter system designed to accurately track and count "
        "people entering and exiting a designated area in real-time. By leveraging computer vision techniques, specifically "
        "YOLO (You Only Look Once) for object detection and SORT (Simple Online and Realtime Tracking) for object tracking, "
        "the system provides a robust solution for automated passenger analytics. The system accounts for bi-directional flow, "
        "detecting whether a person is entering or leaving, and provides live analytics via a web interface."
    )
    pdf.chapter_body(abstract_text)

    # 2. Introduction
    pdf.chapter_title('2', 'Introduction')
    intro_text = (
        "Automated passenger counting is critical for public transportation, retail analytics, and venue management. "
        "Traditional methods often rely on hardware sensors which can be expensive and inflexible. This project presents a "
        "software-based approach using standard video feeds.\n\n"
        "The objective is to build a real-time system that:\n"
        "- Detects humans in video streams.\n"
        "- Tracks individual movements frame-by-frame.\n"
        "- Counts crossings over a virtual line to determine direction (IN/OUT).\n"
        "- Displays live statistics through a web API."
    )
    pdf.chapter_body(intro_text)

    # 3. System Architecture
    pdf.chapter_title('3', 'System Architecture')
    arch_text = (
        "The system works in a pipeline of three main stages:\n\n"
        "A. Detection (YOLOv8):\n"
        "   The system uses the YOLOv8 neural network (ONNX format) to detect objects in each video frame. "
        "It filters detections to retain only the 'person' class with a confidence score above a set threshold (e.g., 0.4).\n\n"
        "B. Tracking (SORT):\n"
        "   To associate detections across frames, the SORT algorithm is used. It employs a Kalman Filter to predict object locations "
        "and the Hungarian Algorithm to match new detections to existing tracks based on Intersection over Union (IoU).\n\n"
        "C. Counting Logic:\n"
        "   A virtual line is defined in the frame (e.g., y=360). The system monitors the centroid of each tracked person. "
        "If a person's centroid crosses the line from top to bottom, it increments the 'IN' counter. "
        "Crossing from bottom to top increments the 'OUT' counter."
    )
    pdf.chapter_body(arch_text)

    # 4. Implementation Details
    pdf.chapter_title('4', 'Implementation Details')
    impl_text = (
        "The project is implemented in Python and structured as follows:\n\n"
        "- core/detection_tracker.py: Orchestrates the pipeline, feeding video frames to YOLO and then to SORT.\n"
        "- core/passenger_counter.py: Maintains the state of each track ID (previous position) and determines if a line crossing occurred.\n"
        "- core/input_reader.py: Handles video input from webcam, file, or RTSP stream in a separate thread for performance.\n"
        "- server/api.py: A Flask-based web server that streams the processed video and serves a JSON API for analytics.\n\n"
        "Dependencies include OpenCV for image processing, PyTorch for deep learning (optional if using ONNX runtime), "
        "and Flask for the web interface."
    )
    pdf.chapter_body(impl_text)

    # 5. Conclusion
    pdf.chapter_title('5', 'Conclusion')
    conc_text = (
        "The Smart Passenger Counter successfully demonstrates the capability of modern computer vision to solve real-world problems. "
        "The system achieves real-time performance on standard hardware by optimizing the detection-tracking pipeline. "
        "Future improvements could include integrating ReID (Re-identification) to handle occlusions better and deploying the "
        "solution to edge devices for tracking on buses or trains."
    )
    pdf.chapter_body(conc_text)

    output_path = "Project_Report.pdf"
    pdf.output(output_path, 'F')
    print(f"PDF Report generated successfully: {output_path}")

if __name__ == "__main__":
    create_report()
