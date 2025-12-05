import cv2
import numpy as np


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2, labels=None):
    """
    Draw bounding boxes on frame.
    
    Args:
        frame: Input frame
        boxes: List of boxes [[x1, y1, x2, y2], ...]
        color: Box color (B, G, R)
        thickness: Line thickness
        labels: Optional list of labels for each box
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        if labels and i < len(labels):
            label = str(labels[i])
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
    
    return annotated


def draw_lines(frame, lines, color=(255, 0, 0), thickness=2):
    """
    Draw counting lines on frame.
    
    Args:
        frame: Input frame
        lines: List of line definitions with "coords": [x1, y1, x2, y2]
        color: Line color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for line in lines:
        coords = line.get("coords", line)
        if isinstance(coords, dict):
            coords = coords.get("coords")
        
        x1, y1, x2, y2 = map(int, coords[:4])
        cv2.line(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw line name if available
        name = line.get("name", "")
        if name:
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                annotated,
                name,
                (mid_x, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    
    return annotated


def calculate_centroid(box):
    """
    Calculate centroid of bounding box.
    
    Args:
        box: [x1, y1, x2, y2]
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_text_with_background(frame, text, position, font_scale=0.7, 
                               text_color=(255, 255, 255), 
                               bg_color=(0, 0, 0), thickness=2):
    """
    Draw text with background rectangle for better visibility.
    
    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        thickness: Text thickness
    
    Returns:
        Annotated frame
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )
    
    return frame


def draw_counting_info(frame, counts, position=(10, 30)):
    """
    Draw counting information on frame.
    
    Args:
        frame: Input frame
        counts: Dictionary with "in", "out", "total" keys
        position: Starting position (x, y)
    
    Returns:
        Annotated frame
    """
    x, y = position
    line_height = 30
    
    info_lines = [
        f"IN: {counts.get('in', 0)}",
        f"OUT: {counts.get('out', 0)}",
        f"TOTAL: {counts.get('total', 0)}"
    ]
    
    for i, line in enumerate(info_lines):
        draw_text_with_background(
            frame,
            line,
            (x, y + i * line_height),
            font_scale=0.8,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            thickness=2
        )
    
    return frame


def resize_frame(frame, width=None, height=None, keep_aspect=True):
    """
    Resize frame to specified dimensions.
    
    Args:
        frame: Input frame
        width: Target width (None to auto-calculate)
        height: Target height (None to auto-calculate)
        keep_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if keep_aspect:
        if width is not None:
            aspect = width / w
            height = int(h * aspect)
        elif height is not None:
            aspect = height / h
            width = int(w * aspect)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
    
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)