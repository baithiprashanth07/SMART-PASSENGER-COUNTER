"""
Enhanced WebSocket Manager for Smart Passenger Counter
Handles real-time communication between the backend and frontend dashboard
"""

from flask_socketio import SocketIO, emit
from flask import request
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize SocketIO with CORS support
socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# Track connected clients
connected_clients = set()


# ============================================================================
# Connection Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        client_id = request.sid
        connected_clients.add(client_id)
        logger.info(f"✅ Client connected: {client_id} (Total: {len(connected_clients)})")
        emit('connection_response', {
            'status': 'connected',
            'timestamp': datetime.now().isoformat(),
            'message': 'Successfully connected to Smart Passenger Counter Dashboard'
        })
    except Exception as e:
        logger.error(f"Error in handle_connect: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        client_id = request.sid
        connected_clients.discard(client_id)
        logger.info(f"⚠️ Client disconnected: {client_id} (Total: {len(connected_clients)})")
    except Exception as e:
        logger.error(f"Error in handle_disconnect: {e}")

@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'timestamp': datetime.now().isoformat()})


# ============================================================================
# Real-time Count Updates
# ============================================================================

def send_realtime_update(counts):
    """
    Broadcast real-time passenger count updates to all connected clients
    """
    try:
        socketio.emit('update_counts', counts)
        # logger.debug(f"📊 Broadcast count update: {counts}") 
    except Exception as e:
        logger.error(f"❌ Error broadcasting count update: {e}")


def send_detection_update(detection_count):
    """
    Broadcast detection count to all connected clients
    """
    try:
        socketio.emit('detection_update', {
            'count': detection_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error broadcasting detection update: {e}")


def send_fps_update(fps):
    """
    Broadcast FPS (frames per second) to all connected clients
    """
    try:
        socketio.emit('fps_update', {
            'fps': fps,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error broadcasting FPS update: {e}")