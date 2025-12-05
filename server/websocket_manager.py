from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*")

def send_realtime_update(counts):
    socketio.emit("update_counts", counts)
