import sys
import os
sys.path.insert(0, os.path.abspath('.'))
print(f"Path: {sys.path[0]}")

try:
    import server.websocket_manager
    print(f"Imported server.websocket_manager from: {server.websocket_manager.__file__}")
    if hasattr(server.websocket_manager, 'socketio'):
        print("socketio attribute exists")
    else:
        print("socketio attribute MISSING")
except Exception as e:
    print(f"Failed to import module: {e}")
    import traceback
    traceback.print_exc()
