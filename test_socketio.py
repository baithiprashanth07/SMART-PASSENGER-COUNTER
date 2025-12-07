print("Importing flask_socketio...")
try:
    from flask_socketio import SocketIO, emit, broadcast
    print("Success importing SocketIO, emit, broadcast")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("Importing request...")
try:
    from flask import request
    print("Success importing request")
except Exception as e:
    print(f"Failed request: {e}")
