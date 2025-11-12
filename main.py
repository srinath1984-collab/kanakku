from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# This enables CORS for all domains on all routes.
# For better security later, replace '*' with your specific frontend domain.
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello():
    return "Hello from Flask with CORS enabled!"

if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
