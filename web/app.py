from pathlib import Path
from flask import Flask, jsonify, render_template, request
from brain.core import GrowingBrain

ROOT = Path(__file__).resolve().parents[1]
brain = GrowingBrain()
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/state")
def state():
    requests = sorted((ROOT / "code_requests").glob("*.md"), reverse=True)
    return jsonify({
        "lessons": [lesson.__dict__ for lesson in brain.lessons],
        "capabilities": [cap.to_dict() for cap in brain.capabilities.all()],
        "code_requests": [p.name for p in requests],
    })

@app.post("/api/learn")
def learn():
    data = request.get_json(force=True)
    topic = str(data.get("topic", "")).strip()
    content = str(data.get("content", "")).strip()
    if not topic or not content:
        return jsonify({"error": "Topic and lesson are required."}), 400
    lesson = brain.learn(topic, content)
    return jsonify(lesson.__dict__)

@app.post("/api/request")
def code_request():
    data = request.get_json(force=True)
    capability = str(data.get("capability", "")).strip()
    reason = str(data.get("reason", "")).strip()
    target_file = str(data.get("target_file", "TBD")).strip() or "TBD"
    requirements = [str(x).strip() for x in data.get("requirements", []) if str(x).strip()]
    if not capability or not reason:
        return jsonify({"error": "Capability and reason are required."}), 400
    path = brain.request_code(capability, reason, requirements, target_file)
    return jsonify({"file": str(path.relative_to(ROOT)), "status": "WAITING_FOR_HUMAN_CODE"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
