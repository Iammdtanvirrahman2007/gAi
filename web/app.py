from pathlib import Path
from flask import Flask, jsonify, render_template, request, Response
from brain.core import GrowingBrain
from brain.neural_network import MLP
import base64, json, os, urllib.error, urllib.request, urllib.parse

ROOT = Path(__file__).resolve().parents[1]
brain = GrowingBrain()
app = Flask(__name__, template_folder="templates", static_folder="static")

GITHUB_API = "https://api.github.com"
GITHUB_REPO = os.getenv("GITHUB_REPO", "Iammdtanvirrahman2007/gAi")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://iammdtanvirrahman2007.github.io")

@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

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

@app.get("/api/request/<name>")
def request_file(name):
    safe = Path(name).name
    path = ROOT / "code_requests" / safe
    if path.suffix != ".md" or not path.exists():
        return jsonify({"error": "Request not found."}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain; charset=utf-8")

@app.post("/api/learn")
def learn():
    data = request.get_json(force=True)
    topic = str(data.get("topic", "")).strip()
    content = str(data.get("content", "")).strip()
    if not topic or not content:
        return jsonify({"error": "Topic and lesson are required."}), 400
    return jsonify(brain.learn(topic, content).__dict__)

@app.post("/api/question")
def question():
    data = request.get_json(force=True)
    q = str(data.get("question", "")).strip()
    if not q:
        return jsonify({"error": "Question is required."}), 400
    supplied_data = data.get("data")
    try:
        result = brain.answer_question(q, supplied_data)
        return jsonify(result.__dict__)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/question/csv")
def question_csv():
    data = request.get_json(force=True)
    q = str(data.get("question", "")).strip()
    csv_text = str(data.get("csv", ""))
    if not q:
        return jsonify({"error": "Question is required."}), 400
    if not csv_text.strip():
        return jsonify({"error": "CSV data is required."}), 400
    try:
        result = brain.answer_with_csv(q, csv_text)
        return jsonify(result.__dict__)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/neural/train")
def neural_train():
    data = request.get_json(force=True) or {}
    epochs = int(data.get("epochs", 800))
    learning_rate = float(data.get("learning_rate", 0.8))
    network = MLP(2, 4, 1, seed=7)
    samples = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    history = network.train(samples, epochs=epochs, learning_rate=learning_rate)
    predictions = [{"input": x, "target": y, "output": network.predict(x)} for x, y in samples]
    snapshot = network.snapshot()
    snapshot["training"] = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "history": [point.__dict__ for point in history],
        "predictions": predictions,
    }
    return jsonify(snapshot)

@app.post("/api/request")
def code_request():
    data = request.get_json(force=True)
    capability = str(data.get("capability", "")).strip()
    reason = str(data.get("reason", "")).strip()
    target = str(data.get("target_file", "TBD")).strip() or "TBD"
    requirements = [str(x).strip() for x in data.get("requirements", []) if str(x).strip()]
    if not capability or not reason:
        return jsonify({"error": "Capability and reason are required."}), 400
    path = brain.request_code(capability, reason, requirements, target)
    return jsonify({"file": str(path.relative_to(ROOT)), "status": "WAITING_FOR_HUMAN_CODE"})

def github_request(method, path, payload=None):
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not configured on the backend.")
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        GITHUB_API + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return json.loads(res.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API error {e.code}: {detail[:500]}") from e

@app.post("/api/github/submit-code")
def submit_code():
    data = request.get_json(force=True)
    path = str(data.get("path", "")).strip().lstrip("/")
    code = str(data.get("code", "")).replace("\r\n", "\n")
    message = str(data.get("message", "")).strip() or f"Implement gAi code upgrade: {path}"
    if not path or not code:
        return jsonify({"error": "File path and code are required."}),400
    if path.startswith(".git/") or ".." in Path(path).parts:
        return jsonify({"error": "Unsafe repository path."}),400
    try:
        encoded_path=urllib.parse.quote(path,safe="/")
        existing=github_request("GET",f"/repos/{GITHUB_REPO}/contents/{encoded_path}?ref={urllib.parse.quote(GITHUB_BRANCH)}")
    except Exception as e:
        existing=None
        if "404" not in str(e): return jsonify({"error":str(e)}),502
    payload={"message":message,"content":base64.b64encode(code.encode()).decode(),"branch":GITHUB_BRANCH}
    if existing and isinstance(existing,dict) and existing.get("sha"): payload["sha"]=existing["sha"]
    try:
        result=github_request("PUT",f"/repos/{GITHUB_REPO}/contents/{encoded_path}",payload)
        return jsonify({"ok":True,"path":path,"commit":result.get("commit",{}).get("sha"),"url":result.get("content",{}).get("html_url")})
    except Exception as e:
        return jsonify({"error":str(e)}),502

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000,debug=True)
