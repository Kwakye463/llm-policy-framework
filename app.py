# app.py — Flask API backend for the policy engine demo UI
# Run with: python app.py
# Serves BOTH the API and the dashboard on http://localhost:5050

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_engine import PolicyEngine, extract_action, evaluate_tool_call, validate_output, CLASSIFIER_MODEL
import requests as req

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
CORS(app)

engine = PolicyEngine()
OLLAMA_URL = "http://localhost:11434/api/generate"
RESPONSE_MODEL = "tinyllama"

# Build a human-readable label from the model name.
# e.g. "phi3:mini" -> "Phi-3 Mini", "llama3.2:3b" -> "Llama 3.2 3B"
def model_display_name(model_id: str) -> str:
    labels = {
        "phi3:mini":      "Phi-3 Mini",
        "llama3.2:3b":    "Llama 3.2 3B",
        "llama3.1:8b":    "Llama 3.1 8B",
        "mistral":        "Mistral 7B",
        "gemma2:2b":      "Gemma 2 2B",
    }
    return labels.get(model_id, model_id)

CLASSIFIER_LABEL = model_display_name(CLASSIFIER_MODEL)


@app.route("/")
def index():
    """Serve the dashboard directly — avoids CORS file:// issues"""
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    prompt = data.get("prompt", "").strip()
    source = data.get("source", "user")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    start = time.time()
    result = {"prompt": prompt, "source": source, "timeline": []}

    # Layer 1 + Layer 2
    action = extract_action(prompt, source)
    layer1_flags = {k: v for k, v in action.items() if k != "llm_judged_unsafe"}
    layer2_result = action["llm_judged_unsafe"]

    result["timeline"].append({
        "step": "Layer 1 — Hybrid Detection (Keyword + Semantic)",
        "detail": layer1_flags,
        "status": "flagged" if any(v for k, v in layer1_flags.items() if k != "source_is_external" and v) else "clean"
    })

    # Dynamic label — reads CLASSIFIER_MODEL from policy_engine.py
    result["timeline"].append({
        "step": f"Layer 2 — Semantic Classifier ({CLASSIFIER_LABEL})",
        "detail": {"llm_judged_unsafe": layer2_result},
        "status": "flagged" if layer2_result else "clean"
    })

    # OPA evaluation
    policy_request = {"prompt": prompt, "source": source, "action": action}
    policy_result = engine.evaluate(policy_request)

    result["timeline"].append({
        "step": "OPA Policy Engine (Rego)",
        "detail": {
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "layer": policy_result["layer"]
        },
        "status": "blocked" if policy_result["decision"] == "DENY" else "clean"
    })

    if policy_result["decision"] == "DENY":
        result["final_decision"] = "BLOCKED (Input)"
        result["final_reason"] = policy_result["reason"]
        result["response"] = None
        result["output_validation"] = None
        result["latency"] = round(time.time() - start, 2)
        return jsonify(result)

    # Call LLM
    try:
        llm_resp = req.post(
            OLLAMA_URL,
            json={"model": RESPONSE_MODEL, "prompt": prompt, "stream": False},
            timeout=300
        )
        raw_response = llm_resp.json().get("response", "No response from LLM")
    except Exception as e:
        raw_response = f"LLM error: {str(e)}"

    result["timeline"].append({
        "step": "LLM Response Generated (TinyLlama)",
        "detail": {"model": RESPONSE_MODEL, "response_length": len(raw_response)},
        "status": "clean"
    })

    # Output validation
    output_check = validate_output(raw_response)
    result["timeline"].append({
        "step": "Output Validation",
        "detail": {
            "valid": output_check["valid"],
            "violation_type": output_check["violation_type"],
            "reason": output_check["reason"]
        },
        "status": "blocked" if not output_check["valid"] else "clean"
    })

    if not output_check["valid"]:
        result["final_decision"] = "BLOCKED (Output)"
        result["final_reason"] = output_check["reason"]
        result["response"] = "[RESPONSE BLOCKED — output policy violation]"
    else:
        result["final_decision"] = "ALLOWED"
        result["final_reason"] = "Passed all policy checks"
        result["response"] = raw_response

    result["output_validation"] = output_check
    result["latency"] = round(time.time() - start, 2)
    return jsonify(result)


@app.route("/tool", methods=["POST"])
def tool():
    data = request.json
    result = evaluate_tool_call(data.get("tool", ""), data.get("source", "user"), data.get("args", {}))
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classifier_model": CLASSIFIER_MODEL, "classifier_label": CLASSIFIER_LABEL})


if __name__ == "__main__":
    print("\n" + "="*55)
    print(" LLM Policy Engine — Demo Interface")
    print("="*55)
    print(f" Classifier model : {CLASSIFIER_LABEL} ({CLASSIFIER_MODEL})")
    print(" Open in your browser:")
    print(" http://localhost:5050")
    print("="*55 + "\n")
    app.run(port=5050, debug=False)