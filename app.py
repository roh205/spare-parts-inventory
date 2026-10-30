"""
Flask Web Application & API Server for Stochastic Spare Parts Inventory
========================================================================
Runs local web server and REST API, serving the interactive dashboard.
"""

import os
import sys
import webbrowser
from typing import Dict, Any
from flask import Flask, jsonify, request, send_from_directory

# Ensure local directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from simulation import (
    InventoryConfig,
    run_optimization,
    run_sensitivity_analysis,
    compute_policy_metrics,
    simulate_breakdowns_vectorized
)

app = Flask(__name__, static_folder=CURRENT_DIR)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def serve_index():
    return send_from_directory(CURRENT_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "SpareOpt Stochastic Engine"})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    try:
        data = request.get_json(silent=True) or {}
        config = InventoryConfig(
            num_machines=int(data.get("num_machines", 25)),
            hours_per_month=float(data.get("hours_per_month", 600.0)),
            weibull_k=float(data.get("weibull_k", 2.2)),
            weibull_lambda=float(data.get("weibull_lambda", 1800.0)),
            cost_procure=float(data.get("cost_procure", 1500.0)),
            cost_holding=float(data.get("cost_holding", 120.0)),
            cost_emergency=float(data.get("cost_emergency", 4200.0)),
            cost_downtime=float(data.get("cost_downtime", 6000.0)),
            max_storage=int(data.get("max_storage", 30)),
            num_scenarios=int(data.get("num_scenarios", 15000)),
            random_seed=data.get("random_seed", 42)
        )
        results = run_optimization(config)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    try:
        data = request.get_json(silent=True) or {}
        config = InventoryConfig(
            num_machines=int(data.get("num_machines", 25)),
            hours_per_month=float(data.get("hours_per_month", 600.0)),
            weibull_k=float(data.get("weibull_k", 2.2)),
            weibull_lambda=float(data.get("weibull_lambda", 1800.0)),
            cost_procure=float(data.get("cost_procure", 1500.0)),
            cost_holding=float(data.get("cost_holding", 120.0)),
            cost_emergency=float(data.get("cost_emergency", 4200.0)),
            cost_downtime=float(data.get("cost_downtime", 6000.0)),
            max_storage=int(data.get("max_storage", 30)),
            num_scenarios=int(data.get("num_scenarios", 5000))
        )
        k_values = data.get("k_values", [1.2, 1.6, 2.0, 2.2, 2.5, 3.0])
        results = run_sensitivity_analysis(config, k_values)
        return jsonify({"sensitivity": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def start_server(port: int = 5000, open_browser: bool = True):
    print("=" * 65)
    print(" SPARING OPTIMIZATION PRO - STOCHASTIC DECISION SYSTEM")
    print("=" * 65)
    print(f" Starting local dashboard at: http://localhost:{port}")
    print(f" Serving UI from: {os.path.join(CURRENT_DIR, 'index.html')}")
    print(" Press Ctrl+C to terminate.")
    print("=" * 65)

    if open_browser:
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spare Parts Stochastic Inventory Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve dashboard (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)

