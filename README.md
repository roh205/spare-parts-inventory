# Stochastic Decision Support System: Spare Parts Inventory Optimization

An executive-grade decision support platform for optimizing spare parts inventory under stochastic machine breakdown renewal processes and high-consequence downtime penalties.

---

## 📌 Executive Summary

Industrial spare parts management faces severe cost asymmetries: holding an extra component in inventory costs **₹120/month**, whereas a component stockout induces emergency procurement (**₹4,200**) plus production line downtime (**₹6,000**), amounting to **₹10,200 per stockout incident**.

A naive **deterministic policy** ordering the rounded expected demand ($S_{\text{det}} = 2$ units) results in a **36.5% stockout probability** and monthly costs of **₹9,368**. In contrast, the **stochastic optimal policy** ($S^* = 4$ units):
- Reduces expected total monthly costs to **₹7,104**
- Yields a **Value of Stochastic Solution (VSS)** of **₹2,264 / month** (**24.2% direct cost reduction**)
- Boosts **Cycle Service Level (CSL)** from **63.5%** to **93.8%**
- Slashes catastrophic 95% CVaR tail risk from **₹37,420** to **₹20,020**

---

## 🚀 How to Run the Application

### 1. Instant Interactive Web UI (Zero Installation)
Simply open the included `index.html` directly in any modern browser (Google Chrome, Microsoft Edge, Brave, Firefox):
```bash
# Double-click or open from terminal:
start index.html
```
- **Live 60fps Monte Carlo Simulator** built into the browser (no Python needed).
- Interactive sliders for fleet size $N$, operating hours $H$, Weibull parameters $(k, \lambda)$, and cost penalties.
- High-fidelity **Chart.js** visualizations:
  1. Cost curves & decomposition (Procurement, Holding, Stockout Penalty, 95% CVaR).
  2. Cost probability density comparison (Stochastic vs. Deterministic).
  3. Breakdown demand frequency and cumulative CDF.
- **What-If Policy Sandbox**: test any arbitrary stock level $S \in [0, 40]$ in real-time.
- **One-click CSV Data Export** and **Executive Printable Report**.

### 2. Local Python Web Server & REST API
```bash
python app.py
```
Starts a local Flask server at `http://localhost:5000` with automated browser launch and provides REST endpoints:
- `GET /` — Serves the interactive dashboard.
- `POST /api/optimize` — Runs server-side Monte Carlo optimization.
- `POST /api/sensitivity` — Evaluates parameter sensitivities across wear rates ($k$).

### 3. Command-Line Interface (CLI Runner)
```bash
python cli.py
```
- High-speed vectorized simulation running 20,000 scenarios in ~0.06 seconds.
- Formats executive comparison tables.
- Exports results to `optimization_results.csv`.
- Resilient design: does not crash if `matplotlib` is missing.

---

## 📐 Mathematical Formulation

### 1. Machine Failure Dynamics (Weibull Renewal Process)
Each of the $N$ machines experiences wear-out degradation. Component time-to-failure follows a 2-parameter Weibull distribution:
$$f(t; k, \lambda) = \frac{k}{\lambda} \left(\frac{t}{\lambda}\right)^{k-1} e^{-(t/\lambda)^k}$$
- **$k = 2.2$**: Shape parameter indicating wear-out / aging stage ($k > 1$).
- **$\lambda = 1,800$ hrs**: Characteristic scale parameter.
- **MTBF**: $\lambda \cdot \Gamma\left(1 + \frac{1}{k}\right) \approx 1,594.1$ operating hours.

### 2. Cost Objective Function
For a monthly planning horizon $H = 600$ hours and base-stock order quantity $S$:
$$\text{Cost}(S, D) = C_p \cdot S + C_h \cdot \max(0, S - D) + (C_e + C_d) \cdot \max(0, D - S)$$
Where:
- $C_p =$ ₹1,500 (Regular procurement)
- $C_h =$ ₹120 (Holding cost per unit/month)
- $C_e =$ ₹4,200 (Emergency rush order)
- $C_d =$ ₹6,000 (Line stoppage & downtime)
- $C_s = C_e + C_d =$ ₹10,200 (Stockout penalty)

### 3. Key Decision Metrics
- **Value of Stochastic Solution (VSS)**:
  $$\text{VSS} = E[\text{Cost}(S_{\text{det}})] - E[\text{Cost}(S^*)]$$
- **Expected Value of Perfect Information (EVPI)**:
  $$\text{EVPI} = E[\text{Cost}(S^*)] - E[\text{Wait-and-See}]$$
- **Conditional Value at Risk (95% CVaR)**:
  $$\text{CVaR}_{0.95} = E[\text{Cost} \mid \text{Cost} \ge \text{VaR}_{0.95}]$$

---

## 📊 Benchmark Results Summary

| Metric | Deterministic Policy ($S_{\text{det}} = 2$) | Stochastic Policy ($S^* = 4$) | Impact / Benefit |
| :--- | :---: | :---: | :---: |
| **Base Stock Level ($S$)** | 2 units | 4 units | +2 units Safety Buffer |
| **Expected Total Cost** | ₹9,368.79 | **₹7,104.10** | **-₹2,264.69 (24.2% savings)** |
| **Stockout Penalty Cost** | ₹6,312.78 | **₹872.10** | **-86.2% risk exposure** |
| **Cycle Service Level (CSL)** | 63.5% | **93.8%** | **+30.3% reliability** |
| **Order Fill Rate** | 71.2% | **96.0%** | **+24.8% parts fulfillment** |
| **Stockout Incident Rate** | 36.5% | **6.2%** | Slashed to rare events |
| **95% CVaR Financial Risk** | ₹37,420.90 | **₹20,020.90** | **-₹17,400 downside cap** |

---

## 🛠 Project Structure

```
stochastic-spare-parts-inventory/
├── index.html              # Standalone executive interactive web dashboard (Chart.js + Tailwind)
├── simulation.py           # Vectorized Monte Carlo & stochastic optimization engine
├── app.py                  # Flask web server & REST API service
├── cli.py                  # High-speed console runner and CSV exporter
├── optimization_results.csv# Exported policy curve data
├── requirements.txt        # Python dependency manifest
└── README.md               # Mathematical formulation & user guide
```

