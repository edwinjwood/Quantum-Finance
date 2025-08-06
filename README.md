# 🚀 Quantum Portfolio Optimization & Diversification

A cutting-edge portfolio optimization platform that leverages both classical and quantum computing algorithms to deliver superior investment strategies.

## ✨ Features

### 🔬 **Advanced Optimization Methods**
- **Classical Optimization**: Proven mean-variance optimization using SciPy
- **Quantum Optimization**: Next-generation optimization using Qiskit
- **Comparative Analysis**: Side-by-side quantum vs classical performance evaluation

### 📊 **Comprehensive Risk Analysis**
- Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations
- Maximum Drawdown analysis
- Asset correlation heatmaps
- Monte Carlo simulations with 1,000+ scenarios

### 🤖 **AI-Powered Insights**
- Diversification scoring using Herfindahl-Hirschman Index
- Risk and return contribution analysis
- Intelligent portfolio recommendations
- Automated risk assessment and alerts

### 💼 **Professional Features**
- Real-time stock data integration via Yahoo Finance
- Interactive visualizations with Plotly
- Professional-grade risk metrics
- Client-ready reporting and analysis

## 🏗️ Architecture

```
src/
├── config/          # Application configuration
├── data/            # Data management and fetching
├── optimization/    # Portfolio optimization algorithms
├── analysis/        # Risk analysis and AI insights
└── ui/             # Streamlit user interface
```

### Clean Architecture Benefits:
- **Separation of Concerns**: Each module has a single responsibility
- **Maintainability**: Easy to update and extend functionality
- **Testability**: Modular design enables comprehensive unit testing
- **Scalability**: Simple to add new optimization methods or analysis tools

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run main.py
   ```

3. **Access the App**
   Open your browser to `http://localhost:8501`

## 📖 Usage

1. **Choose Data Source**: Select real stock data or demo data
2. **Configure Portfolio**: Enter stock tickers and date ranges
3. **Select Optimization Method**: Choose Classical, Quantum, or Compare Both
4. **Advanced Settings**: Enable risk analysis, Monte Carlo, and AI insights
5. **Optimize**: Click the optimization button and analyze results

## 🔧 Configuration

Key configuration settings in `src/config/app_config.py`:

- **Default tickers**: AAPL, GOOGL, MSFT, AMZN
- **Risk analysis**: VaR confidence levels, Monte Carlo simulations
- **Quantum settings**: Algorithm selection, iteration limits
- **UI settings**: Chart heights, asset limits

## 🧪 Quantum Computing

The application supports multiple quantum optimization approaches:

- **COBYLA**: Classical-Quantum hybrid optimization
- **QAOA**: Quantum Approximate Optimization Algorithm (future)
- **VQE**: Variational Quantum Eigensolver (future)

Quantum optimization automatically falls back to classical methods if Qiskit packages are not available.

## 📊 Professional Use Cases

- **Investment Advisory**: Client portfolio optimization with quantum advantage
- **Risk Management**: Comprehensive portfolio risk assessment
- **Research & Development**: Quantum finance algorithm testing
- **Educational**: Teaching modern portfolio theory with quantum computing

## 🔬 Technical Details

### Dependencies
- **Streamlit**: Interactive web application framework
- **NumPy/Pandas**: Numerical computing and data manipulation
- **SciPy**: Classical optimization algorithms
- **Qiskit**: Quantum computing framework
- **Plotly**: Interactive visualizations
- **yfinance**: Real-time stock data

### Optimization Algorithms
- **Mean-Variance Optimization**: Markowitz portfolio theory
- **Risk-Return Trade-off**: Configurable risk aversion parameters
- **Constraint Handling**: Weight normalization and bounds
- **Quantum Formulation**: Quadratic programming for quantum solvers

## 🎯 Future Enhancements

- [ ] Additional quantum algorithms (QAOA, VQE)
- [ ] Multi-objective optimization (ESG, sector constraints)
- [ ] Backtesting engine with historical performance
- [ ] API endpoints for programmatic access
- [ ] Advanced ML-powered asset selection

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📧 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact our team.

---

**Built with ❤️ and ⚛️ quantum computing for the future of finance.**
