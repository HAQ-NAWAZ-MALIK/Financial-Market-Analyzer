# Financial Market Analyzer

![Financial Markets](https://img.shields.io/badge/AI-Financial%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A sophisticated multi-agent system for financial market analysis using AI. This project leverages language models and specialized tools to collect market data, analyze financial news, perform technical analysis, and generate investment insights with informative visualizations.
![image](https://github.com/user-attachments/assets/1e204db2-e976-46fd-bcce-6c6316edf398)

<p align="center">

</p>

## 🌟 Features

- **Real-time Market Data** - Retrieves stock prices, volumes, and technical indicators from reliable sources
- **Financial News Analysis** - Searches and interprets financial news for market sentiment
- **Technical Analysis** - Identifies chart patterns, support/resistance levels, and key indicators
- **Multi-Agent Architecture** - Uses specialized agents for different aspects of financial analysis
- **Professional Visualizations** - Creates publication-quality financial charts and dashboards
- **Investment Insights** - Delivers data-driven financial analysis and market insights

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Example Visualizations](#-example-visualizations)
- [API Keys](#-api-keys)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Alpha Vantage API key (for financial data)
- News API key (for financial news)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/HAQ-NAWAZ-MALIK/financial-market-analyzer.git
   cd financial-market-analyzer
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
   NEWS_API_KEY=your-news-api-key-here
   DEFAULT_MODEL=gpt-4.1-mini
   ```

## 🚀 Usage

### Basic Example

```python
from financial_market_analyzer import run_financial_analysis

# Run a financial analysis query
results = run_financial_analysis(
    "Analyze the recent performance of major tech stocks (AAPL, MSFT, GOOGL) over the past month.",
    symbols=["AAPL", "MSFT", "GOOGL"]
)

# Print the results
print(results)

# The visualization is saved as "financial_visualization.png"
```

### Example in Jupyter Notebook

```python
from financial_market_analyzer import run_financial_analysis
import matplotlib.pyplot as plt
from IPython.display import display

# Run the analysis
results = run_financial_analysis(
    "Compare the performance of Tesla (TSLA) against traditional auto manufacturers (F, GM) " +
    "and analyze their correlation with broader market trends.",
    symbols=["TSLA", "F", "GM", "SPY"]
)

# Display results
print(results)

# Display the visualization
plt.figure(figsize=(12, 8))
img = plt.imread("financial_visualization.png")
plt.imshow(img)
plt.axis('off')
plt.show()
```

## 🏗️ Architecture

The Financial Market Analyzer uses a multi-agent architecture:

```
┌────────────────────────────────┐
│    Financial Manager Agent     │
│   (Coordinates Analysis)       │
└─────────────┬──────────────────┘
              │
              ▼
┌─────────────┴──────────────────┐
│                                │
▼                                ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ Market Data Agent  │  │ News Analysis Agent│  │Technical Analysis   │
│ (Price Data)       │  │ (Financial News)   │  │Agent (Patterns)     │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

1. **Financial Manager Agent**: Coordinates the analysis process and creates visualizations
2. **Market Data Agent**: Retrieves stock prices, volumes, and market indicators
3. **News Analysis Agent**: Searches and analyzes financial news and sentiment
4. **Technical Analysis Agent**: Identifies patterns, trends, and technical indicators

## 📊 Example Visualizations

The agent can create various types of financial visualizations:

- Multi-stock price comparison charts
- Technical indicator overlays (MACD, RSI, Moving Averages)
- Volume analysis charts
- Correlation heatmaps
- Candlestick charts with pattern recognition
- Performance comparison dashboards

## 🔑 API Keys

This project requires the following API keys:

1. **OpenAI API Key**: Powers the language models
   - Sign up at [OpenAI Platform](https://platform.openai.com/signup)
   - Get your API key from [API Keys page](https://platform.openai.com/account/api-keys)

2. **Alpha Vantage API Key**: Provides financial market data
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Free tier available with limited daily API calls

3. **News API Key**: Powers financial news search
   - Sign up at [News API](https://newsapi.org/register)
   - Free tier available with limited daily API calls

## ❓ Troubleshooting

### Common Issues

1. **Authentication Error (401)**:
   ```
   AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided...
   ```
   
   **Solution**: Ensure you've set a valid OpenAI API key that starts with `sk-`.

2. **Alpha Vantage API Error**:
   ```
   {"error": "API request failed: Invalid API call..."}
   ```
   
   **Solution**: Verify your Alpha Vantage API key and check if you've reached the daily limit for free tier usage.

3. **Missing Financial Visualization**:
   ```
   Visualization file not found
   ```
   
   **Solution**: Ensure the agent completed successfully and check permissions to write files in the current directory.

4. **Symbol Not Found Error**:
   ```
   {"error": "No data found for XYZ. Verify the symbol is correct."}
   ```
   
   **Solution**: Ensure you're using valid stock ticker symbols that are available on Alpha Vantage.




## full Usage Example

```
from financial_market_analyzer import run_financial_analysis
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from IPython.display import display, Image

# Load environment variables from .env file
load_dotenv(override=True)

# IMPORTANT: Set your actual API keys below or in a .env file
# Example:
# os.environ["OPENAI_API_KEY"] = "your-actual-openai-key-here"
# os.environ["ALPHA_VANTAGE_API_KEY"] = "your-actual-alpha-vantage-key-here"
# os.environ["NEWS_API_KEY"] = "your-actual-news-api-key-here"

# Check if API keys are set before running
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith(("your_", "sk-your")):
    print("⚠️ Warning: OpenAI API key not set properly!")
    print("Please set your OpenAI API key by adding this line before running:")
    print('os.environ["OPENAI_API_KEY"] = "your-actual-key-here"')
    print("Or set it in your .env file")
else:
    # Example 1: Basic Tech Stock Analysis
    print("\n=== EXAMPLE 1: TECH STOCK ANALYSIS ===")
    tech_analysis = """
    Analyze the recent performance of major tech stocks (AAPL, MSFT, GOOGL) over the past month.
    Compare their price movements, trading volumes, and key technical indicators.
    Identify any patterns or correlations between these stocks.
    Include relevant financial news that might explain their performance.
    """
    
    tech_results = run_financial_analysis(
        tech_analysis,
        symbols=["AAPL", "MSFT", "GOOGL"]
    )
    
    print("\n=== TECH STOCK ANALYSIS RESULTS ===")
    print(tech_results)
    
    # Example 2: Sector Comparison
    print("\n=== EXAMPLE 2: SECTOR COMPARISON ===")
    sector_analysis = """
    Compare the financial performance of different market sectors using representative ETFs:
    - Technology (XLK)
    - Energy (XLE)
    - Healthcare (XLV)
    - Financials (XLF)
    
    Analyze which sectors are outperforming the overall market (SPY) in the past 3 months.
    Include recent sector rotation trends and correlation with economic indicators.
    """
    
    sector_results = run_financial_analysis(
        sector_analysis,
        symbols=["XLK", "XLE", "XLV", "XLF", "SPY"]
    )
    
    print("\n=== SECTOR COMPARISON RESULTS ===")
    print(sector_results)
    
    # Display the visualization from the most recent analysis
    if os.path.exists("financial_visualization.png"):
        plt.figure(figsize=(12, 8))
        img = plt.imread("financial_visualization.png")
        plt.imshow(img)
        plt.axis('off')
        plt.title("Financial Market Analysis Visualization")
        plt.show()
    else:
        print("Visualization file not found")
 ``` 
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [SmolaGents](https://github.com/smol-ai/smolagents) for the agent framework
- [Alpha Vantage](https://www.alphavantage.co/) for financial data API
- [News API](https://newsapi.org/) for financial news
- [Plotly](https://plotly.com/) for visualization capabilities
- [OpenAI](https://openai.com/) for language model APIs

---

<p align="center">
  Made with ❤️ for data-driven investment decisions
</p>
