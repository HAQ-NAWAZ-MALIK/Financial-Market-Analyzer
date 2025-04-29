import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool, Tool
from PIL import Image
from smolagents.utils import make_image_url, encode_image_base64
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('financial_market_analyzer')

# Load environment variables
load_dotenv(override=True)

# Check if API keys are properly set
def check_api_keys():
    """Verify that required API keys are properly set"""
    openai_key = os.getenv("OPENAI_API_KEY")
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not openai_key or openai_key.startswith(('your_', 'sk-your')):
        raise ValueError(
            "OpenAI API key not properly configured. Please set a valid OPENAI_API_KEY in your .env file. "
            "You can get an API key from https://platform.openai.com/account/api-keys"
        )
    
    if not alpha_vantage_key or alpha_vantage_key.startswith('your_'):
        raise ValueError(
            "Alpha Vantage API key not properly configured. Please set a valid ALPHA_VANTAGE_API_KEY in your .env file. "
            "You can get a free API key from https://www.alphavantage.co/support/#api-key"
        )
    
    if not news_api_key or news_api_key.startswith('your_'):
        raise ValueError(
            "News API key not properly configured. Please set a valid NEWS_API_KEY in your .env file. "
            "You can get a free API key from https://newsapi.org/register"
        )

# ====== FINANCIAL DATA TOOLS ======

@tool
def get_stock_data(symbol: str, interval: str = "daily", output_size: str = "compact") -> str:
    """
    Fetches stock price data for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
        interval: Time interval between data points (options: daily, weekly, monthly)
        output_size: Amount of data to retrieve (options: compact, full)
    
    Returns:
        JSON string with OHLCV (Open, High, Low, Close, Volume) data
    """
    import requests
    
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if interval == "daily":
        function = "TIME_SERIES_DAILY"
        key_prefix = "Time Series (Daily)"
    elif interval == "weekly":
        function = "TIME_SERIES_WEEKLY"
        key_prefix = "Weekly Time Series"
    elif interval == "monthly":
        function = "TIME_SERIES_MONTHLY"
        key_prefix = "Monthly Time Series"
    else:
        return json.dumps({"error": f"Invalid interval: {interval}. Use daily, weekly, or monthly."})
    
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={output_size}&apikey={alpha_vantage_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors in the API response
        if "Error Message" in data:
            return json.dumps({"error": data["Error Message"]})
        
        if key_prefix not in data:
            return json.dumps({"error": f"No data found for {symbol}. Verify the symbol is correct."})
        
        # Format the time series data
        time_series = data[key_prefix]
        formatted_data = []
        
        for date, values in time_series.items():
            formatted_data.append({
                "date": date,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"])
            })
        
        # Sort by date, most recent first
        formatted_data.sort(key=lambda x: x["date"], reverse=True)
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "data": formatted_data
        })
        
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except (KeyError, ValueError) as e:
        return json.dumps({"error": f"Error parsing data: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@tool
def get_technical_indicators(symbol: str, indicator: str, time_period: int = 14) -> str:
    """
    Retrieves technical indicators for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
        indicator: Technical indicator (e.g., SMA, EMA, RSI, MACD, BBANDS)
        time_period: Time period for the indicator calculation (default: 14)
    
    Returns:
        JSON string with technical indicator data
    """
    import requests
    
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    # Map common indicator names to Alpha Vantage function names
    indicator_map = {
        "SMA": "SMA",
        "EMA": "EMA",
        "RSI": "RSI",
        "MACD": "MACD",
        "BBANDS": "BBANDS",
        "ADX": "ADX",
        "CCI": "CCI",
        "STOCH": "STOCH",
        "OBV": "OBV",
        "ATR": "ATR"
    }
    
    # Verify the indicator is supported
    indicator = indicator.upper()
    if indicator not in indicator_map:
        supported = ", ".join(indicator_map.keys())
        return json.dumps({"error": f"Indicator {indicator} not supported. Use one of: {supported}"})
    
    function = f"TECHNICAL_INDICATOR_{indicator_map[indicator]}"
    series_type = "close"  # Most indicators use close prices
    
    url = f"https://www.alphavantage.co/query?function={indicator_map[indicator]}&symbol={symbol}&interval=daily&time_period={time_period}&series_type={series_type}&apikey={alpha_vantage_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors in the API response
        if "Error Message" in data:
            return json.dumps({"error": data["Error Message"]})
        
        # Extract the indicator data
        if "Technical Analysis" in data:
            # Most indicators follow this format
            indicator_data = data["Technical Analysis: " + indicator]
            formatted_data = []
            
            for date, values in indicator_data.items():
                entry = {"date": date}
                for key, value in values.items():
                    entry[key] = float(value)
                formatted_data.append(entry)
                
            # Sort by date, most recent first
            formatted_data.sort(key=lambda x: x["date"], reverse=True)
            
            return json.dumps({
                "symbol": symbol,
                "indicator": indicator,
                "time_period": time_period,
                "data": formatted_data
            })
        elif "MACD" in data:
            # MACD has a special format
            macd_data = data["MACD"]
            formatted_data = []
            
            for date, values in macd_data.items():
                formatted_data.append({
                    "date": date,
                    "macd": float(values["MACD"]),
                    "macd_signal": float(values["MACD_Signal"]),
                    "macd_hist": float(values["MACD_Hist"])
                })
                
            # Sort by date, most recent first
            formatted_data.sort(key=lambda x: x["date"], reverse=True)
            
            return json.dumps({
                "symbol": symbol,
                "indicator": "MACD",
                "data": formatted_data
            })
        else:
            return json.dumps({"error": f"Unexpected response format for {indicator}. API may have changed."})
            
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except (KeyError, ValueError) as e:
        return json.dumps({"error": f"Error parsing data: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@tool
def search_financial_news(query: str, sources: str = "bloomberg,cnbc,financial-times,the-wall-street-journal", days: int = 7) -> str:
    """
    Searches for financial news articles based on a query.
    
    Args:
        query: Search query (e.g., company name, ticker symbol, financial term)
        sources: Comma-separated list of news sources
        days: Number of days to look back for articles (1-30)
    
    Returns:
        JSON string with relevant news articles
    """
    import requests
    from datetime import datetime, timedelta
    
    news_api_key = os.getenv("NEWS_API_KEY")
    
    # Calculate date range
    if days < 1 or days > 30:
        days = min(max(days, 1), 30)  # Clamp between 1 and 30
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for the API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sources": sources,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": news_api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "ok":
            return json.dumps({"error": data.get("message", "Unknown error from News API")})
        
        articles = data["articles"]
        
        # Format and clean the article data
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "author": article.get("author", "Unknown"),
                "published_at": article["publishedAt"],
                "url": article["url"],
                "description": article["description"],
                "content": article.get("content", "No content available")
            })
        
        return json.dumps({
            "query": query,
            "sources": sources,
            "date_range": f"{from_date} to {to_date}",
            "articles_count": len(formatted_articles),
            "articles": formatted_articles[:10]  # Limit to top 10 articles for brevity
        })
        
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except (KeyError, ValueError) as e:
        return json.dumps({"error": f"Error parsing data: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


class WebpageVisitorTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given URL and reads its content as a markdown string. "
        "Use this to browse financial websites, read articles, and gather market information."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            import re
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: "
                "run `pip install markdownify requests`."
            ) from e
        
        try:
            # Add headers to mimic a browser request (to avoid being blocked)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            content = response.text
            
            # Check if it's a PDF and handle accordingly
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                return "This is a PDF document. Please use a specialized PDF extraction tool to analyze this content."
            
            markdown_content = markdownify(content).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            
            # Prioritize content based on financial relevance
            financial_markers = ["stock", "market", "price", "earnings", "investment", 
                             "financial", "economy", "trade", "nasdaq", "dow jones", "s&p"]
            
            for marker in financial_markers:
                pattern = re.compile(f'({marker})', re.IGNORECASE)
                markdown_content = pattern.sub(r'**\1**', markdown_content)
            
            return truncate_content(markdown_content, 40000)

        except requests.exceptions.Timeout:
            return "The request to the financial resource timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred when accessing the resource: {str(e)}"


# ====== FINANCIAL MARKET ANALYZER AGENTS ======

# 1. Market Data Agent
def create_market_data_agent(model_name="gpt-4.1-mini"):
    """Create a market data agent specialized for financial data retrieval"""
    market_agent = CodeAgent(
        model=OpenAIServerModel(
            model_name,
            max_completion_tokens=8096,
        ),
        tools=[get_stock_data, get_technical_indicators],
        max_steps=10,
        name="market_data_agent",
        description="Retrieves financial market data, stock prices, and technical indicators."
    )
    market_agent.logger.console.width = 80
    return market_agent

# 2. News Analysis Agent
def create_news_analysis_agent(model_name="gpt-4.1-mini"):
    """Create a news analysis agent specialized in financial news research"""
    news_agent = CodeAgent(
        model=OpenAIServerModel(
            model_name,
            max_completion_tokens=8096,
        ),
        tools=[search_financial_news, WebpageVisitorTool()],
        max_steps=10,
        name="news_analysis_agent",
        description="Searches and analyzes financial news, articles, and market commentary."
    )
    news_agent.logger.console.width = 80
    return news_agent

# 3. Technical Analysis Agent
def create_technical_analysis_agent(model_name="gpt-4.1-mini"):
    """Create an agent specialized in technical analysis of financial data"""
    technical_agent = CodeAgent(
        model=OpenAIServerModel(
            model_name,
            max_completion_tokens=8096,
        ),
        tools=[get_stock_data, get_technical_indicators],
        max_steps=10,
        name="technical_analysis_agent",
        description="Performs technical analysis on financial data, identifying patterns and trends."
    )
    technical_agent.logger.console.width = 80
    return technical_agent

# 4. Data Visualization validation function
def validate_financial_visualization(final_answer, agent_memory):
    """
    Validates that the financial visualization meets quality standards:
    - Clear and accurate data representation
    - Appropriate visualization type for financial data
    - Proper labeling and citation of data sources
    - Professional financial presentation
    """
    multimodal_model = OpenAIServerModel(
        "gpt-4o",
    )
    
    filepath = "financial_visualization.png"
    assert os.path.exists(filepath), "Make sure to save the visualization as financial_visualization.png!"
    
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given financial analysis task and the agent steps: {agent_memory.get_succinct_steps()}. "
        f"Now here is the financial data visualization that was produced.\n\n"
        "Please evaluate this visualization against these criteria:\n"
        "1. Accuracy - Does it correctly represent the financial data?\n"
        "2. Clarity - Is the financial information clear and understandable?\n"
        "3. Professionalism - Is the visualization appropriate for financial analysis?\n"
        "4. Sources - Are market data sources properly cited?\n"
        "5. Appropriateness - Is this the right visualization type for this financial data?\n\n"
        "First list reasons why the visualization succeeds or fails for each criterion, then provide your final decision: "
        "PASS if it meets financial data visualization standards, FAIL if it does not.\n\n"
        "A financial visualization should PASS if it accurately represents the market data, even if there are minor issues."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    
    output = multimodal_model(messages).content
    print("Financial Visualization Feedback: ", output)
    
    if "FAIL" in output:
        raise Exception(output)
    
    return True

# 5. Manager Agent for Financial Market Analysis
def create_financial_manager_agent(market_agent, news_agent, technical_agent, model_name="gpt-4.1-mini"):
    """Create a manager agent that coordinates the financial market analysis process"""
    
    # Create the manager agent with appropriate tools and sub-agents
    manager_agent = CodeAgent(
        model=OpenAIServerModel(
            model_name,
            max_tokens=8096,
        ),
        tools=[],
        managed_agents=[market_agent, news_agent, technical_agent],
        additional_authorized_imports=[
            "pandas", 
            "numpy", 
            "plotly",
            "plotly.express",
            "plotly.graph_objects",
            "json",
            "re",
            "datetime",
            "matplotlib",
            "seaborn",
            "scipy.stats",
            "sklearn.preprocessing",
            "statsmodels.api"
        ],
        planning_interval=5,
        verbosity_level=2,
        final_answer_checks=[validate_financial_visualization],
        max_steps=25,
    )
    
    manager_agent.logger.console.width = 80
    return manager_agent

# ====== USAGE FUNCTION ======

def run_financial_analysis(query, symbols=None):
    """Run a complete financial market analysis on the given query and symbols"""
    
    # Check API keys before proceeding
    try:
        check_api_keys()
    except ValueError as e:
        logger.error(f"API key error: {str(e)}")
        return str(e)
    
    # Set up the model name - you can change this based on available models
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
    
    try:
        # Create the agent hierarchy
        logger.info("Setting up Financial Market Analysis Agents...")
        market_agent = create_market_data_agent(model_name)
        news_agent = create_news_analysis_agent(model_name)
        technical_agent = create_technical_analysis_agent(model_name)
        
        # Create the manager agent
        manager_agent = create_financial_manager_agent(
            market_agent, news_agent, technical_agent, model_name
        )
        
        # Remove previous visualization file if it exists
        if os.path.exists("financial_visualization.png"):
            os.remove("financial_visualization.png")
        
        # Prepare symbols list if provided
        symbols_str = ""
        if symbols:
            if isinstance(symbols, str):
                symbols_str = symbols
            elif isinstance(symbols, list):
                symbols_str = ", ".join(symbols)
        
        # Add specific instructions for visualization
        full_query = f"""
        {query}
        
        {f'Focus your analysis on these symbols: {symbols_str}' if symbols_str else ''}
        
        Based on your research, create a comprehensive financial visualization that clearly presents the key findings.
        The visualization should:
        1. Be professional and suitable for financial analysts
        2. Clearly indicate data sources and time periods
        3. Visualize market trends, correlations, or comparative performance
        4. Use appropriate colors and design for financial data (avoid using red/green if showing non-gain/loss data)
        
        Save the visualization as financial_visualization.png and return a summary of your findings.
        
        Example visualization code:
        ```python
        import plotly.graph_objects as go
        import pandas as pd
        
        # Create dataframe with findings
        df = pd.DataFrame({{
            'Date': pd.date_range(start='2023-01-01', periods=30),
            'AAPL': [150, 151, 149, ...],  # Sample data
            'MSFT': [250, 253, 248, ...],  # Sample data
            'GOOGL': [2100, 2090, 2110, ...],  # Sample data
        }})
        
        # Create visualization
        fig = go.Figure()
        
        # Add traces for each stock
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['AAPL'],
            mode='lines',
            name='AAPL',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['MSFT'],
            mode='lines',
            name='MSFT',
            line=dict(color='purple')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['GOOGL'],
            mode='lines',
            name='GOOGL',
            line=dict(color='orange')
        ))
        
        # Layout customization
        fig.update_layout(
            title='Comparative Stock Performance',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend_title='Symbols',
            template='plotly_white',
            annotations=[
                dict(
                    text='Source: Alpha Vantage API, Daily Close Prices',
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15
                )
            ]
        )
                          
        fig.write_image("financial_visualization.png")
        ```
        
        Do not invent any financial data! You must only use information sourced from the APIs or reliable financial websites.
        """
        
        # Run the analysis
        logger.info(f"Starting Financial Market Analysis on: {query}")
        result = manager_agent.run(full_query)
        
        logger.info("\n===== Financial Market Analysis Completed =====")
        return result
    
    except Exception as e:
        import traceback
        logger.error(f"\n===== ERROR DURING FINANCIAL MARKET ANALYSIS =====")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nStack trace:")
        traceback.print_exc()
        
        return f"Analysis failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    analysis_query = """
    Analyze the recent performance of major tech stocks (AAPL, MSFT, GOOGL) over the past month.
    Compare their price movements, trading volumes, and key technical indicators.
    Identify any patterns or correlations between these stocks.
    Include relevant financial news that might explain their performance.
    """
    
    results = run_financial_analysis(analysis_query, symbols=["AAPL", "MSFT", "GOOGL"])
    print(results)
