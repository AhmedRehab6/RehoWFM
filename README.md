# RehoWFM – AI Agent for Workforce Management & Data Analysis

An AI agent built with Python and LangChain that works as a senior workforce management (WFM) expert & data analyst. It analyzes Excel and CSV files, forecasts contact volumes, calculates staffing with Erlang C, and builds capacity plans. It can run on either Claude or Gemini.

## What it can do


- **Forecasting**: backtests several methods (seasonal naive, moving average, trend + seasonality, Holt-Winters), picks the most accurate one and reports WAPE, MAPE and bias with an 80% forecast range
- **Intraday planning**: builds interval profiles and day-of-week indexes from history
- **Staffing**: Erlang C staffing per interval, from a file, or from a daily forecast
- **Capacity planning**: required FTE vs. projected headcount with attrition, shrinkage and hires needed
- **Power BI help**: generates and saves DAX and Power Query (M) code
- **Reports**: saves analysis summaries to the output folder
- **Data analysis**: list, inspect and analyze Excel/CSV files, run pandas code, create charts

## How to run

1. Install Python 3.10 or newer.
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a file named `.env` in the project folder with your keys:
   ```
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   GEMINI_MODEL=your_gemini_model_name
   ```
4. Put your Excel/CSV files in a folder named `data`.
5. Start the agent:
   ```
   python main.py
   ```
6. Choose the model (1 = Claude, 2 = Gemini) and type your question.

Results (forecasts, plans, charts, reports) are saved in the `output` folder.

## Notes and limits

- Erlang C ignores abandonment and multi-skill pooling.
- Default assumptions when you give none: service level 80% in 20 seconds, occupancy cap 85%, shrinkage 30%, 30-minute intervals, 40 paid hours per week, 300 seconds AHT, 30% annual attrition.
- The pandas tool runs code on your own machine, so use the agent only on your own computer.

## Roadmap

Erlang A (abandonment), multi-skill pooling, holiday and campaign calendars, shift optimization and a Streamlit web interface.
