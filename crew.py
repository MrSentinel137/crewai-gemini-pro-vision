from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.tools import DuckDuckGoSearchRun, tool
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from PIL import Image
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from textwrap import dedent
import json
import os

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.2,
    convert_system_message_to_human=True
)

company = "Apple Inc."
path = "F:/crew/images/apple.png"

class Vision:
    @tool("Graph image analysis")
    def vision(inp):
      """
      This is a tool used to classify images

      Parameters: 
      - prompt: this is the prompt based provided to the tool, it is basically a query to the tool 
      - path: This is the full path of the image in question

      {"prompt": prompt, "path": path}

      Returns:
      Response from the vision model based on the prompt and image provided
      """

      inp = json.loads(inp)

      image = Image.open(inp['path'])
      model = genai.GenerativeModel(model_name="gemini-pro-vision")
      response = model.generate_content([inp['prompt'], image])

      return response.text
    
class FinancialData:
    @tool("Financial Data collector")
    def scrape(inp):
        """
        This is a tool used to collect financial stock data

        Parameters: 
        - exchange: stock exchange MIC for the stock involved 
        - ticker: ticker symbol of the stock

        {"exchange": exchange, "ticker": ticker}

        Returns:
        Response based on the input
        """
        data = json.loads(inp)

        BASE_URL = "https://www.google.com/finance"
        INDEX = data['exchange']
        SYMBOL = data['ticker']
        LANGUAGE = "en"
        TARGET_URL = f"{BASE_URL}/quote/{SYMBOL}:{INDEX}?hl={LANGUAGE}"

        page = requests.get(TARGET_URL)
        soup = BeautifulSoup(page.content, "html.parser")

        items = soup.find_all("div", {"class": "gyFHrc"})

        stock_description = {}
        for item in items:
            item_description = item.find("div", {"class": "mfs7Fc"}).text
            item_value = item.find("div", {"class": "P6K39c"}).text
            stock_description[item_description] = item_value


        print(stock_description)
        return stock_description
       

search_tool = DuckDuckGoSearchRun()
finance_tool = FinancialData().scrape
vision_tool = Vision().vision

stock_analyst = Agent(
    role='Financial Analyst',
    goal="""Impress all customers with your financial data 
      and market trends analysis""",
    backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
    
    verbose=True,
    llm=llm,
    tools=[finance_tool],
    allow_delegation=False
)

image_analyst = Agent(
    role = "Image Analyst",
    goal = "Analyse the contents of the image and classify",
    backstory="""You are one of the most exprienced image analyst and classifier to ever exist
    you have knowledge of all images every clicked.""",
    tools=[vision_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

research_analyst = Agent(
     role='Research Analyst',
    goal="""Being the best at gather, interpret data and amaze
      your customer with it""",
    backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",

    llm = llm,
    tools=[search_tool],
    verbose=True,
    allow_delegation=True
)

advisor = Agent(
    role='Investment Advisor',
    goal="""Compile investment analysis""",
    backstory="""You are one of the most exprienced infromation compilers,
    compile all the data collected.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

research_task = Task(
    description=dedent(f"""
        Collect and summarize recent news articles, press
        releases, and market analyses related to the stock and
        its industry.
        Pay special attention to any significant events, market
        sentiments, and analysts' opinions. Also include upcoming 
        events like earnings and others.
  
        Your final answer MUST be a report that includes a
        comprehensive summary of the latest news, any notable
        shifts in market sentiment, and potential impacts on 
        the stock.
        Also make sure to return the stock ticker.
        
        If you do your BEST WORK, I'll give you a $10,000 commission!
  
        Make sure to use the most recent data as possible.
  
        Selected company by the customer: {company}
      """),
    agent=research_analyst
)

financial_task = Task(
    description=dedent(f"""
        Conduct a thorough analysis of the stock's financial
        health and market performance based on the ticker of the company: {company}

        Your final report MUST expand on the summary provided
        but now including a clear assessment of the stock's
        financial standing, its strengths and weaknesses, 
        and how it fares against its competitors in the current
        market scenario. 
        
        If you do your BEST WORK, I'll give you a $10,000 commission!

        Make sure to use the most recent data possible.
      """),
    agent=stock_analyst
)

recommend_task = Task(
    description=dedent(f"""
        Review and synthesize the analyses provided by the
        Financial Analyst AND ALSO the Research Analyst.
        Combine BOTH these insights to form a comprehensive
        investment recommendation. 
        
        You MUST Consider all aspects, including financial
        health, market sentiment, and qualitative data.

        Make sure to include a section that shows insider 
        trading activity, and upcoming events like earnings.

        Your final answer MUST be a recommendation for your
        customer. It should be a full super detailed report, providing a 
        clear investment stance and strategy with supporting evidence.
        Make it pretty and well formatted for your customer.
        If you do your BEST WORK, I'll give you a $10,000 commission!
      """),
    agent=advisor
)

vision_task = Task(
    description=dedent(f"""
        Image path:- {path}
        This image depicts the stock price graph of {company}, for a duration of 6 months. 
        Your task is to conduct a thorough assessment of the image and 
        provide your perspective on whether the price of {company} is 
        likely to increase or decrease in the near future.
        
        Your final response MUST be a comprehensive report that includes 
        a thorough summary of the likelihood of the stock price either increasing or decreasing.

        If you do your BEST WORK, I'll give you a $10,000 commission!
      """),
    agent=image_analyst
)

crew = Crew(
      agents=[
        research_analyst,
        stock_analyst,
        advisor
    ],
    tasks=[
        financial_task,
        vision_task,
        research_task,
        recommend_task
    ],
    verbose=True,
    process=Process.sequential
)

result = crew.kickoff()