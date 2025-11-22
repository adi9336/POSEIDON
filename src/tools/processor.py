import os
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.messages import HumanMessage


# ---------------------------
# 1. Load API Key
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------
# 2. CSV → SQLite
# ---------------------------
DB_PATH = "argo_data.db"
CSV_PATH = "data.csv"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS argo_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform_number TEXT,
            latitude REAL,
            longitude REAL,
            time TEXT,
            temp REAL,
            psal REAL,
            pres REAL,
            nitrate REAL
        );
    """)

    cur.execute("SELECT COUNT(*) FROM argo_data;")
    count = cur.fetchone()[0]

    if count == 0:
        df = pd.read_csv(CSV_PATH)
        df.to_sql("argo_data", conn, if_exists="append", index=False)
        print(f"✔ Imported {len(df)} rows.")
    else:
        print(f"✔ Database already has {count} rows.")

    conn.commit()
    conn.close()


# ---------------------------
# 3. Build SQL Agent
# ---------------------------


def sql_to_df(sql_query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df


def build_sql_agent():
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    sql_tool = QuerySQLDataBaseTool(db=db)
    return llm, sql_tool


# ---------------------------
# NLP SUMMARY
# ---------------------------
def generate_nlp_summary(df, user_query):
    if df.empty:
        print("\n⚠ No data found for NLP summary.")
        return

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    msg = f"""
User asked: {user_query}

Here is data description:
{df.describe().to_string()}

Explain this in simple Hinglish + easy ocean science terms.
"""

    res = llm.invoke([HumanMessage(content=msg)])
    print("\n=== NLP SUMMARY ===")
    print(res.content)


# ---------------------------
# GRAPH GENERATION
# ---------------------------
def plot_graph(df):
    if df.empty:
        print("\n⚠ No data to plot.")
        return

    # Depth vs Temperature
    if "temp" in df.columns:
        plt.figure()
        plt.plot(df["pres"], df["temp"])
        plt.xlabel("Depth (pres)")
        plt.ylabel("Temperature (°C)")
        plt.title("Depth vs Temperature")
        plt.show()

    # Depth vs Salinity
    if "psal" in df.columns:
        plt.figure()
        plt.plot(df["pres"], df["psal"])
        plt.xlabel("Depth (pres)")
        plt.ylabel("Salinity")
        plt.title("Depth vs Salinity")
        plt.show()

    # Depth vs Nitrate
    if "nitrate" in df.columns:
        plt.figure()
        plt.plot(df["pres"], df["nitrate"])
        plt.xlabel("Depth (pres)")
        plt.ylabel("Nitrate")
        plt.title("Depth vs Nitrate")
        plt.show()


# ---------------------------
# 4. SQL GENERATOR + NLP + GRAPH
# ---------------------------
def run_query(llm, sql_tool, query):
    print("\n=== USER QUERY ===")
    print(query)

    prompt = f"""
You are an SQL generator. Convert user query into CLEAN SQL ONLY.

STRICT RULES:
- NO markdown
- NO backticks
- NO comments
- Only SQL

Use table: argo_data
Columns: platform_number, latitude, longitude, time, temp, psal, pres, nitrate.
Depth = pres.

User Query: {query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = response.content.strip()

    print("\n=== GENERATED SQL ===")
    print(sql_query)

    # NEW — direct SQL → DataFrame (NO parsing problems)
    df = sql_to_df(sql_query)

    print("\n=== DATAFRAME ===")
    print(df.head())


    # NLP Summary
    generate_nlp_summary(df, query)

    # Graphs
    plot_graph(df)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    init_db()

    llm, sql_tool = build_sql_agent()

    print("\n=== ARGO DATA QUERY ASSISTANT ===")
    print("Example: January 2024 me 0 se 50 depth ka temperature do\n")

    user_query = input("Bol be bhed ke chutad kya bataun: ")

    run_query(llm, sql_tool, user_query)