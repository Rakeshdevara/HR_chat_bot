from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from neo4j import GraphDatabase
from IPython.display import HTML
import os
import warnings

# Initialize Flask
app = Flask(__name__)

# ---- ENVIRONMENT VARIABLES ----
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Neo4j setup
neo4j_uri = "neo4j+s://d22644da.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "7aidZZ4FP6_BS6QCuDLvba5aKq03bQ3Eaj43MlRysPc"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Suppress warnings
warnings.filterwarnings("ignore")

# Load index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(service_context=storage_context)

# Prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If the information contains a table, include the table in the response.

Context: {context}
Graph Insights: {graph_insights}
Question: {question}

Answer the question, include tables if relevant, and provide additional helpful information.
"""

context = "This directory contains multiple documents providing examples and solutions for various Human resource tasks within an organization."

# Function to get graph insights from Neo4j
def get_graph_insights(question):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($question)
            OPTIONAL MATCH (c)-[r:RELATED_TO]->(other:Concept)
            RETURN c.name AS concept, collect(other.name) AS related_concepts
            """,
            question=question
        )
        insights = []
        for record in result:
            insights.append(f"Concept: {record['concept']}, Related Concepts: {', '.join(record['related_concepts'])}")
        return "\n".join(insights) if insights else "No relevant graph insights found."

# Function to format tables for HTML display
def format_table_cleanly(table_text):
    rows = table_text.strip().split("\n")
    html = "<table style='border: 1px solid black; border-collapse: collapse; width: 80%; margin: 10px 0;'>"
    for i, row in enumerate(rows):
        cells = row.split("|")
        tag = "th" if i == 0 else "td"  # Use <th> for header row
        html += "<tr>" + "".join([f"<{tag} style='border: 1px solid black; padding: 10px; text-align: left;'>{cell.strip()}</{tag}>" for cell in cells]) + "</tr>"
    html += "</table>"
    return html

# Flask route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Flask route to handle form submission
@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question")
    graph_insights = get_graph_insights(question)
    query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=question)
    response = query_engine.query(query_prompt)

    # Prepare the response
    response_text = response.response
    tables_html = ""
    if hasattr(response, "tables") and response.tables:
        tables_html = "".join([format_table_cleanly(table) for table in response.tables])
    return jsonify({"response": response_text, "tables": tables_html})

if __name__ == "__main__":
    app.run(debug=True)
