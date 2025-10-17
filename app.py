from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import json
from urllib import request, parse
from io import StringIO  # For in-memory CSV handling
from typing import Optional

app = FastAPI(title="LLM-Based Data Analyst Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "AIzaSyA6mMMo6vWimpykdj995vblNeoYN0ti6qc"

def call_gemini(prompt, model="gemini-1.5-flash-latest"):
    """
    Calls the Gemini API to generate content based on the prompt.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 0.8,
            "maxOutputTokens": 1024,
        }
    }
    data_str = json.dumps(data)
    req = request.Request(url, data=data_str.encode('utf-8'), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            result = json.loads(response_body)
            
            # Debug: Print the full response for troubleshooting (uncomment for debugging)
            # print(f"DEBUG - Full Gemini API response: {result}")
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                    return candidate['content']['parts'][0]['text']
                else:
                    return "Error: Invalid candidate structure in response."
            elif 'error' in result:
                return f"Error: {result['error'].get('message', 'Unknown API error')}"
            else:
                return "Error: No candidates in response."
    except Exception as e:
        return f"Error: {str(e)}"

def setup_db(csv_content: str):
    """
    Loads CSV content into an in-memory SQLite database.
    Returns the DB connection.
    """
    conn = sqlite3.connect(':memory:')  # In-memory for security
    df = pd.read_csv(StringIO(csv_content))  # Read from memory
    df.to_sql('data', conn, if_exists='replace', index=False)
    return conn

def get_schema(conn):
    """
    Retrieves the schema of the 'data' table.
    """
    schema_df = pd.read_sql("PRAGMA table_info(data);", conn)
    return schema_df.to_string()

def generate_fallback_sql(question: str, columns: list) -> str:
    """
    Generate a basic SQL query as fallback when Gemini API fails.
    """
    question_lower = question.lower()
    
    # Basic keyword matching for common queries
    if any(word in question_lower for word in ['average', 'mean', 'avg']):
        # Look for numeric columns
        numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['salary', 'price', 'amount', 'value', 'cost', 'revenue', 'income', 'age', 'count', 'number'])]
        if numeric_cols:
            if any(word in question_lower for word in ['by', 'group', 'department', 'category', 'region']):
                # Group by analysis
                group_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['department', 'category', 'region', 'type', 'group', 'class'])]
                if group_cols:
                    return f"SELECT {group_cols[0]}, AVG({numeric_cols[0]}) as average_{numeric_cols[0]} FROM data GROUP BY {group_cols[0]}"
                else:
                    return f"SELECT AVG({numeric_cols[0]}) as average_{numeric_cols[0]} FROM data"
            else:
                return f"SELECT AVG({numeric_cols[0]}) as average_{numeric_cols[0]} FROM data"
    
    elif any(word in question_lower for word in ['sum', 'total']):
        numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['salary', 'price', 'amount', 'value', 'cost', 'revenue', 'income', 'count', 'number'])]
        if numeric_cols:
            if any(word in question_lower for word in ['by', 'group', 'department', 'category', 'region']):
                group_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['department', 'category', 'region', 'type', 'group', 'class'])]
                if group_cols:
                    return f"SELECT {group_cols[0]}, SUM({numeric_cols[0]}) as total_{numeric_cols[0]} FROM data GROUP BY {group_cols[0]}"
                else:
                    return f"SELECT SUM({numeric_cols[0]}) as total_{numeric_cols[0]} FROM data"
    
    elif any(word in question_lower for word in ['count', 'how many', 'number of']):
        if any(word in question_lower for word in ['by', 'group', 'department', 'category', 'region']):
            group_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['department', 'category', 'region', 'type', 'group', 'class'])]
            if group_cols:
                return f"SELECT {group_cols[0]}, COUNT(*) as count FROM data GROUP BY {group_cols[0]}"
            else:
                return "SELECT COUNT(*) as total_count FROM data"
        else:
            return "SELECT COUNT(*) as total_count FROM data"
    
    elif any(word in question_lower for word in ['max', 'maximum', 'highest']):
        numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['salary', 'price', 'amount', 'value', 'cost', 'revenue', 'income', 'age', 'count', 'number'])]
        if numeric_cols:
            return f"SELECT MAX({numeric_cols[0]}) as max_{numeric_cols[0]} FROM data"
    
    elif any(word in question_lower for word in ['min', 'minimum', 'lowest']):
        numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['salary', 'price', 'amount', 'value', 'cost', 'revenue', 'income', 'age', 'count', 'number'])]
        if numeric_cols:
            return f"SELECT MIN({numeric_cols[0]}) as min_{numeric_cols[0]} FROM data"
    
    # Default: return all data
    return "SELECT * FROM data LIMIT 10"

def generate_fallback_explanation(question: str, sql_query: str, result: list) -> str:
    """
    Generate a basic explanation when Gemini API fails.
    """
    if not result:
        return "No data was returned from the query."
    
    num_results = len(result)
    
    if "AVG" in sql_query.upper():
        return f"The analysis shows the average values across {num_results} records. The query used was: {sql_query}"
    elif "SUM" in sql_query.upper():
        return f"The analysis shows the total values across {num_results} records. The query used was: {sql_query}"
    elif "COUNT" in sql_query.upper():
        return f"The analysis shows the count of records: {num_results} total records found. The query used was: {sql_query}"
    elif "MAX" in sql_query.upper():
        return f"The analysis shows the maximum values across {num_results} records. The query used was: {sql_query}"
    elif "MIN" in sql_query.upper():
        return f"The analysis shows the minimum values across {num_results} records. The query used was: {sql_query}"
    else:
        return f"The analysis returned {num_results} records based on your question. The query used was: {sql_query}"

def data_analyst_assistant(conn, question: str):
    """
    Generates SQL via Gemini, executes it, and explains results.
    """
    schema = get_schema(conn)
    
    # First, try to get column names from schema for fallback
    schema_df = pd.read_sql("PRAGMA table_info(data);", conn)
    columns = schema_df['name'].tolist()
    
    # Try Gemini API first
    sql_prompt = f"""You are an expert data analyst. The database has a single table named 'data'.
Schema:
{schema}

User question: "{question}"

Translate this to a precise SQL query that answers the question. Consider ambiguities like date ranges (e.g., 'last quarter' might mean the most recent 3 months based on data).

IMPORTANT: You must respond with ONLY the SQL query wrapped in ```sql and ``` markers. Do not include any other text, explanations, or formatting.

Example format:
```sql
SELECT column1, column2 FROM data WHERE condition;
```

Your response:"""
    
    sql_response = call_gemini(sql_prompt)
    
    # Debug: Print the raw response for troubleshooting (uncomment for debugging)
    # print(f"DEBUG - Raw Gemini response: {sql_response}")
    
    # Check if Gemini API failed
    if sql_response.startswith("Error:") or "Error:" in sql_response:
        # print("DEBUG - Gemini API failed, using fallback SQL generation")
        # Fallback: Generate basic SQL based on question keywords
        sql_query = generate_fallback_sql(question, columns)
        if not sql_query:
            return {"error": f"Gemini API failed and fallback SQL generation failed. Gemini error: {sql_response}", "sql_query": None, "result": None}
    else:
        # Extract SQL with improved parsing
        sql_query = None
        
        # Try multiple extraction methods
        try:
            # Method 1: Look for ```sql markers
            if '```sql' in sql_response:
                sql_query = sql_response.split('```sql')[1].split('```')[0].strip()
            # Method 2: Look for ``` markers (fallback)
            elif '```' in sql_response:
                parts = sql_response.split('```')
                if len(parts) >= 2:
                    sql_query = parts[1].strip()
            # Method 3: Look for SELECT statements directly
            elif 'SELECT' in sql_response.upper():
                lines = sql_response.split('\n')
                sql_lines = []
                in_sql = False
                for line in lines:
                    if 'SELECT' in line.upper():
                        in_sql = True
                    if in_sql:
                        sql_lines.append(line)
                        if line.strip().endswith(';') or (line.strip() and not line.strip().startswith('--')):
                            break
                sql_query = '\n'.join(sql_lines).strip()
            # Method 4: If response looks like pure SQL, use it directly
            elif sql_response.strip().upper().startswith('SELECT'):
                sql_query = sql_response.strip()
            else:
                # If Gemini response is not parseable, try fallback
                # print("DEBUG - Gemini response not parseable, using fallback SQL generation")
                sql_query = generate_fallback_sql(question, columns)
                if not sql_query:
                    return {"error": f"Error extracting SQL query from Gemini response. Raw response: {sql_response[:200]}...", "sql_query": None, "result": None}
                
            # Clean up the SQL query
            if sql_query:
                # Remove any remaining markdown formatting
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                # Remove any leading/trailing quotes
                sql_query = sql_query.strip('"\'')
                
            if not sql_query:
                return {"error": f"Error extracting SQL query from Gemini response. Raw response: {sql_response[:200]}...", "sql_query": None, "result": None}
            
            # Additional validation - check if it looks like SQL
            sql_upper = sql_query.upper().strip()
            if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH') or sql_upper.startswith('INSERT') or sql_upper.startswith('UPDATE') or sql_upper.startswith('DELETE')):
                # If validation fails, try fallback
                # print("DEBUG - Generated SQL validation failed, using fallback SQL generation")
                sql_query = generate_fallback_sql(question, columns)
                if not sql_query:
                    return {"error": f"Extracted text does not appear to be a valid SQL query: {sql_query[:100]}...", "sql_query": None, "result": None}
                
        except Exception as e:
            # print(f"DEBUG - SQL extraction failed: {e}, using fallback SQL generation")
            sql_query = generate_fallback_sql(question, columns)
            if not sql_query:
                return {"error": f"Error extracting SQL query from Gemini response: {str(e)}", "sql_query": None, "result": None}
    
    # Execute query
    try:
        result = pd.read_sql(sql_query, conn).to_dict(orient='records')  # JSON-friendly
    except Exception as e:
        return {"error": f"Error executing SQL: {str(e)}", "sql_query": sql_query, "result": None}
    
    # Explain insights
    explain_prompt = f"""You are an expert data analyst.
User question: "{question}"
SQL query used: {sql_query}
Query results:
{json.dumps(result)}

Provide a clear natural language explanation of the results, including key insights, trends, and any caveats (e.g., data assumptions or limitations). Keep it concise."""
    
    explanation = call_gemini(explain_prompt)
    
    # If Gemini explanation fails, provide a basic explanation
    if explanation.startswith("Error:") or "Error:" in explanation:
        explanation = generate_fallback_explanation(question, sql_query, result)
    
    return {"explanation": explanation, "result": result, "sql_query": sql_query}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), question: str = Form(...)):
    """
    Endpoint to upload CSV and analyze with a question.
    """
    try:
        csv_content = await file.read()  # Read file content
        csv_str = csv_content.decode('utf-8')  # Decode to string
        conn = setup_db(csv_str)
        response = data_analyst_assistant(conn, question)
        conn.close()
        return response
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
