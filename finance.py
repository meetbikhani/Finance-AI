from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import pymysql
from contextlib import contextmanager
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        ssl={'ssl': True}
    )
    try:
        yield conn
    finally:
        conn.close()


@tool
def Sql_runner(query: str) -> str:
    """Run a SQL query and return the results."""
    print(f"Running query: {query}")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            if cursor.description:
                # SELECT query - fetch results
                results = cursor.fetchall()
                cursor.close()
                print("Query Results:", results)
                
                if not results:
                    return "Query executed successfully. No results found."
                return f"Query executed successfully. Results: {results}"
            else:
                # INSERT/UPDATE/DELETE query
                rows_affected = cursor.rowcount
                conn.commit()
                cursor.close()
                print(f"Query executed successfully! Rows affected: {rows_affected}")
                return f"Query executed successfully! Rows affected: {rows_affected}"
                
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(error_msg)
        return error_msg


@tool
def send_email(subject: str, body: str) -> str:
    """
    Send an email with the given subject and body to a fixed email address.
    
    Args:
        subject: The subject line of the email
        body: The HTML or plain text body of the email
    
    Returns:
        Success or error message
    """
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient_email
        
        # Add body as HTML
        html_part = MIMEText(body, "html")
        message.attach(html_part)
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        print(f"Email sent successfully to {recipient_email}")
        return f"Email sent successfully to {recipient_email} with subject: '{subject}'"
        
    except Exception as e:
        error_msg = f"Error sending email: {str(e)}"
        print(error_msg)
        return error_msg


tools = [Sql_runner, send_email]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    with open("newpromptfor finace.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()
    system_prompt = SystemMessage(content=prompt_text)
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END
})
graph.add_edge("tools", "our_agent")
app = graph.compile()

def main():
    print("Finance Tracker Agent Started!")
    print("You can now track expenses and email reports!")
    
    conversation_state = {
        "messages": []
    }
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        current_input = {
            "messages": conversation_state["messages"] + [("user", user_input)]
        }
        
        final_state = None
        printed_messages = set()  
        
        for s in app.stream(current_input, stream_mode="values"):
            final_state = s
            message = s["messages"][-1]
            
            msg_id = id(message)
            
            if msg_id not in printed_messages:
                if isinstance(message, tuple):
                    pass
                elif hasattr(message, 'type'):
                    if message.type == 'ai' and hasattr(message, 'content') and message.content:
                        print("Bot: ", message.content, flush=True)
                        printed_messages.add(msg_id)
        
        if final_state:
            conversation_state = final_state
            
        print() 


if __name__ == "__main__":
    main()