from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq

# --- MODEL ---
model_llm = ChatGroq(model="openai/gpt-oss-120b")

# --- 4 TOOLS ---
@tool
def save_name(name: str) -> str:
    """Saves the customer's name"""
    return f"Nice to meet you, {name}! I have saved your name."

@tool
def get_weather(location: str) -> str:
    """Gets weather for a given location"""
    return f"The weather in {location} is sunny, 32°C - perfect for delivery!"

@tool
def get_shawarma_menu() -> str:
    """Returns available shawarma menu and prices"""
    return "Chicken: N3500, Beef: N4000, Mixed: N5000, Extra Cheese: N500"

@tool
def check_delivery(area: str) -> str:
    """Checks delivery fee and time for Port Harcourt areas"""
    area = area.lower()
    delivery_zones = {
        "d-line": {"fee": "N1000", "time": "25 mins"},
        "gra": {"fee": "N1500", "time": "30 mins"},
        "trans-amadi": {"fee": "N2000", "time": "45 mins"},
        "peter odili": {"fee": "N1500", "time": "35 mins"},
        "artillery": {"fee": "N1000", "time": "20 mins"}
    }
    for zone, details in delivery_zones.items():
        if zone in area:
            return f"Yes! We deliver to {area}. Fee: {details['fee']}, Time: {details['time']}"
    return f"Sorry, we don't deliver to {area}. We cover D-line, GRA, Trans-Amadi, Peter Odili, Artillery."

# --- BIND ALL 4 ---
model_llm_with_tools = model_llm.bind_tools([save_name, get_weather, get_shawarma_menu, check_delivery])

# --- NODES ---
def chat_with_node(state: MessagesState):
    system_msg = SystemMessage(content="You are Amaka Shawarma AI assistant in Port Harcourt. Be friendly.")
    messages = [system_msg] + state["messages"]
    response = model_llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    results = []
    for t in last_message.tool_calls:
        if t["name"] == "save_name":
            result = save_name.invoke(t["args"])
        elif t["name"] == "get_weather":
            result = get_weather.invoke(t["args"])
        elif t["name"] == "get_shawarma_menu":
            result = get_shawarma_menu.invoke(t["args"])
        elif t["name"] == "check_delivery":
            result = check_delivery.invoke(t["args"])
        else:
            result = "Tool not found"
        results.append(ToolMessage(content=str(result), tool_call_id=t["id"]))
    return {"messages": results}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- GRAPH - MUST BE CALLED 'graph' ---
graph = StateGraph(MessagesState)
graph.add_node("chat", chat_with_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "chat")
graph = graph.compile()