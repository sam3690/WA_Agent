import os
import json
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

#Langgraph
from langgraph.graph import StateGraph, START, END

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-key")
BRAND_NAME = os.getenv("BRAND_NAME", "Velour Fragrances")
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "PKR")
DELIVERY_TIME_WINDOW = os.getenv("DELIVERY_TIME_WINDOW", "2-4 business days")
RETURN_POLICY = os.getenv("RETURN_POLICY", "Returns accepted within 7 days if unopened.")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in environment variables.")

client = OpenAI(
        base_url="https://openrouter.ai/api/v1", 
        api_key=OPENROUTER_API_KEY
    )


CATALOG = [
    {
        "id": "ou1",
        "name": "Oud Royale",
        "brand": BRAND_NAME,
        "price": 4800,
        "notes": ["oud", "amber", "spice"],
        "gender": "unisex",
        "occasion": ["evening", "formal"],
        "strength": "EDP",
        "stock": 12,
    },
    {
        "id": "rs1",
        "name": "Rose Noir",
        "brand": BRAND_NAME,
        "price": 3900,
        "notes": ["rose", "patchouli", "musk"],
        "gender": "female",
        "occasion": ["date", "evening"],
        "strength": "EDT",
        "stock": 7,
    },
    {
        "id": "cy1",
        "name": "Citrus Yuzu",
        "brand": BRAND_NAME,
        "price": 3200,
        "notes": ["yuzu", "citrus", "green"],
        "gender": "male",
        "occasion": ["daytime", "office", "summer"],
        "strength": "EDT",
        "stock": 20,
    },
]


#Tools


#Search Catalog Tool
def search_catalog(criteria: Dict[str, Any]) -> List[Dict[str, Any]]: 
    """Very simple filter on the in-memory catalog.
    criteria can include: max_price, gender, occasion, notes(list[str])
    """
    results = []
    for item in CATALOG:
            if "max_price" in criteria and item["price"] > int(criteria["max_price"]):
                continue
            if "gender" in criteria and criteria["gender"] and item["gender"] != criteria["gender"]:
                continue
            if "occasion" in criteria and criteria["occasion"] and criteria["occasion"] not in item["occasion"]:
                continue
            if "notes" in criteria and criteria["notes"]:
                if not any(n.lower() in [x.lower() for x in item["notes"]] for n in criteria["notes"]):
                    continue
            results.append(item)
    return results

# Cart Type
CARTS: Dict[str, List[Dict[str, Any]]] = {}


# Add to Cart Tool
def add_to_cart(user_id:str , product_id: str, qty: int=1) -> Dict[str, Any]:
    prod = next((p for p in CATALOG if p["id"] == product_id), None)
    if not prod:
        return {"ok": False, "error": "Product not found."}
    CARTS.setdefault(user_id, [])
    CARTS[user_id].append({"id": prod["id"], "name": prod["name"], "price":
    prod["price"], "qty": qty})
    return {"ok": True, "cart": CARTS[user_id]}


# View Cart Tool
def view_cart(user_id:str)-> Dict[str,Any]:
    cart = CARTS.get(user_id, [])
    total = sum(item["price"] * item["qty"] for item in cart)
    return {"ok": True, "cart": cart, "total": total}

# Checkout Tool
def checkout(user_id: str, customer: Dict[str, str]) -> Dict[str, Any]:
    cart = CARTS.get(user_id, [])
    if not cart:
        return {"ok": False, "error": "Cart is empty."}
    total = sum(item["price"] * item["qty"] for item in cart)
    order = {
    "order_id": f"ORD-{user_id}-{len(cart)}",
    "items": cart,
    "total": total,
    "currency": DEFAULT_CURRENCY,
    "customer": customer,
    "payment": "Cash on Delivery",
    "delivery": DELIVERY_TIME_WINDOW,
    }
    # TODO: persist to DB, notify internal ops, etc.
    CARTS[user_id] = []
    return {"ok": True, "order": order}

# FAQ Tool
def faq_answer(question: str) -> str:
    q = question.lower()
    if "deliver" in q or "shipping" in q:
        return f"Delivery time is {DELIVERY_TIME_WINDOW}."
    if "return" in q or "refund" in q:
        return RETURN_POLICY
    return "You can ask about delivery, returns, prices, recommendations, or place an order."


# LangGraph State + Nodes
class AgentState(TypedDict, total=False):
    user_id: str
    user_text: str
    intent: str
    plan: Dict[str, Any]
    tool_result: Any
    reply: str

#System Prompt
SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for a fragrance brand. "
    f"Brand: {BRAND_NAME}. Currency: {DEFAULT_CURRENCY}. "
    "You MUST output a strict JSON object describing the next action as: "
    '{"intent": "recommend|faq|add_to_cart|view_cart|checkout|smalltalk","criteria":{...}, "product_id":"...", "qty":1, "question":"...", "customer":{"name":"","address":"","phone":""}}. '
    " include only fields relevant to the intent. Do not add extra commentary."
)

def llm_plan(user_text: str) -> Dict[str, Any]:
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json"},
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"intent": "smalltalk"}
    return data

# Node 1: Classify/Plan

def node_plan(state: AgentState) -> AgentState:
    plan = llm_plan(state.get("user_text", ""))
    state["plan"] = plan
    state["intent"] = plan.get("intent", "smalltalk")
    return state

# Node 2: Execution of Tools

def node_act(state: AgentState) -> AgentState:
    intent = state.get("intent", "smalltalk")
    plan = state.get("plan", {})
    user_id = state.get("user_id", "guest")
    
    if intent == "recommend":
        criteria = plan.get("criteria", {})
        result = search_catalog(criteria)
        state["tool_result"] = {"type": "search_results", "results": result}
        
    elif intent == "add_to_cart":
        pid = plan.get("product_id", "")
        qty = int(plan.get("qty", 1))
        result = add_to_cart(user_id, pid, qty)
        state["tool_result"] = result
        
    elif intent == "view_cart":
        result = view_cart(user_id)
        state["tool_result"] = result
        
    elif intent == "checkout":
        customer = plan.get("question", {})
        result = checkout(user_id, customer)
        state["tool_result"] = result
        
    elif intent == "faq":
        question = plan.get("question", state.get("user_text", ""))
        answer = faq_answer(question)
        state["tool_result"] = {"type": "faq_answer", "answer": answer}

    else:
        state["tool_result"] = {"type": "smalltalk"}
        
    return state