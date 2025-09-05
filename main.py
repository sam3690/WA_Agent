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
        "id": "p1",
        "name": "Oud Royale",
        "price": 4800,
        "currency": "PKR",
        "gender": "Unisex",
        "occasion": "Evening, formal",
        "notes": ["Oud", "Amber", "Spice"],
        "strength": "EDP",
    },
    {
        "id": "p2",
        "name": "Rose Noir",
        "price": 3900,
        "currency": "PKR",
        "gender": "Female",
        "occasion": "Date, evening",
        "notes": ["Rose", "Patchouli", "Musk"],
        "strength": "EDT",
    },
    {
        "id": "p3",
        "name": "Citrus Yuzu",
        "price": 3200,
        "currency": "PKR",
        "gender": "Male",
        "occasion": "Daytime, office, summer",
        "notes": ["Yuzu", "Citrus", "Green"],
        "strength": "EDT",
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
def add_to_cart(user_id:str, product_id: str, qty: int=1, set_quantity: bool=False) -> Dict[str, Any]:
    # Log what we're trying to add
    print(f"Attempting to add product '{product_id}' to cart for user {user_id}")
    
    # First try exact ID match (case insensitive)
    prod = next((p for p in CATALOG if p["id"].lower() == product_id.lower()), None)
    
    # If not found by ID, try exact name match (case insensitive)
    if not prod:
        prod = next((p for p in CATALOG if p["name"].lower() == product_id.lower()), None)
    
    # If still not found, try partial name match
    if not prod:
        prod = next((p for p in CATALOG if p["name"].lower() in product_id.lower() or product_id.lower() in p["name"].lower()), None)
    
    if not prod:
        # Debug what was searched for and what was available
        return {
            "ok": False, 
            "error": f"Product not found: '{product_id}'. Available products: {[p['name'] for p in CATALOG]}",
            "catalog": CATALOG,
            "exact_pricing": {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG}
        }
    
    # We found a product, now add it to the cart
    print(f"Found product: {prod['name']} ({prod['id']})")
    
    # Check if item already exists in cart and update quantity instead of adding duplicate
    CARTS.setdefault(user_id, [])
    existing_item = next((item for item in CARTS[user_id] if item["id"] == prod["id"]), None)
    
    # Perform quantity update
    if existing_item:
        if set_quantity:
            # Set to exact quantity
            existing_item["qty"] = qty
            print(f"Set quantity in cart to: {existing_item}")
        else:
            # Add to existing quantity
            existing_item["qty"] += qty
            print(f"Updated quantity in cart: {existing_item}")
    else:
        cart_item = {"id": prod["id"], "name": prod["name"], "price": prod["price"], "qty": qty}
        CARTS[user_id].append(cart_item)
        print(f"Added new item to cart: {cart_item}")
    
    # Return current cart state
    cart = CARTS.get(user_id, [])
    print(f"Current cart for user {user_id}: {cart}")
    
    return {
        "ok": True, 
        "cart": cart,
        "added_product": {
            "id": prod["id"], 
            "name": prod["name"], 
            "price": prod["price"],
            "formatted_price": f"{prod['price']} {prod['currency']}",
            "occasion": prod["occasion"]
        }
    }


# Update Cart Tool
def update_cart(user_id: str, product_id: str, qty: int=1) -> Dict[str, Any]:
    """Update cart quantity or remove item if qty is 0"""
    print(f"Updating cart for user {user_id}, product {product_id}, qty {qty}")
    
    # Get the cart
    cart = CARTS.get(user_id, [])
    if not cart:
        return {"ok": False, "error": "Cart is empty."}
    
    # First try exact ID match (case insensitive)
    prod = next((p for p in CATALOG if p["id"].lower() == product_id.lower()), None)
    
    # If not found by ID, try exact name match (case insensitive)
    if not prod:
        prod = next((p for p in CATALOG if p["name"].lower() == product_id.lower()), None)
    
    # If still not found, try partial name match
    if not prod:
        prod = next((p for p in CATALOG if p["name"].lower() in product_id.lower() or product_id.lower() in p["name"].lower()), None)
    
    if not prod:
        return {
            "ok": False, 
            "error": f"Product not found: '{product_id}'. Cannot update.",
            "cart": cart
        }
    
    # Find the item in the cart
    cart_item = next((item for item in cart if item["id"] == prod["id"]), None)
    
    if not cart_item:
        return {
            "ok": False, 
            "error": f"Product {prod['name']} not in cart. Cannot update.",
            "cart": cart
        }
    
    # Update or remove the item
    if qty <= 0:
        # Remove the item
        CARTS[user_id] = [item for item in cart if item["id"] != prod["id"]]
        print(f"Removed {prod['name']} from cart")
        action = "removed"
    else:
        # Update the quantity
        cart_item["qty"] = qty
        print(f"Updated {prod['name']} quantity to {qty}")
        action = "updated"
    
    updated_cart = CARTS.get(user_id, [])
    print(f"Cart after update: {updated_cart}")
    
    return {
        "ok": True, 
        "cart": updated_cart,
        "updated_product": {
            "id": prod["id"],
            "name": prod["name"],
            "price": prod["price"],
            "formatted_price": f"{prod['price']} {prod['currency']}",
            "occasion": prod["occasion"]
        },
        "action": action
    }
def checkout(user_id: str, customer: Dict[str, str]) -> Dict[str, Any]:
    cart = CARTS.get(user_id, [])
    if not cart:
        return {"ok": False, "error": "Cart is empty."}
    
    # Validate customer info
    if not customer.get("name") or not customer.get("address") or not customer.get("phone"):
        return {
            "ok": False, 
            "error": "Missing customer information. Please provide name, address, and phone number.",
            "cart": cart
        }
    
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


# View Cart Tool
def view_cart(user_id:str)-> Dict[str,Any]:
    cart = CARTS.get(user_id, [])
    total = sum(item["price"] * item["qty"] for item in cart)
    return {"ok": True, "cart": cart, "total": total}


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
    "Our available products are:\n" + 
    "\n".join([f"- {p['name']} (ID: {p['id']}): Price: {p['price']} {p['currency']}, Gender: {p['gender']}, Occasion: {p['occasion']}, Notes: {', '.join(p['notes'])}" for p in CATALOG]) + 
    "\n\nYou MUST output a strict JSON object describing the next action as: "
    '{"intent": "recommend | faq | add_to_cart | update_cart | view_cart | checkout | smalltalk","criteria":{...}, "product_id":"...", "qty":1, "question":"...", "customer":{"name":"","address":"","phone":""}}. '
    "For add_to_cart, ALWAYS use the exact product id (p1, p2, p3) or exact product name. "
    "For update_cart, use when the user wants to SET a specific quantity or REMOVE items. "
    "For view_cart, check what's in the user's cart before recommending more products. "
    "Do not hallucinate products that are not in our catalog. DO NOT change product prices or details. "
    "Include only fields relevant to the intent. Do not add extra commentary."
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
        response_format={"type": "json_object"},
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
    user_text = state.get("user_text", "").lower()
    
    # Format product details for consistent display
    formatted_catalog = []
    for product in CATALOG:
        formatted_catalog.append({
            **product,
            "formatted_price": f"{product['price']} {product['currency']}",
            "formatted_notes": ", ".join(product["notes"]),
        })
    
    # Special handling for cart commands that might not be properly identified by the LLM
    if "remove" in user_text and any(p["name"].lower() in user_text for p in CATALOG):
        for p in CATALOG:
            if p["name"].lower() in user_text:
                # Try to update cart directly
                result = update_cart(user_id, p["id"], 0)  # qty=0 means remove
                state["intent"] = "update_cart"
                state["tool_result"] = result
                return state
    
    if intent == "recommend":
        criteria = plan.get("criteria", {})
        result = search_catalog(criteria)
        # Include formatted catalog for context
        state["tool_result"] = {
            "type": "search_results", 
            "results": result,
            "full_catalog": formatted_catalog,
            "exact_pricing": {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG},
            "exact_occasions": {p["name"]: p["occasion"] for p in CATALOG}
        }
        
    elif intent == "add_to_cart":
        pid = plan.get("product_id", "")
        qty = int(plan.get("qty", 1))
        result = add_to_cart(user_id, pid, qty)
        
        # If cart operation failed, provide the catalog for context
        if not result.get("ok", False):
            result["catalog"] = formatted_catalog
            result["exact_pricing"] = {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG}
            result["exact_occasions"] = {p["name"]: p["occasion"] for p in CATALOG}
            
        state["tool_result"] = result
        
    elif intent == "update_cart":
        pid = plan.get("product_id", "")
        qty = int(plan.get("qty", 1))
        
        # Check if message indicates a removal (quantity = 0)
        if "remove" in user_text or "delete" in user_text or "take out" in user_text:
            qty = 0
        
        # If the user is explicitly asking to set a specific quantity
        if "set" in user_text or "only want" in user_text or "just want" in user_text:
            result = add_to_cart(user_id, pid, qty, set_quantity=True)
        else:
            result = update_cart(user_id, pid, qty)
        
        state["tool_result"] = result
        
    elif intent == "view_cart":
        result = view_cart(user_id)
        # Include catalog with cart view for context
        result["catalog"] = formatted_catalog
        result["exact_pricing"] = {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG}
        result["exact_occasions"] = {p["name"]: p["occasion"] for p in CATALOG}
        state["tool_result"] = result
        
    elif intent == "checkout":
        customer = plan.get("customer", {})
        result = checkout(user_id, customer)
        state["tool_result"] = result
        
    elif intent == "faq":
        question = plan.get("question", state.get("user_text", ""))
        answer = faq_answer(question)
        state["tool_result"] = {
            "type": "faq_answer", 
            "answer": answer,
            "catalog": formatted_catalog,
            "exact_pricing": {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG}
        }

    else:
        state["tool_result"] = {
            "type": "smalltalk",
            "catalog": formatted_catalog,  # Include catalog for context
            "exact_pricing": {p["name"]: f"{p['price']} {p['currency']}" for p in CATALOG},
            "exact_occasions": {p["name"]: p["occasion"] for p in CATALOG}
        }
        
    return state


ASSISTANT_STYLE = (
    "Keep replies concise, friendly, and sales-oriented. Use bullet points for lists."
    f" If showing items, ALWAYS include name, EXACT price with currency which is {DEFAULT_CURRENCY}, the occasion it's best for, and a short scent profile."
    f" NEVER change product prices - always use the exact prices from the catalog data."
)


def llm_reply(user_text:str, intent:str, tool_result: Any) -> str:
    # Create a formatted product list with exact prices and occasions
    product_details = []
    for p in CATALOG:
        product_details.append(f"{p['name']}: {p['price']} {DEFAULT_CURRENCY}, Occasion: {p['occasion']}, Notes: {', '.join(p['notes'])}")
    
    formatted_products = "\n".join(product_details)
    
    sys = (
        f"You are a Whatsapp assistant for {BRAND_NAME}. "
        f"{ASSISTANT_STYLE} "
        f"IMPORTANT: Our product catalog ONLY contains these products with EXACT prices and details:\n{formatted_products}\n "
        f"DO NOT change any product prices or details. Always use the EXACT prices and occasions shown above. "
        f"NEVER mention or recommend products that are not in this list. "
        f"When a user asks to add a product to cart, ALWAYS check if it succeeded by looking at the tool_result."
    )
    
    context = json.dumps(tool_result, ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_text},
        {"role": "system", "content": f"TOOL_RESULT_JSON: {context}"},
        {"role": "system", "content": f"INTENT: {intent}"},
    ]
    
    resp = client.chat.completions.create(
        model = OPENROUTER_MODEL,
        messages= messages,
        temperature=0.3,  # Lower temperature for more consistent replies
    )
    return resp.choices[0].message.content.strip()


# Node 3: Generate Reply
def node_reply(state: AgentState) -> AgentState:
    state["reply"] = llm_reply(
        user_text = state.get("user_text"),
        intent = state.get("intent", "smalltalk"),
        tool_result = state.get("tool_result")
    )
    return state

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", node_plan)
workflow.add_node("act", node_act)
workflow.add_node("reply", node_reply)
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "act")
workflow.add_edge("act", "reply")
workflow.add_edge("reply", END)
app_graph = workflow.compile()


# FastAPI + Twilio Webhook
app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}



@app.post("/webhooks/twilio/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(request: Request):
    try:
        form = await request.form()
        user_text = (form.get("Body") or "").strip()
        user_id = (form.get("From") or "").replace("whatsapp:", "")
        
        # Debug logging
        print(f"Received message from {user_id}: {user_text}")
        print(f"Current carts: {json.dumps(CARTS)}")
        
        # Run through LangGraph 
        state: AgentState = {"user_id": user_id, "user_text": user_text}
        result = app_graph.invoke(state)
        reply = result.get("reply", "Sorry, I had trouble processing that.")
        
        # Debug logging of response
        print(f"Replying with: {reply}")
        
        # TwiML response
        twiml = MessagingResponse()
        twiml.message(reply)
        
        # Return TwiML as PlainTextResponse with correct media type
        return PlainTextResponse(str(twiml), media_type="application/xml")
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        twiml = MessagingResponse()
        twiml.message("Sorry, I'm having technical difficulties. Please try again in a moment.")
        return PlainTextResponse(str(twiml), media_type="application/xml")
