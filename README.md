# 🤖 WhatsApp AI Sales Agent

An intelligent conversational sales agent deployed over WhatsApp — built for e-commerce businesses to automate customer interactions, product discovery, and purchase intent qualification.

---

## 📸 Demo

> _Add a screenshot or screen recording of a WhatsApp conversation here_

---

## ✨ Features

- 💬 **Conversational product discovery** — customers describe what they want in plain language, the agent recommends products
- 🧠 **LLM-powered responses** — context-aware replies using LangChain + OpenAI API
- 🛒 **Purchase intent qualification** — identifies buying signals and escalates to human agents when needed
- 🔁 **Objection handling** — responds to common hesitations (price, delivery, returns) automatically
- 📲 **WhatsApp Business API integration** — works natively inside WhatsApp, no app install required for the customer
- 🔌 **Webhook-based architecture** — stateless, scalable, and easy to deploy

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| AI / LLM | LangChain · OpenAI API |
| Messaging | WhatsApp Business API (Meta Cloud API) |
| Backend | FAST API / Python |
| Deployment | Docker |

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ or Python 3.10+
- Meta Developer account with WhatsApp Business API access
- OpenAI API key

### Installation

```bash
git clone https://github.com/sam3690/WA_Agent.git
cd WA_Agent
npm install       # or: pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
WHATSAPP_TOKEN=your_whatsapp_business_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
VERIFY_TOKEN=your_webhook_verify_token
```

### Run Locally

```bash
npm run dev       # or: python main.py
```

For local webhook testing, use [ngrok](https://ngrok.com):

```bash
ngrok http 3000
```

Point your Meta webhook URL to the ngrok HTTPS URL.

---

## 🏗 Architecture

```
Customer (WhatsApp)
        │
        ▼
WhatsApp Business API (Meta)
        │  webhook
        ▼
Backend Server (Node.js / Python)
        │
        ├──▶ LangChain Agent
        │         │
        │         ├──▶ OpenAI API (LLM reasoning)
        │         └──▶ Tool calls (product lookup, FAQ, escalation)
        │
        └──▶ WhatsApp API (send reply)
```

---

## 📁 Project Structure

```
WA_Agent/
├── src/
│   ├── agent/          # LangChain agent setup & tools
│   ├── webhook/        # WhatsApp webhook handler
│   ├── services/       # OpenAI, WhatsApp API clients
│   └── utils/          # Helpers, message formatting
├── .env.example
├── package.json
└── README.md
```

---

## 🤝 Use Cases

- E-commerce stores with high inbound WhatsApp volume
- Businesses replacing manual first-response sales with automation
- Any product catalog that benefits from natural language search

---

## 👤 Author

**Usama Ayoub** — Backend Developer & AI Automation Engineer  
[LinkedIn](https://www.linkedin.com/in/usama-ayoub-972822405/) · [Portfolio](https://sam3690.github.io/UsamaAyoub/) · usamabinayoub@gmail.com

---

## 📄 License

MIT License — feel free to use, fork, and build on this.
