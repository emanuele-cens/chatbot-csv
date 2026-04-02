import os
import re
import json
import time
import threading
import requests
import xml.etree.ElementTree as ET

from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================================================
# CONFIG
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

SITE_URL = "https://www.mgfishing.eu"
SITEMAP_URL = "https://www.mgfishing.eu/1_index_sitemap.xml"

PRODUCT_CACHE_FILE = "products_cache.json"
PRODUCT_CACHE_TTL = 60 * 60 * 12  # 12 ore
AUTO_REFRESH_INTERVAL = 60 * 60 * 6  # refresh automatico ogni 6 ore

REQUEST_TIMEOUT = 25

app = Flask(__name__)
CORS(app)

PRODUCTS = []
LAST_CATALOG_UPDATE = 0


# =========================================================
# UTILS TESTO
# =========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = unquote(text)
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\sàèéìòù]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slug_to_title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""

    last = path.split("/")[-1]
    last = re.sub(r"\.html?$", "", last, flags=re.IGNORECASE)
    last = last.replace("-", " ")
    last = last.replace("_", " ")
    last = re.sub(r"\s+", " ", last).strip()

    return last.title()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =========================================================
# SITEMAP / CATALOGO PRODOTTI
# =========================================================

def is_product_url(url: str) -> bool:
    u = url.lower().strip()

    if not u.startswith("http"):
        return False

    excluded_parts = [
        "/blog",
        "/content",
        "/module",
        "/login",
        "/my-account",
        "/cart",
        "/carrello",
        "/order",
        "/checkout",
        "/search",
        "/brand",
        "/manufacturer",
        "/supplier",
        "/stores",
        "/sitemap",
        "/new-products",
        "/best-sales",
        "/prices-drop",
        "/contatt",
        "/chi-siamo",
        "/pagina",
        "/categoria",
        "/category",
    ]

    for part in excluded_parts:
        if part in u:
            return False

    path = urlparse(u).path.strip("/")

    if not path:
        return False

    # Nei PrestaShop i prodotti sono spesso URL .html
    if ".html" in u:
        return True

    # fallback: ultimo slug sufficientemente "ricco"
    last = path.split("/")[-1]
    if len(last) >= 10 and re.search(r"[a-z]", last):
        return True

    return False


def extract_urls_from_sitemap(sitemap_url: str):
    response = requests.get(sitemap_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for loc in root.findall(".//sm:loc", ns):
        if loc.text:
            urls.append(loc.text.strip())

    return urls


def load_all_sitemap_urls():
    try:
        urls = extract_urls_from_sitemap(SITEMAP_URL)

        child_sitemaps = [u for u in urls if u.lower().endswith(".xml")]
        page_urls = [u for u in urls if not u.lower().endswith(".xml")]

        for child in child_sitemaps:
            try:
                sub_urls = extract_urls_from_sitemap(child)
                page_urls.extend(sub_urls)
            except Exception as e:
                print(f"[SITEMAP] Errore sitemap figlia {child}: {e}")

        # deduplica
        unique = []
        seen = set()
        for url in page_urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)

        return unique

    except Exception as e:
        print(f"[SITEMAP] Errore caricamento sitemap principale: {e}")
        return []


def build_product_catalog():
    urls = load_all_sitemap_urls()
    products = []
    seen = set()

    for url in urls:
        if not is_product_url(url):
            continue

        title = slug_to_title_from_url(url)
        normalized_title = normalize_text(title)

        if not normalized_title:
            continue

        key = (normalized_title, url)
        if key in seen:
            continue
        seen.add(key)

        products.append({
            "title": title,
            "normalized_title": normalized_title,
            "url": url
        })

    return products


def save_product_cache(products):
    payload = {
        "timestamp": int(time.time()),
        "products": products
    }
    with open(PRODUCT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_product_cache():
    if not os.path.exists(PRODUCT_CACHE_FILE):
        return None

    try:
        with open(PRODUCT_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        ts = int(payload.get("timestamp", 0))
        age = int(time.time()) - ts

        if age > PRODUCT_CACHE_TTL:
            return None

        products = payload.get("products", [])
        if not isinstance(products, list):
            return None

        return products

    except Exception as e:
        print(f"[CACHE] Errore lettura cache: {e}")
        return None


def refresh_products(force=False):
    global PRODUCTS, LAST_CATALOG_UPDATE

    cached = load_product_cache()
    if cached and not force:
        PRODUCTS = cached
        LAST_CATALOG_UPDATE = int(time.time())
        print(f"[CATALOGO] Caricato da cache: {len(PRODUCTS)} prodotti")
        return

    fresh_products = build_product_catalog()
    if fresh_products:
        PRODUCTS = fresh_products
        LAST_CATALOG_UPDATE = int(time.time())
        save_product_cache(PRODUCTS)
        print(f"[CATALOGO] Caricato da sitemap: {len(PRODUCTS)} prodotti")
    else:
        # se la sitemap fallisce ma esiste la cache vecchia, prova a riusarla
        if cached:
            PRODUCTS = cached
            LAST_CATALOG_UPDATE = int(time.time())
            print(f"[CATALOGO] Uso cache esistente: {len(PRODUCTS)} prodotti")
        else:
            PRODUCTS = []
            LAST_CATALOG_UPDATE = int(time.time())
            print("[CATALOGO] Nessun prodotto disponibile")


def auto_refresh_catalog():
    while True:
        try:
            time.sleep(AUTO_REFRESH_INTERVAL)
            print("[CATALOGO] Refresh automatico in corso...")
            refresh_products(force=True)
        except Exception as e:
            print(f"[CATALOGO] Errore refresh automatico: {e}")


# =========================================================
# RICERCA PRODOTTI
# =========================================================

def tokenize(text: str):
    return [w for w in normalize_text(text).split() if w]


def score_product_match(query: str, product_normalized_title: str) -> float:
    q = normalize_text(query)
    p = normalize_text(product_normalized_title)

    if not q or not p:
        return 0.0

    if q == p:
        return 1.0

    score = similarity(q, p)

    q_words = set(tokenize(q))
    p_words = set(tokenize(p))

    if q_words and p_words:
        common = q_words.intersection(p_words)
        coverage = len(common) / max(len(q_words), 1)
        score += coverage * 0.25

        # bonus se tutte le parole query sono nel titolo prodotto
        if q_words.issubset(p_words):
            score += 0.20

    if q in p:
        score += 0.15

    if p in q:
        score += 0.10

    return min(score, 1.0)


def extract_possible_product_name(user_message: str) -> str:
    text = normalize_text(user_message)

    patterns = [
        r"(?:link di|link del|link prodotto|url di|url del|mandami il link di|mandami il prodotto|pagina prodotto di)\s+(.+)",
        r"(?:cerco|voglio|mi serve|dammi)\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    return text


def find_best_product(user_query: str):
    if not PRODUCTS:
        return None

    q = extract_possible_product_name(user_query)
    q = normalize_text(q)

    if not q:
        return None

    best = None
    best_score = 0.0

    for product in PRODUCTS:
        s = score_product_match(q, product["normalized_title"])
        if s > best_score:
            best_score = s
            best = product

    if best and best_score >= 0.78:
        return best

    return None


def is_link_request(user_message: str) -> bool:
    msg = normalize_text(user_message)

    keywords = [
        "link",
        "url",
        "pagina prodotto",
        "scheda prodotto",
        "mandami il prodotto",
        "dammi il prodotto",
        "apri il prodotto",
        "prodotto specifico",
        "hai questo prodotto",
        "cerco questo prodotto",
    ]

    return any(k in msg for k in keywords)


# =========================================================
# RISPOSTE RAPIDE
# =========================================================

def get_quick_reply(user_message: str):
    msg = normalize_text(user_message)

    greetings = ["ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"]
    thanks = ["grazie", "perfetto grazie", "ok grazie", "ti ringrazio"]

    if msg in greetings:
        return "Ciao! 👋 Benvenuto su MGFishing Verde Pesca. Se vuoi posso aiutarti a trovare un prodotto, un link specifico oppure darti consigli su attrezzatura e tecniche di pesca."

    if msg in thanks:
        return "Di nulla! Se vuoi, scrivimi pure il nome del prodotto e ti aiuto a trovarlo."

    if msg in ["ok", "va bene", "perfetto"]:
        return "Perfetto 👍 Scrivimi pure cosa stai cercando."

    return None


# =========================================================
# OPENAI
# =========================================================

def call_openai(user_message: str, matched_product=None):
    if not OPENAI_API_KEY:
        return "Al momento il servizio chat non è configurato correttamente. Riprova più tardi."

    system_prompt = f"""
Sei l'assistente virtuale di MGFishing Verde Pesca.
Rispondi sempre in italiano, in modo chiaro, utile e naturale.

Regole fondamentali:
- Non parlare di sitemap, cache, aggiornamenti interni, file, API o aspetti tecnici.
- Non inventare disponibilità, prezzi o dettagli non certi.
- Se conosci il link preciso di un prodotto fornito dal sistema, usalo.
- Se il cliente saluta, rispondi in modo breve e cordiale.
- Se il cliente cerca un prodotto specifico e il sistema lo ha trovato, indica il prodotto con il link diretto.
- Se il prodotto non è stato trovato con certezza, invita il cliente a scrivere il nome completo del prodotto.
- Non mandare link di categoria come se fossero il prodotto esatto.
- Mantieni un tono commerciale ma non insistente.
- Se il cliente chiede consigli di pesca o attrezzatura, rispondi in modo utile e concreto.
"""

    product_context = ""
    if matched_product:
        product_context = (
            f"\nProdotto trovato dal sistema:\n"
            f"Titolo: {matched_product['title']}\n"
            f"URL: {matched_product['url']}\n"
        )

    user_input = f"Messaggio cliente: {user_message}{product_context}"

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.6
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            print(f"[OPENAI] Errore HTTP {response.status_code}: {response.text}")
            return "Al momento non riesco a rispondere correttamente. Riprova tra poco."

        data = response.json()

        # nuovo formato Responses API
        text = data.get("output_text")
        if text and isinstance(text, str) and text.strip():
            return text.strip()

        # fallback robusto
        output = data.get("output", [])
        collected = []

        for item in output:
            content_list = item.get("content", [])
            for c in content_list:
                if c.get("type") == "output_text":
                    piece = c.get("text", "")
                    if piece:
                        collected.append(piece)

        final_text = "\n".join(collected).strip()
        if final_text:
            return final_text

        return "Al momento non riesco a generare una risposta utile. Riprova tra poco."

    except Exception as e:
        print(f"[OPENAI] Eccezione: {e}")
        return "Al momento non riesco a rispondere correttamente. Riprova tra poco."


# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "MGFishing Chatbot"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "products_loaded": len(PRODUCTS),
        "catalog_last_update": LAST_CATALOG_UPDATE
    })


@app.route("/refresh-products", methods=["POST"])
def refresh_products_endpoint():
    # endpoint opzionale protetto con token semplice
    admin_token = os.getenv("ADMIN_TOKEN", "").strip()
    sent_token = request.headers.get("X-Admin-Token", "").strip()

    if not admin_token or sent_token != admin_token:
        return jsonify({"ok": False, "message": "Non autorizzato"}), 401

    refresh_products(force=True)
    return jsonify({
        "ok": True,
        "message": "Catalogo aggiornato",
        "products_loaded": len(PRODUCTS)
    })


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({
                "reply": "Scrivi pure la tua domanda."
            })

        # 1) risposta rapida per saluti ecc.
        quick_reply = get_quick_reply(user_message)
        if quick_reply:
            return jsonify({"reply": quick_reply})

        # 2) prova ricerca prodotto esatto
        matched_product = find_best_product(user_message)

        # 3) se l'utente vuole un link e il prodotto è stato trovato -> manda subito link diretto
        if is_link_request(user_message) and matched_product:
            return jsonify({
                "reply": f"Ho trovato il prodotto che cerchi:\n{matched_product['title']}\n{matched_product['url']}"
            })

        # 4) se l'utente vuole chiaramente un link prodotto ma non c'è match certo
        if is_link_request(user_message) and not matched_product:
            return jsonify({
                "reply": "Non ho trovato con certezza il prodotto esatto. Scrivimi il nome completo del prodotto e provo a trovarti il link diretto."
            })

        # 5) se il prodotto viene riconosciuto in una normale domanda, dai contesto a OpenAI
        ai_reply = call_openai(user_message, matched_product=matched_product)

        # 6) piccola sicurezza finale: se AI non risponde bene
        if not ai_reply or not ai_reply.strip():
            ai_reply = "Posso aiutarti a trovare prodotti, link specifici o consigliarti l'attrezzatura giusta."

        return jsonify({"reply": ai_reply.strip()})

    except Exception as e:
        print(f"[CHAT] Errore: {e}")
        return jsonify({
            "reply": "Si è verificato un problema temporaneo. Riprova tra poco."
        }), 500


# =========================================================
# AVVIO
# =========================================================

def startup():
    refresh_products(force=False)

    thread = threading.Thread(target=auto_refresh_catalog, daemon=True)
    thread.start()


startup()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
