import os
import re
import json
import time
import html
import math
import queue
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlparse
from difflib import SequenceMatcher

from bs4 import BeautifulSoup
from openai import OpenAI
from duckduckgo_search import DDGS

# =========================================================
# CONFIG
# =========================================================

SITE_URL = "https://www.mgfishing.eu"
SITEMAP_URL = "https://www.mgfishing.eu/sitemap.xml"
WHATSAPP_NUMBER = "3494166335"
WHATSAPP_LABEL = "WhatsApp 3494166335 - Emanuele"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Modello consigliato: puoi cambiarlo se vuoi
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Quanti prodotti massimo caricare dal sito
MAX_PRODUCTS_TO_INDEX = 4000

# Quanti risultati internet usare come contesto
MAX_WEB_RESULTS = 5

# Timeout richieste
HTTP_TIMEOUT = 20

# =========================================================
# REGOLE FISSE DEL NEGOZIO
# QUI PUOI TENERE LE STESSE REGOLE CHE AVEVAMO GIÀ MESSO
# =========================================================

SHOP_RULES = """
Regole del negozio MGFishing:

- Per informazioni su spedizioni, ordini, assistenza e supporto clienti, rispondi in modo chiaro e sintetico.
- Se il cliente chiede di parlare con un operatore, contattare il negozio o se non sai rispondere con sicurezza, invita sempre a scrivere su WhatsApp al numero 3494166335 chiedendo di Emanuele.
- Non inventare mai disponibilità, prezzi, tempi o condizioni se non risultano dal contesto disponibile.
- Se non trovi un prodotto in modo preciso, prova a cercare prodotti molto simili del catalogo e proponi alternative.
- Non citare mai altri siti web nella risposta al cliente.
- Quando trovi il link corretto di un prodotto del sito MGFishing, inseriscilo chiaramente in risposta.
- Se il cliente chiede tracking e non hai dati certi, indirizzalo a WhatsApp 3494166335 - Emanuele.
- Se il cliente chiede tempi di spedizione, rispondi in linea generale in modo prudente e professionale.
"""

# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="MGFishing Assistant",
    page_icon="🎣",
    layout="wide"
)

st.title("🎣 MGFishing Assistant")
st.caption("Consigli sui prodotti MGFishing, tecniche di pesca, meteo e assistenza.")

# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY non impostata.")
    return OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# TEXT HELPERS
# =========================================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\sàèéìòù]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def contains_operator_request(user_text: str) -> bool:
    t = normalize_text(user_text)
    triggers = [
        "operatore", "parlare con operatore", "assistenza", "umano",
        "whatsapp", "contatto", "contattare", "numero", "telefono"
    ]
    return any(x in t for x in triggers)

def looks_like_product_question(user_text: str) -> bool:
    t = normalize_text(user_text)

    product_keywords = [
        "prodotto", "mulinello", "canna", "trecciato", "monofilo", "esca",
        "artificiale", "bombarda", "galleggiante", "amo", "girella", "fluorocarbon",
        "trabucco", "shimano", "daiwa", "tubertini", "colmic", "yuki",
        "rapala", "major craft", "molix", "fassa", "ryobi", "nomura",
        "xenos", "caldia", "crossfire", "ballistic", "atmos", "tk4"
    ]

    generic_non_product = [
        "meteo", "vento", "onde", "montatura", "terminali", "come pescare",
        "tecnica", "maree", "pressione", "luna", "licenza", "bollettino"
    ]

    if any(k in t for k in product_keywords):
        return True

    if any(k in t for k in generic_non_product):
        return False

    # Se sembra richiesta di link prodotto
    if "link" in t or "url" in t:
        return True

    return False

def fallback_operator_message() -> str:
    return (
        f"Per assistenza diretta ti consiglio di contattare {WHATSAPP_LABEL}."
    )

# =========================================================
# SITEMAP + PRODUCT INDEX
# =========================================================

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_xml(url: str) -> str:
    r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def parse_sitemap_urls(xml_text: str) -> List[str]:
    urls = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # sitemap index
        sitemap_locs = root.findall(".//sm:sitemap/sm:loc", ns)
        if sitemap_locs:
            return [loc.text.strip() for loc in sitemap_locs if loc.text]

        # urlset
        url_locs = root.findall(".//sm:url/sm:loc", ns)
        if url_locs:
            return [loc.text.strip() for loc in url_locs if loc.text]
    except Exception:
        pass
    return urls

def is_product_url(url: str) -> bool:
    u = url.lower()
    # Adatta ai pattern più comuni di PrestaShop / shop
    product_signals = [
        "/",
    ]
    bad_signals = [
        "/category", "/categoria", "/blog", "/content", "/login",
        "/carrello", "/cart", "/ordine", "/checkout", "/module"
    ]

    if any(x in u for x in bad_signals):
        return False

    # Molti PrestaShop usano URL con id-nome-prodotto oppure slug prodotto
    # Qui filtriamo in modo ampio
    path = urlparse(url).path.strip("/")
    if not path:
        return False
    if path.count("/") > 2:
        return False

    return True

def extract_candidate_product_sitemaps(sitemap_urls: List[str]) -> List[str]:
    preferred = []
    for url in sitemap_urls:
        lu = url.lower()
        if "product" in lu or "prodot" in lu:
            preferred.append(url)
    if preferred:
        return preferred
    return sitemap_urls

def safe_get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def extract_product_data_from_html(url: str, html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")

    title = ""
    meta_desc = ""
    body_text = ""

    if soup.title and soup.title.text:
        title = clean_text(soup.title.text)

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = clean_text(meta.get("content"))

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = clean_text(h1.get_text(" ", strip=True)) or title

    body_candidates = []

    selectors = [
        "article",
        "#content",
        ".product-information",
        ".product-description",
        ".product-details",
        ".page-content",
        "main"
    ]

    for sel in selectors:
        for node in soup.select(sel):
            txt = clean_text(node.get_text(" ", strip=True))
            if len(txt) > 80:
                body_candidates.append(txt)

    if body_candidates:
        body_text = max(body_candidates, key=len)
    else:
        body_text = clean_text(soup.get_text(" ", strip=True))

    # JSON-LD
    product_name = ""
    try:
        scripts = soup.find_all("script", type="application/ld+json")
        for s in scripts:
            content = s.string or s.get_text()
            if not content:
                continue
            data = json.loads(content)
            data_list = data if isinstance(data, list) else [data]
            for item in data_list:
                if isinstance(item, dict) and item.get("@type") == "Product":
                    product_name = clean_text(item.get("name", ""))
                    if product_name:
                        title = product_name
    except Exception:
        pass

    text_for_search = clean_text(f"{title}. {meta_desc}. {body_text}")

    return {
        "title": title or urlparse(url).path.strip("/").replace("-", " "),
        "url": url,
        "description": meta_desc,
        "content": text_for_search[:4000]
    }

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)
def build_product_index() -> List[Dict[str, Any]]:
    all_products: List[Dict[str, Any]] = []

    sitemap_xml = download_xml(SITEMAP_URL)
    sitemap_urls = parse_sitemap_urls(sitemap_xml)

    if not sitemap_urls:
        return []

    candidate_sitemaps = extract_candidate_product_sitemaps(sitemap_urls)

    collected_urls: List[str] = []

    for sm_url in candidate_sitemaps:
        sm_xml = safe_get(sm_url)
        if not sm_xml:
            continue

        nested_urls = parse_sitemap_urls(sm_xml)
        if not nested_urls:
            continue

        for u in nested_urls:
            if is_product_url(u):
                collected_urls.append(u)

    # Rimuovi duplicati
    seen = set()
    final_urls = []
    for u in collected_urls:
        if u not in seen:
            seen.add(u)
            final_urls.append(u)

    final_urls = final_urls[:MAX_PRODUCTS_TO_INDEX]

    for url in final_urls:
        html_text = safe_get(url)
        if not html_text:
            continue
        try:
            product = extract_product_data_from_html(url, html_text)
            if product.get("title"):
                all_products.append(product)
        except Exception:
            continue

    return all_products

def score_product(product: Dict[str, Any], query: str) -> float:
    query_n = normalize_text(query)
    title_n = normalize_text(product.get("title", ""))
    content_n = normalize_text(product.get("content", ""))

    score = 0.0

    # Match titolo forte
    title_sim = similarity(query, product.get("title", ""))
    score += title_sim * 0.55

    # Bonus se tutte/parziali parole del query nel titolo
    words = [w for w in query_n.split() if len(w) > 2]
    if words:
        title_hits = sum(1 for w in words if w in title_n)
        content_hits = sum(1 for w in words if w in content_n)

        score += min(title_hits / max(len(words), 1), 1.0) * 0.30
        score += min(content_hits / max(len(words), 1), 1.0) * 0.15

    # Bonus se query intera nel titolo
    if query_n and query_n in title_n:
        score += 0.25

    return score

def find_best_products(query: str, products: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    scored = []
    for p in products:
        s = score_product(p, query)
        if s > 0.15:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]

def extract_product_name_candidate(user_text: str) -> str:
    # Prova a ripulire frasi tipo "mi mandi il link del trabucco xenos sw"
    t = clean_text(user_text)
    patterns = [
        r"link del(?:la|lo)?\s+(.+)",
        r"link prodotto\s+(.+)",
        r"mi mandi il link di\s+(.+)",
        r"cerco\s+(.+)",
        r"hai\s+(.+)",
        r"informazioni su\s+(.+)",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return clean_text(m.group(1))
    return t

# =========================================================
# WEB SEARCH (NO REFERENCES IN FINAL ANSWER)
# =========================================================

@st.cache_data(ttl=60 * 30, show_spinner=False)
def web_search_context(query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict[str, str]]:
    results = []
    try:
        with DDGS() as ddgs:
            items = list(ddgs.text(query, region="it-it", safesearch="moderate", max_results=max_results))
            for item in items:
                results.append({
                    "title": clean_text(item.get("title", "")),
                    "body": clean_text(item.get("body", "")),
                    "href": item.get("href", "")
                })
    except Exception:
        return []
    return results

# =========================================================
# INTENT LOGIC
# =========================================================

def build_product_context(user_text: str, products: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    candidate = extract_product_name_candidate(user_text)
    best = find_best_products(candidate, products, top_k=5)

    if not best:
        best = find_best_products(user_text, products, top_k=5)

    if not best:
        return "", []

    lines = []
    for i, p in enumerate(best, start=1):
        lines.append(
            f"[PRODOTTO {i}]\n"
            f"Titolo: {p.get('title', '')}\n"
            f"URL: {p.get('url', '')}\n"
            f"Descrizione: {p.get('description', '')}\n"
            f"Contenuto: {p.get('content', '')[:1200]}\n"
        )

    return "\n\n".join(lines), best

def build_web_context(user_text: str) -> str:
    results = web_search_context(user_text, MAX_WEB_RESULTS)
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[RISULTATO WEB {i}]\n"
            f"Titolo: {r['title']}\n"
            f"Testo: {r['body']}\n"
        )
    return "\n\n".join(lines)

# =========================================================
# OPENAI RESPONSE
# =========================================================

def generate_answer(
    user_text: str,
    products: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]]
) -> str:
    if contains_operator_request(user_text):
        return fallback_operator_message()

    is_product = looks_like_product_question(user_text)

    product_context, matched_products = build_product_context(user_text, products)
    web_context = ""

    if is_product:
        # Se è domanda prodotto, lavora prima col sito
        # Solo se il contesto è troppo scarso, il modello può comunque rispondere prudentemente
        pass
    else:
        web_context = build_web_context(user_text)

    system_prompt = f"""
Sei l'assistente clienti di MGFishing.

OBIETTIVO:
- Aiutare il cliente a scegliere prodotti del catalogo MGFishing.
- Rispondere a domande generali di pesca, montature, tecniche, meteo e consigli usando il contesto web se disponibile.
- NON devi mai citare o nominare altri siti nella risposta finale.
- Se viene chiesto un prodotto e trovi un link prodotto corretto del sito MGFishing, devi inserirlo chiaramente.
- Se non trovi il prodotto esatto ma trovi prodotti molto simili, proponi alternative del catalogo MGFishing.
- Se non sai rispondere con sufficiente sicurezza, oppure il cliente chiede un operatore, devi rispondere invitando a contattare {WHATSAPP_LABEL}.
- Non inventare mai dati certi che non hai.
- Se la domanda riguarda prodotti, dai priorità assoluta al catalogo MGFishing.
- Se la domanda riguarda consigli generali, montature, meteo, tecniche o informazioni di pesca, puoi usare il contesto web ma non devi citare fonti esterne.
- Se parli di spedizioni, tracking o assistenza, segui le regole del negozio qui sotto.

REGOLE NEGOZIO:
{SHOP_RULES}

STILE RISPOSTA:
- Scrivi in italiano.
- Tono professionale, semplice, utile e commerciale.
- Risposta concreta, non troppo lunga.
- Se consigli un prodotto, spiega brevemente perché.
- Se c'è un link prodotto MGFishing, scrivilo su una riga separata con prefisso: "Link prodotto:".
"""

    user_prompt = f"""
CHAT PRECEDENTE:
{json.dumps(chat_history[-8:], ensure_ascii=False)}

DOMANDA CLIENTE:
{user_text}

TIPO DOMANDA:
{"PRODOTTO / CATALOGO" if is_product else "CONSIGLIO GENERALE / WEB"}

CONTESTO PRODOTTI MGFISHING:
{product_context if product_context else "Nessun prodotto trovato con sufficiente confidenza."}

CONTESTO WEB:
{web_context if web_context else "Nessun contesto web disponibile o non necessario."}

ISTRUZIONI IMPORTANTI:
- Se la domanda è su un prodotto, usa soprattutto il contesto prodotti.
- Se hai trovato uno o più prodotti adatti, dai il consiglio usando quelli.
- Se il cliente chiede un link di un prodotto specifico e lo trovi, fornisci il link più pertinente.
- Se hai trovato solo categorie o risultati vaghi, NON fingere che sia il prodotto esatto.
- Se non sei sicuro, invita a contattare {WHATSAPP_LABEL}.
- Non menzionare mai altri siti web.
"""

    client = get_openai_client()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3
    )

    answer = ""
    try:
        answer = response.output_text.strip()
    except Exception:
        answer = str(response)

    answer = clean_text(answer)

    if not answer:
        return fallback_operator_message()

    # Sicurezza finale: se il modello parla di fonti o siti esterni, pulisci un po'
    banned_phrases = [
        "secondo il sito",
        "come riportato da",
        "fonte",
        "fonti",
        "ho trovato su",
        "sito web"
    ]
    lower_answer = answer.lower()
    if any(bp in lower_answer for bp in banned_phrases):
        # risposta prudente
        return fallback_operator_message()

    return answer

# =========================================================
# UI CHAT
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ciao! Sono l’assistente MGFishing. Posso aiutarti a scegliere prodotti, darti consigli di pesca o indirizzarti all’assistenza."
        }
    ]

with st.sidebar:
    st.subheader("Impostazioni")
    auto_load = st.checkbox("Carica catalogo prodotti del sito", value=True)
    st.markdown(f"**Sito:** {SITE_URL}")
    st.markdown(f"**Assistenza:** {WHATSAPP_LABEL}")

products_data: List[Dict[str, Any]] = []

if auto_load:
    try:
        with st.spinner("Sto caricando il catalogo prodotti MGFishing..."):
            products_data = build_product_index()
    except Exception as e:
        st.warning("Non sono riuscito a caricare il catalogo prodotti in questo momento.")
        products_data = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Scrivi qui la tua domanda...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Sto scrivendo la risposta..."):
            try:
                answer = generate_answer(
                    user_text=user_input,
                    products=products_data,
                    chat_history=st.session_state.messages
                )
            except Exception:
                answer = fallback_operator_message()

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
