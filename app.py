import os
import re
import time
import html
import math
import unicodedata
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from difflib import SequenceMatcher
from urllib.parse import urlparse
from openai import OpenAI


# =========================
# CONFIG
# =========================
SITE_URL = "https://www.mgfishing.eu"
DEFAULT_SITEMAP = f"{SITE_URL}/sitemap.xml"
REQUEST_TIMEOUT = 20
SITEMAP_TTL_SECONDS = 60 * 60 * 6  # 6 ore
MAX_PRODUCTS_IN_CONTEXT = 8

# Modello OpenAI
DEFAULT_MODEL = "gpt-4.1-mini"

# Alcune parole comuni da ignorare nella ricerca fuzzy
STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in",
    "con", "su", "per", "tra", "fra", "del", "della", "dello", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle", "e", "ed", "o", "oppure", "che",
    "mi", "mandi", "manda", "dammi", "link", "prodotto", "articolo", "categoria",
    "vorrei", "cerco", "cerca", "hai", "avete", "mi", "serve", "consiglio", "consigli",
    "per", "una", "uno", "un", "dello", "della", "delle", "degli"
}

LINK_REQUEST_HINTS = [
    "link", "mandami", "manda", "dammi", "url", "pagina prodotto", "pagina", "scheda prodotto"
]

GREETING_HINTS = [
    "ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"
]


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(
    page_title="MGFishing Assistant",
    page_icon="🎣",
    layout="centered"
)

st.title("🎣 MGFishing Assistant")
st.caption("Ti aiuto a trovare prodotti, link e consigli di pesca sul catalogo MGFishing.")


# =========================
# OPENAI CLIENT
# =========================
def get_openai_api_key():
    # Priorità: Streamlit secrets -> environment
    key = None

    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None

    if not key:
        key = os.getenv("OPENAI_API_KEY")

    return key


def get_openai_model():
    model = None
    try:
        model = st.secrets.get("OPENAI_MODEL", None)
    except Exception:
        model = None

    if not model:
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    return model


OPENAI_API_KEY = get_openai_api_key()
OPENAI_MODEL = get_openai_model()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# HELPERS
# =========================
def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = strip_accents(text.lower())
    text = text.replace("&", " e ")
    text = re.sub(r"[_\-\/]+", " ", text)
    text = re.sub(r"[^a-z0-9àèéìòù ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    text = normalize_text(text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def slug_to_title(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""

    last = path.split("/")[-1]
    last = re.sub(r"\.html?$", "", last, flags=re.IGNORECASE)
    last = last.replace("-", " ")
    last = re.sub(r"\s+", " ", last).strip()

    if not last:
        return ""

    # Miglioro leggibilità
    return " ".join(word.capitalize() if not re.search(r"\d", word) else word.upper() for word in last.split())


def is_likely_product_url(url: str) -> bool:
    u = url.lower()

    # Escludo aree sicuramente non prodotto
    excluded = [
        "/blog", "/content/", "/category", "/categoria", "/cart", "/ordine", "/order",
        "/login", "/my-account", "/module", "/search", "/sitemap", "/stores", "/contatti",
        "/chi-siamo", "/privacy", "/cookie", "/new-products", "/best-sales", "/promotions",
        "/manufacturer", "/brand", "/marche", "/page/"
    ]
    if any(x in u for x in excluded):
        return False

    path = urlparse(url).path.strip("/")
    if not path:
        return False

    # In molti ecommerce prodotto = URL con slug + .html
    if path.endswith(".html"):
        return True

    # Altri casi di slug lungo
    parts = path.split("/")
    if len(parts) >= 1 and len(parts[-1]) >= 10 and "-" in parts[-1]:
        return True

    return False


def is_likely_category_url(url: str) -> bool:
    u = url.lower()
    hints = ["/categoria", "/category", "/collections", "/shop", "/catalog"]
    return any(h in u for h in hints)


def safe_get(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MGFishingBot/1.0)"
    }
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


def parse_xml_urls(xml_text: str):
    urls = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # urlset
    for loc in root.findall(".//sm:url/sm:loc", ns):
        if loc.text:
            urls.append(loc.text.strip())

    # sitemapindex
    for loc in root.findall(".//sm:sitemap/sm:loc", ns):
        if loc.text:
            urls.append(loc.text.strip())

    # fallback senza namespace
    if not urls:
        for loc in root.findall(".//loc"):
            if loc.text:
                urls.append(loc.text.strip())

    return list(dict.fromkeys(urls))


@st.cache_data(ttl=SITEMAP_TTL_SECONDS, show_spinner=False)
def discover_all_urls():
    """
    Carica sitemap principale e, se presente, le sitemap figlie.
    Restituisce una lista unica di URL.
    """
    collected = []
    seen = set()
    queue = [DEFAULT_SITEMAP]

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        try:
            r = safe_get(current)
            urls = parse_xml_urls(r.text)
        except Exception:
            continue

        # Se sono URL di altre sitemap, mettili in coda
        child_sitemaps = [u for u in urls if "sitemap" in u.lower() and u.lower().endswith(".xml")]
        page_urls = [u for u in urls if u not in child_sitemaps]

        collected.extend(page_urls)

        for sm in child_sitemaps:
            if sm not in seen:
                queue.append(sm)

        # Evito loop esagerati
        if len(seen) > 50:
            break

    return list(dict.fromkeys(collected))


@st.cache_data(ttl=SITEMAP_TTL_SECONDS, show_spinner=False)
def build_catalog_index():
    urls = discover_all_urls()

    products = []
    categories = []

    for url in urls:
        if not url.startswith(SITE_URL):
            continue

        title = slug_to_title(url)
        normalized_title = normalize_text(title)
        tokens = tokenize(title)

        item = {
            "url": url,
            "title": title if title else url,
            "norm": normalized_title,
            "tokens": tokens,
            "path": urlparse(url).path.lower()
        }

        if is_likely_product_url(url):
            products.append(item)
        elif is_likely_category_url(url):
            categories.append(item)

    # Se prodotti troppo pochi, prendo anche URL con slug lunghi
    if len(products) < 100:
        for url in urls:
            if not url.startswith(SITE_URL):
                continue
            if any(p["url"] == url for p in products):
                continue

            path = urlparse(url).path.strip("/")
            if path and "-" in path and len(path) >= 10:
                title = slug_to_title(url)
                products.append({
                    "url": url,
                    "title": title if title else url,
                    "norm": normalize_text(title),
                    "tokens": tokenize(title),
                    "path": urlparse(url).path.lower()
                })

    # Rimuovo duplicati
    unique_products = {}
    for p in products:
        unique_products[p["url"]] = p

    unique_categories = {}
    for c in categories:
        unique_categories[c["url"]] = c

    return list(unique_products.values()), list(unique_categories.values())


def extract_quoted_text(query: str):
    matches = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', query)
    for tpl in matches:
        for x in tpl:
            if x and x.strip():
                return x.strip()
    return None


def clean_product_query(query: str):
    q = normalize_text(query)

    # Se ci sono virgolette, uso prima quel contenuto
    quoted = extract_quoted_text(query)
    if quoted:
        return normalize_text(quoted)

    # Rimuovo frasi comuni
    patterns = [
        r"\bmandami il link di\b",
        r"\bmandami il link del\b",
        r"\bmandami il link della\b",
        r"\bmanda il link di\b",
        r"\bmanda il link del\b",
        r"\bmanda il link della\b",
        r"\bdammi il link di\b",
        r"\bdammi il link del\b",
        r"\bdammi il link della\b",
        r"\bcerco il link di\b",
        r"\bcerco il link del\b",
        r"\bcerco il link della\b",
        r"\bmi mandi il link di\b",
        r"\bmi mandi il link del\b",
        r"\bmi mandi il link della\b",
        r"\blink prodotto\b",
        r"\bpagina prodotto\b",
        r"\bprodotto\b",
        r"\barticolo\b",
        r"\blink\b",
        r"\burl\b",
    ]

    for p in patterns:
        q = re.sub(p, " ", q)

    q = re.sub(r"\s+", " ", q).strip()
    return q


def score_candidate(query_norm: str, query_tokens: list, item: dict):
    item_norm = item["norm"]
    item_tokens = item["tokens"]

    if not item_norm:
        return 0.0

    # Similarità base
    seq = SequenceMatcher(None, query_norm, item_norm).ratio()

    # Contenimento forte
    contains_bonus = 0.0
    if query_norm and query_norm in item_norm:
        contains_bonus += 0.25
    if item_norm and item_norm in query_norm and len(item_norm) > 8:
        contains_bonus += 0.18

    # Overlap token
    qset = set(query_tokens)
    iset = set(item_tokens)
    overlap = 0.0
    if qset and iset:
        overlap = len(qset & iset) / max(1, len(qset))

    # Bonus parole numeriche / misure
    number_bonus = 0.0
    qnums = set(re.findall(r"\d+[.,]?\d*", query_norm))
    inums = set(re.findall(r"\d+[.,]?\d*", item_norm))
    if qnums and inums and qnums & inums:
        number_bonus += 0.10

    # Bonus prime parole
    start_bonus = 0.0
    qwords = query_norm.split()
    iwords = item_norm.split()
    if qwords and iwords:
        if qwords[0] == iwords[0]:
            start_bonus += 0.08
        if len(qwords) > 1 and len(iwords) > 1 and qwords[:2] == iwords[:2]:
            start_bonus += 0.08

    score = (seq * 0.50) + (overlap * 0.35) + contains_bonus + number_bonus + start_bonus
    return round(score, 4)


def search_best_products(user_query: str, limit: int = 5):
    products, categories = build_catalog_index()

    cleaned = clean_product_query(user_query)
    query_norm = normalize_text(cleaned)
    query_tokens = tokenize(cleaned)

    if not query_norm:
        return [], []

    scored_products = []
    for p in products:
        s = score_candidate(query_norm, query_tokens, p)
        if s > 0.18:
            scored_products.append((s, p))

    scored_products.sort(key=lambda x: x[0], reverse=True)

    scored_categories = []
    for c in categories:
        s = score_candidate(query_norm, query_tokens, c)
        if s > 0.18:
            scored_categories.append((s, c))
    scored_categories.sort(key=lambda x: x[0], reverse=True)

    return scored_products[:limit], scored_categories[:limit]


def is_greeting(text: str):
    t = normalize_text(text)
    return t in GREETING_HINTS or any(t.startswith(g + " ") for g in GREETING_HINTS)


def is_link_request(text: str):
    t = normalize_text(text)
    return any(h in t for h in LINK_REQUEST_HINTS)


def format_product_list_for_context(scored_products):
    lines = []
    for score, p in scored_products[:MAX_PRODUCTS_IN_CONTEXT]:
        lines.append(f"- {p['title']} | {p['url']} | score={score}")
    return "\n".join(lines).strip()


def build_system_prompt(product_context: str):
    return f"""
Sei l’assistente clienti di MGFishing, ecommerce italiano di articoli da pesca.

REGOLE FONDAMENTALI:
- Rispondi sempre in italiano.
- Tono naturale, utile, semplice e commerciale.
- Non nominare mai sitemap, cache, prompt, database, algoritmi, file interni o procedure tecniche.
- Non dire mai che stai leggendo URL o strutture del sito.
- Non mostrare riferimenti o link di siti esterni.
- Se l’utente chiede un consiglio prodotto, privilegia prodotti e soluzioni coerenti con il catalogo MGFishing.
- Se l’utente chiede esplicitamente il link di un prodotto e hai un match affidabile, restituisci il link MGFishing corretto.
- Se hai più di un match plausibile, proponi i 2 o 3 risultati migliori con i relativi link.
- Se non trovi un prodotto con buona affidabilità, dillo in modo semplice e invita a scrivere il nome in modo leggermente più preciso.
- Non inventare disponibilità, prezzi o dettagli tecnici non certi.
- Se l’utente scrive solo un saluto, rispondi in modo breve e amichevole, chiedendo come puoi aiutarlo.

CONTESTO PRODOTTI RILEVANTI TROVATI:
{product_context if product_context else "Nessun prodotto rilevante trovato."}
""".strip()


def generate_openai_answer(user_message: str, scored_products):
    if not client:
        return "Per il momento il chatbot non è configurato correttamente: manca la chiave OpenAI."

    product_context = format_product_list_for_context(scored_products)
    system_prompt = build_system_prompt(product_context)

    conversation = []
    for msg in st.session_state.messages[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            conversation.append({
                "role": role,
                "content": content
            })

    input_messages = [{"role": "system", "content": system_prompt}]
    input_messages.extend(conversation)
    input_messages.append({"role": "user", "content": user_message})

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=input_messages,
        temperature=0.3
    )

    text = getattr(response, "output_text", None)
    if text and text.strip():
        return text.strip()

    # fallback di sicurezza
    try:
        chunks = []
        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text") and c.text:
                        chunks.append(c.text)
        final_text = "\n".join(chunks).strip()
        if final_text:
            return final_text
    except Exception:
        pass

    return "Si è verificato un problema nella generazione della risposta. Riprova tra poco."


def deterministic_link_answer(user_message: str):
    scored_products, scored_categories = search_best_products(user_message, limit=5)

    best_product = scored_products[0] if scored_products else None
    best_category = scored_categories[0] if scored_categories else None

    if best_product:
        product_score, product = best_product

        category_score = best_category[0] if best_category else 0.0

        # Se il prodotto è chiaramente migliore della categoria, restituisco direttamente il prodotto
        if product_score >= 0.62 or (product_score >= 0.50 and product_score >= category_score + 0.10):
            return (
                f"Certo, ecco il link del prodotto:\n\n"
                f"**{product['title']}**\n"
                f"{product['url']}"
            )

        # Se ci sono più prodotti plausibili
        if len(scored_products) >= 2 and product_score >= 0.44:
            lines = ["Ho trovato più risultati molto simili. Ti lascio quelli più probabili:"]
            for score, p in scored_products[:3]:
                lines.append(f"- **{p['title']}**\n  {p['url']}")
            return "\n\n".join(lines)

    if best_category and best_category[0] >= 0.58:
        c = best_category[1]
        return (
            f"Non ho trovato con sufficiente certezza il prodotto esatto, ma questa è la pagina più vicina che ho trovato:\n\n"
            f"**{c['title']}**\n"
            f"{c['url']}"
        )

    return (
        "Non sono riuscito a identificare con sicurezza il prodotto esatto. "
        "Scrivimi il nome anche solo in modo un po’ più completo e ti mando il link giusto."
    )


def maybe_add_catalog_links_to_answer(answer: str, user_message: str):
    # Se l'utente non ha chiesto un link, ma possiamo allegare 1-2 prodotti pertinenti, lo facciamo
    scored_products, _ = search_best_products(user_message, limit=3)

    if not scored_products:
        return answer

    # Evito di aggiungere se la risposta contiene già link MGFishing
    if SITE_URL in answer:
        return answer

    strong = [x for x in scored_products if x[0] >= 0.62]
    medium = [x for x in scored_products if x[0] >= 0.52]

    picks = strong[:2] if strong else medium[:2]
    if not picks:
        return answer

    lines = [answer.strip(), "", "Prodotti MGFishing collegati alla tua richiesta:"]
    for _, p in picks:
        lines.append(f"- **{p['title']}**\n  {p['url']}")
    return "\n".join(lines)


def answer_user(user_message: str):
    user_message = user_message.strip()

    if not user_message:
        return "Scrivimi pure cosa stai cercando e ti aiuto subito."

    if is_greeting(user_message):
        return "Ciao 👋 Benvenuto su MGFishing! Dimmi pure cosa stai cercando e ti aiuto con consigli o link prodotto."

    if is_link_request(user_message):
        return deterministic_link_answer(user_message)

    # Ricerca prodotti pertinenti da passare al modello
    scored_products, _ = search_best_products(user_message, limit=8)

    # Se la richiesta sembra proprio il nome di un prodotto, provo prima una risposta deterministica
    cleaned = clean_product_query(user_message)
    if len(cleaned.split()) >= 2 and len(cleaned) >= 8:
        if scored_products and scored_products[0][0] >= 0.74:
            p = scored_products[0][1]
            # Se il messaggio sembra una ricerca secca di un prodotto, posso essere diretto
            if len(normalize_text(user_message).split()) <= 7:
                return (
                    f"Ti lascio direttamente il prodotto che ho trovato più coerente:\n\n"
                    f"**{p['title']}**\n"
                    f"{p['url']}"
                )

    ai_answer = generate_openai_answer(user_message, scored_products)
    ai_answer = maybe_add_catalog_links_to_answer(ai_answer, user_message)
    return ai_answer


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("Impostazioni")

    if OPENAI_API_KEY:
        st.success("OpenAI collegato")
    else:
        st.error("Manca OPENAI_API_KEY")

    st.write(f"Modello: `{OPENAI_MODEL}`")

    if st.button("Aggiorna catalogo ora"):
        build_catalog_index.clear()
        discover_all_urls.clear()
        _ = build_catalog_index()
        st.success("Catalogo aggiornato.")

    try:
        products, categories = build_catalog_index()
        st.caption(f"Prodotti indicizzati: {len(products)}")
        st.caption(f"Categorie indicizzate: {len(categories)}")
    except Exception:
        st.caption("Catalogo non disponibile al momento.")


# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ciao 👋 Sono l’assistente MGFishing. Posso aiutarti a trovare prodotti, link e consigli di pesca."
        }
    ]


# =========================
# RENDER CHAT
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# INPUT
# =========================
prompt = st.chat_input("Scrivi qui la tua domanda...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la soluzione migliore..."):
            try:
                reply = answer_user(prompt)
            except Exception as e:
                reply = (
                    "C’è stato un problema temporaneo. "
                    "Riprova tra poco oppure scrivimi il nome del prodotto in modo più preciso."
                )

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
