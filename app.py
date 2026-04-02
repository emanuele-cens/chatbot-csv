import os
import re
import html
import time
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from openai import OpenAI

# =========================
# CONFIG
# =========================
SITEMAP_URL = "https://www.mgfishing.eu/1_index_sitemap.xml"
SITE_DOMAIN = "www.mgfishing.eu"
DEFAULT_MODEL = "gpt-5-mini"

REQUEST_TIMEOUT = 20
MAX_URLS_TO_INDEX = 120
MAX_PAGES_FOR_CONTEXT = 6
MAX_CHARS_PER_PAGE = 5000

USER_AGENT = (
    "Mozilla/5.0 (compatible; MGFishingBot/1.0; +https://www.mgfishing.eu)"
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Assistente MGFishing",
    page_icon="🎣",
    layout="wide",
)

st.title("🎣 Assistente MGFishing")
st.caption("Risposte basate sui contenuti del sito MGFishing")

# =========================
# OPENAI CLIENT
# =========================
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    st.error("Manca la OPENAI_API_KEY. Inseriscila nei secrets di Streamlit oppure nelle variabili ambiente.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_page_text(raw_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.text:
        title = normalize_text(soup.title.text)

    # Prova a prendere contenuto più utile
    candidates = []

    selectors = [
        "main",
        "#content",
        ".page-content",
        ".product-information",
        ".product-description",
        ".cms",
        "article",
        "body",
    ]

    for sel in selectors:
        found = soup.select_one(sel)
        if found:
            candidates.append(found.get_text(" ", strip=True))

    text = max(candidates, key=len) if candidates else soup.get_text(" ", strip=True)
    text = normalize_text(text)

    return title, text


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_xml(url: str) -> bytes:
    session = get_session()
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content


def parse_sitemap_recursive(url: str, collected: list[str] | None = None) -> list[str]:
    if collected is None:
        collected = []

    content = fetch_xml(url)

    try:
        root = ET.fromstring(content)
    except Exception:
        return collected

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # sitemap index
    sitemap_tags = root.findall(".//sm:sitemap/sm:loc", ns)
    if sitemap_tags:
        for loc in sitemap_tags:
            child_url = (loc.text or "").strip()
            if child_url:
                parse_sitemap_recursive(child_url, collected)
        return collected

    # urlset
    url_tags = root.findall(".//sm:url/sm:loc", ns)
    for loc in url_tags:
        page_url = (loc.text or "").strip()
        if page_url and SITE_DOMAIN in page_url:
            collected.append(page_url)

    return collected


def is_useful_url(url: str) -> bool:
    blocked_patterns = [
        "/login",
        "/autenticazione",
        "/order",
        "/carrello",
        "/checkout",
        "/password",
        "/quick-order",
        "/guest-tracking",
        "/module/",
        "controller=authentication",
        "controller=order",
        "controller=cart",
    ]
    low_value_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf", ".xml"]

    lower = url.lower()

    if any(lower.endswith(ext) for ext in low_value_extensions):
        return False

    if any(pat in lower for pat in blocked_patterns):
        return False

    return True


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page(url: str) -> dict:
    session = get_session()
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    title, text = clean_page_text(r.text)

    return {
        "url": url,
        "title": title,
        "text": text[:MAX_CHARS_PER_PAGE],
    }


@st.cache_data(show_spinner=True, ttl=3600)
def build_knowledge_base(sitemap_url: str, max_urls: int) -> list[dict]:
    all_urls = parse_sitemap_recursive(sitemap_url)
    seen = set()
    cleaned_urls = []

    for url in all_urls:
        if url not in seen and is_useful_url(url):
            seen.add(url)
            cleaned_urls.append(url)

    cleaned_urls = cleaned_urls[:max_urls]

    docs = []
    for idx, url in enumerate(cleaned_urls, start=1):
        try:
            doc = fetch_page(url)
            if doc["text"] and len(doc["text"]) > 150:
                docs.append(doc)
        except Exception:
            continue

        # piccolo respiro per non stressare il sito
        time.sleep(0.05)

    return docs


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9àèéìòù]+", " ", text)
    return [t for t in text.split() if len(t) > 1]


def score_document(query: str, doc: dict) -> int:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    haystack_title = (doc.get("title") or "").lower()
    haystack_text = (doc.get("text") or "").lower()
    haystack_url = (doc.get("url") or "").lower()

    score = 0
    for token in query_tokens:
        score += haystack_title.count(token) * 8
        score += haystack_url.count(token) * 6
        score += haystack_text.count(token) * 2

    # bonus per query composta
    joined_query = " ".join(query_tokens)
    if joined_query and joined_query in haystack_text:
        score += 20

    return score


def retrieve_relevant_docs(query: str, docs: list[dict], top_k: int = MAX_PAGES_FOR_CONTEXT) -> list[dict]:
    scored = []
    for doc in docs:
        s = score_document(query, doc)
        if s > 0:
            scored.append((s, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_context_snippets(docs: list[dict]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        block = (
            f"[FONTE {i}]\n"
            f"URL: {doc['url']}\n"
            f"TITOLO: {doc['title']}\n"
            f"CONTENUTO:\n{doc['text']}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def ask_openai(question: str, context: str, model: str) -> str:
    system_prompt = """
Sei l'assistente ecommerce di MGFishing.
Rispondi in italiano.
Usa solo le informazioni presenti nel CONTEXTO del sito.
Se il contesto non contiene abbastanza informazioni, dillo chiaramente.
Non inventare disponibilità, prezzi, misure o caratteristiche.
Se utile, suggerisci di visitare la pagina prodotto citando il link presente nel contesto.
Mantieni risposte chiare, commerciali e utili al cliente finale.
"""

    user_prompt = f"""
CONTEXTO DEL SITO:
{context}

DOMANDA UTENTE:
{question}

ISTRUZIONI:
- Rispondi solo con dati supportati dal contesto.
- Se ci sono più prodotti o link utili, elencali in modo chiaro.
- Se non trovi abbastanza informazioni, scrivi che non hai trovato dati sufficienti nel sito.
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return (response.output_text or "").strip()


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Impostazioni")

    sitemap_url = st.text_input("Sitemap URL", value=SITEMAP_URL)
    model_name = st.text_input("Modello OpenAI", value=DEFAULT_MODEL)
    max_urls = st.slider("Numero massimo pagine da indicizzare", 20, 200, MAX_URLS_TO_INDEX, 10)

    if st.button("🔄 Aggiorna indicizzazione"):
        st.cache_data.clear()
        st.success("Cache svuotata. Alla prossima domanda ricarico tutto.")

    st.markdown("---")
    st.write("Consiglio iniziale:")
    st.write("- prova con 60-120 pagine")
    st.write("- poi aumenta solo se serve")


# =========================
# LOAD KNOWLEDGE BASE
# =========================
with st.spinner("Sto leggendo la sitemap e preparando il contenuto del sito..."):
    try:
        docs = build_knowledge_base(sitemap_url, max_urls)
    except Exception as e:
        st.error(f"Errore durante la lettura della sitemap: {e}")
        st.stop()

st.success(f"Indicizzazione pronta: {len(docs)} pagine lette")

# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ciao, sono l'assistente MGFishing. Chiedimi informazioni sui prodotti presenti sul sito.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# USER INPUT
# =========================
question = st.chat_input("Scrivi qui la tua domanda...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Cerco le pagine più utili..."):
            relevant_docs = retrieve_relevant_docs(question, docs, top_k=MAX_PAGES_FOR_CONTEXT)

            if not relevant_docs:
                answer = "Non ho trovato contenuti sufficienti nel sito per rispondere bene a questa domanda."
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                context = build_context_snippets(relevant_docs)

                try:
                    answer = ask_openai(question, context, model_name)
                except Exception as e:
                    answer = f"Errore nella risposta del modello: {e}"

                st.write(answer)

                with st.expander("Link usati per la risposta"):
                    for doc in relevant_docs:
                        st.write(f"- {doc['url']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
