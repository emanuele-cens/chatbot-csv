import os
import re
import time
import html
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Assistente MGFishing", page_icon="🎣", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# =========================
# OPENAI CLIENT
# =========================
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []

if "indexed_urls" not in st.session_state:
    st.session_state.indexed_urls = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "last_sitemap_url" not in st.session_state:
    st.session_state.last_sitemap_url = ""

if "last_max_pages" not in st.session_state:
    st.session_state.last_max_pages = 0

# =========================
# HELPERS
# =========================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MGFishingBot/1.0)"
}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def is_greeting(text: str) -> bool:
    t = normalize_text(text)
    greetings = {
        "ciao",
        "ciao!",
        "salve",
        "salve!",
        "buongiorno",
        "buongiorno!",
        "buonasera",
        "buonasera!",
        "hey",
        "hey!",
        "ehi",
        "ehi!",
        "ciao assistente",
        "ciao bot",
        "ciao chatbot"
    }
    return t in greetings

def greeting_response() -> str:
    return (
        "Ciao 👋 Sono l'assistente MGFishing.\n\n"
        "Posso aiutarti a trovare prodotti, caratteristiche, utilizzi e dettagli presenti sul sito.\n\n"
        "Ad esempio puoi scrivermi:\n"
        "- mulinello per surfcasting\n"
        "- canna da trota lago\n"
        "- trecciato per spinning\n"
        "- oppure il nome preciso di un prodotto"
    )

def is_smalltalk(text: str) -> bool:
    t = normalize_text(text)
    smalltalk_items = {
        "come stai",
        "come va",
        "chi sei",
        "cosa fai",
        "mi aiuti",
        "puoi aiutarmi",
        "sei online",
        "ok",
        "perfetto",
        "grazie",
        "grazie!",
    }
    return t in smalltalk_items

def smalltalk_response(text: str) -> str:
    t = normalize_text(text)

    if t in {"come stai", "come va"}:
        return "Sto bene, grazie 😊 Sono qui per aiutarti a trovare prodotti e informazioni presenti sul sito MGFishing."
    if t == "chi sei":
        return "Sono l'assistente MGFishing. Posso aiutarti a cercare prodotti, caratteristiche e dettagli presenti sul sito."
    if t in {"cosa fai", "mi aiuti", "puoi aiutarmi"}:
        return (
            "Sì, certo 😊 Posso aiutarti a trovare prodotti presenti sul sito MGFishing, "
            "confrontare caratteristiche e indicarti dettagli utili in base alla tua richiesta."
        )
    if t == "sei online":
        return "Sì, sono qui 👍 Scrivimi pure cosa stai cercando."
    if t in {"ok", "perfetto"}:
        return "Perfetto 👍 Scrivimi pure il nome del prodotto oppure la tipologia che stai cercando."
    if t in {"grazie", "grazie!"}:
        return "Di nulla 😊 Se vuoi, scrivimi pure il nome del prodotto o la tecnica di pesca che ti interessa."

    return "Certo 😊 Scrivimi pure cosa stai cercando sul sito MGFishing."

def clean_page_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "img", "form", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def fetch_url(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def parse_sitemap(sitemap_url: str) -> list:
    xml_text = fetch_url(sitemap_url, timeout=20)
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []

    # Caso sitemap index
    sitemap_nodes = root.findall(".//sm:sitemap/sm:loc", ns)
    if sitemap_nodes:
        for node in sitemap_nodes:
            sub_url = node.text.strip()
            urls.extend(parse_sitemap(sub_url))
        return list(dict.fromkeys(urls))

    # Caso urlset
    url_nodes = root.findall(".//sm:url/sm:loc", ns)
    for node in url_nodes:
        if node.text:
            urls.append(node.text.strip())

    return list(dict.fromkeys(urls))

def filter_useful_urls(urls: list) -> list:
    blocked_parts = [
        "/login",
        "/autenticazione",
        "/password",
        "/ordine",
        "/carrello",
        "/checkout",
        "/contattaci",
        "/guest-tracking",
        "/module/",
        "?",
        "#",
    ]

    cleaned = []
    for url in urls:
        low = url.lower()
        if any(part in low for part in blocked_parts):
            continue
        cleaned.append(url)

    return list(dict.fromkeys(cleaned))

def build_knowledge_base(sitemap_url: str, max_pages: int = 20, progress_bar=None, status_box=None):
    urls = parse_sitemap(sitemap_url)
    urls = filter_useful_urls(urls)
    urls = urls[:max_pages]

    kb = []

    for i, url in enumerate(urls, start=1):
        if status_box:
            status_box.info(f"Lettura pagina {i}/{len(urls)}: {url}")

        html_content = fetch_url(url, timeout=20)
        if not html_content:
            continue

        text = clean_page_text(html_content)
        if len(text) < 200:
            continue

        title = ""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = ""

        kb.append({
            "url": url,
            "title": title,
            "content": text[:12000]
        })

        if progress_bar:
            progress_bar.progress(min(i / max(len(urls), 1), 1.0))

        time.sleep(0.2)

    return kb, urls

def score_page(query: str, page_text: str, page_title: str = "") -> int:
    q = normalize_text(query)
    content = normalize_text(page_text)
    title = normalize_text(page_title)

    words = [w for w in re.findall(r"\w+", q) if len(w) > 2]
    if not words:
        return 0

    score = 0
    for w in words:
        score += content.count(w)
        score += title.count(w) * 3

    if q in content:
        score += 15
    if q in title:
        score += 20

    return score

def search_relevant_pages(query: str, kb: list, top_k: int = 4) -> list:
    scored = []
    for item in kb:
        score = score_page(query, item.get("content", ""), item.get("title", ""))
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]

def build_context_from_pages(pages: list) -> str:
    chunks = []
    for i, page in enumerate(pages, start=1):
        chunk = (
            f"Fonte {i}\n"
            f"Titolo: {page.get('title', '')}\n"
            f"URL: {page.get('url', '')}\n"
            f"Contenuto: {page.get('content', '')[:4000]}\n"
        )
        chunks.append(chunk)
    return "\n\n".join(chunks)

def fallback_no_results() -> str:
    return (
        "Non ho trovato abbastanza informazioni nel sito per rispondere con precisione.\n\n"
        "Prova a scrivere:\n"
        "- il nome preciso del prodotto\n"
        "- una categoria\n"
        "- la tecnica di pesca\n\n"
        "Ad esempio: mulinello surfcasting, canna trota lago, filo trecciato."
    )

def ask_openai(question: str, context: str, model_name: str) -> str:
    if client is None:
        return "Manca la API key di OpenAI nelle secrets di Streamlit."

    system_prompt = (
        "Sei l'assistente del sito MGFishing.\n"
        "Rispondi SOLO usando le informazioni contenute nel contesto fornito.\n"
        "Se il contesto non è sufficiente, dillo chiaramente senza inventare.\n"
        "Rispondi in italiano in modo chiaro, utile e commerciale ma naturale.\n"
        "Se possibile suggerisci all'utente di indicare il nome preciso del prodotto o della categoria.\n"
    )

    user_prompt = (
        f"DOMANDA UTENTE:\n{question}\n\n"
        f"CONTESTO DEL SITO:\n{context}\n\n"
        "Rispondi solo in base al contesto."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Errore OpenAI: {e}"

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Impostazioni")

    sitemap_url = st.text_input(
        "Sitemap URL",
        value=st.session_state.last_sitemap_url or "https://www.mgfishing.eu/1_index_sitemap.xml"
    )

    model_name = st.selectbox(
        "Modello OpenAI",
        options=["gpt-5-mini", "gpt-4o-mini"],
        index=0
    )

    max_pages = st.slider(
        "Numero massimo pagine da indicizzare",
        min_value=5,
        max_value=150,
        value=20,
        step=1
    )

    update_clicked = st.button("🔄 Aggiorna indicizzazione", use_container_width=True)

    st.markdown("---")
    st.caption("Consiglio iniziale:")
    st.markdown("- prova con 60-120 pagine")
    st.markdown("- poi aumenta solo se serve")

# =========================
# MAIN UI
# =========================
st.title("🎣 Assistente MGFishing")
st.caption("Risposte basate sui contenuti del sito MGFishing")

# Prima indicizzazione automatica
need_auto_index = (
    not st.session_state.index_ready
    or st.session_state.last_sitemap_url != sitemap_url
    or st.session_state.last_max_pages != max_pages
)

if update_clicked or need_auto_index:
    progress_bar = st.progress(0)
    status_box = st.empty()

    with st.spinner("Sto leggendo la sitemap e preparando il contenuto del sito..."):
        kb, urls = build_knowledge_base(
            sitemap_url=sitemap_url,
            max_pages=max_pages,
            progress_bar=progress_bar,
            status_box=status_box
        )

    st.session_state.knowledge_base = kb
    st.session_state.indexed_urls = urls
    st.session_state.index_ready = True
    st.session_state.last_sitemap_url = sitemap_url
    st.session_state.last_max_pages = max_pages

    progress_bar.empty()
    status_box.empty()

if st.session_state.index_ready:
    st.success(f"Indicizzazione pronta: {len(st.session_state.knowledge_base)} pagine lette")

# Messaggio iniziale
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Ciao, sono l'assistente MGFishing. Chiedimi informazioni sui prodotti presenti sul sito."
    })

# Visualizza chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utente
user_query = st.chat_input("Scrivi qui la tua domanda...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # 1) SALUTI
        if is_greeting(user_query):
            answer = greeting_response()
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # 2) SMALL TALK
        if is_smalltalk(user_query):
            answer = smalltalk_response(user_query)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # 3) CONTROLLO INDICIZZAZIONE
        kb = st.session_state.knowledge_base
        if not kb:
            answer = "Non ho ancora caricato i contenuti del sito. Premi su 'Aggiorna indicizzazione' nella barra laterale."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # 4) CERCA PAGINE RILEVANTI
        relevant_pages = search_relevant_pages(user_query, kb, top_k=4)

        if not relevant_pages:
            answer = fallback_no_results()
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        context = build_context_from_pages(relevant_pages)

        with st.spinner("Sto preparando la risposta..."):
            answer = ask_openai(user_query, context, model_name)

        if not answer or "non so" in normalize_text(answer):
            answer = fallback_no_results()

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
