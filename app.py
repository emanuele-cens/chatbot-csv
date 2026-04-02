import os
import re
import time
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from rapidfuzz import fuzz
from openai import OpenAI

# =========================
# CONFIG
# =========================
SITE_URL = "https://www.mgfishing.eu"
SITEMAP_CANDIDATES = [
    f"{SITE_URL}/sitemap.xml",
    f"{SITE_URL}/1_index_sitemap.xml",
]

WHATSAPP_NUMBER = "3494166335"
WHATSAPP_NAME = "Emanuele"

PRODUCT_QUERY_HINTS = [
    "prodotto", "mulinello", "canna", "trecciato", "monofilo", "fluorocarbon",
    "artificiale", "esca", "popper", "minnow", "jig", "amo", "ami", "girella",
    "galleggiante", "bombarda", "lenza", "trabucco", "daiwa", "shimano", "colmic",
    "tubertini", "yuki", "molix", "major craft", "rapture", "xenos", "atmos",
    "caldia", "crossfire", "ballistic", "salty pop", "tk4", "x-rider"
]

GENERAL_QUERY_HINTS = [
    "meteo", "vento", "mare", "montatura", "trave", "terminale", "surfcasting",
    "spinning", "trota", "bolognese", "feeder", "bolentino", "carpfishing",
    "serra", "spigola", "palamita", "lampuga", "che canna", "che mulinello",
    "consiglio", "consigli", "come fare", "come pescare"
]

OPERATOR_HINTS = [
    "operatore", "umano", "persona", "assistenza", "whatsapp", "contatto",
    "parlare con qualcuno", "parlare con operatore", "aiuto umano"
]

SHIPPING_HINTS = [
    "spedizione", "tracking", "ordine", "consegna", "dove si trova il mio ordine",
    "quando arriva", "corriere", "pacco"
]


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="MGFishing Chatbot", page_icon="🎣", layout="centered")
st.title("🎣 MGFishing Assistant")

st.caption("Consigli su prodotti, pesca, montature e supporto clienti MGFishing")

# =========================
# OPENAI CLIENT
# =========================
def get_openai_api_key():
    # Prima prova Streamlit secrets, poi env var
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None

    if not key:
        key = os.getenv("OPENAI_API_KEY", "")

    return key

OPENAI_API_KEY = get_openai_api_key()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9àèéìòù\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def slug_to_name(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.split("?")[0]
    slug = slug.replace(".html", "")
    slug = slug.replace("-", " ")
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug.title()

def looks_like_product_url(url: str) -> bool:
    u = url.lower()

    # Escludi categorie, blog, pagine generiche, account, carrello, moduli ecc.
    excluded = [
        "/blog", "/category", "/categoria", "/module", "/cart", "/carrello",
        "/order", "/ordine", "/login", "/my-account", "/content", "/cms",
        "/manufacturer", "/brand", "/page", "/new-products", "/best-sales",
        "/prices-drop", "/stores", "/contact", "/contatti", "/sitemap",
        "/search", "/ricerca", "/accessories", "/supplier", "/tag"
    ]
    if any(x in u for x in excluded):
        return False

    # Se URL ha .html oppure slug lungo, probabile prodotto
    if u.endswith(".html"):
        return True

    path = u.replace(SITE_URL.lower(), "")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 1 and len(parts[-1]) > 8:
        return True

    return False

def likely_product_question(text: str) -> bool:
    t = normalize_text(text)
    return any(k in t for k in PRODUCT_QUERY_HINTS)

def likely_general_question(text: str) -> bool:
    t = normalize_text(text)
    return any(k in t for k in GENERAL_QUERY_HINTS)

def likely_operator_request(text: str) -> bool:
    t = normalize_text(text)
    return any(k in t for k in OPERATOR_HINTS)

def likely_shipping_request(text: str) -> bool:
    t = normalize_text(text)
    return any(k in t for k in SHIPPING_HINTS)


# =========================
# SITEMAP / CATALOG CACHE
# =========================
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # 12 ore
def fetch_sitemap_urls():
    urls = []

    for sm in SITEMAP_CANDIDATES:
        try:
            r = requests.get(sm, timeout=20)
            if r.status_code == 200 and ("xml" in r.headers.get("content-type", "").lower() or r.text.strip().startswith("<")):
                found = parse_sitemap(sm, r.text)
                urls.extend(found)
        except Exception:
            continue

    # dedup
    urls = list(dict.fromkeys(urls))
    return urls

def parse_sitemap(base_url: str, xml_text: str):
    found_urls = []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return found_urls

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Caso sitemapindex
    if root.tag.endswith("sitemapindex"):
        for sitemap in root.findall(f".//{ns}sitemap"):
            loc = sitemap.find(f"{ns}loc")
            if loc is not None and loc.text:
                child_url = loc.text.strip()
                try:
                    r = requests.get(child_url, timeout=20)
                    if r.status_code == 200:
                        found_urls.extend(parse_sitemap(child_url, r.text))
                except Exception:
                    pass
        return found_urls

    # Caso urlset
    if root.tag.endswith("urlset"):
        for urlnode in root.findall(f".//{ns}url"):
            loc = urlnode.find(f"{ns}loc")
            if loc is not None and loc.text:
                found_urls.append(loc.text.strip())

    return found_urls

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def build_catalog():
    urls = fetch_sitemap_urls()
    products = []

    for url in urls:
        if looks_like_product_url(url):
            name = slug_to_name(url)
            norm_name = normalize_text(name)
            products.append({
                "name": name,
                "norm_name": norm_name,
                "url": url
            })

    # dedup final by url
    dedup = {}
    for p in products:
        dedup[p["url"]] = p

    return list(dedup.values())


# =========================
# PRODUCT MATCHING
# =========================
def extract_candidate_product_text(user_text: str) -> str:
    text = normalize_text(user_text)

    patterns = [
        r"link (?:del|della|di|per)?\s*(.+)",
        r"cerco (.+)",
        r"hai (.+)",
        r"avete (.+)",
        r"mi trovi (.+)",
        r"mi mandi (.+)",
        r"voglio (.+)",
        r"info su (.+)",
        r"parlami di (.+)",
        r"consiglio su (.+)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 2:
                return candidate

    return text

def score_product_match(query: str, product: dict) -> int:
    q = normalize_text(query)
    p = product["norm_name"]

    score1 = fuzz.token_set_ratio(q, p)
    score2 = fuzz.partial_ratio(q, p)
    score3 = fuzz.ratio(q, p)

    # bonus se tutte le parole query sono nel nome
    q_words = [w for w in q.split() if len(w) > 2]
    contains_bonus = 0
    if q_words and all(w in p for w in q_words):
        contains_bonus = 12

    # bonus se almeno metà parole query sono nel nome
    half_bonus = 0
    if q_words:
        hits = sum(1 for w in q_words if w in p)
        if hits >= max(1, len(q_words) // 2):
            half_bonus = 7

    final_score = int(max(score1, score2, score3) + contains_bonus + half_bonus)
    return min(final_score, 100)

def find_best_products(user_text: str, top_k: int = 5):
    catalog = build_catalog()
    query = extract_candidate_product_text(user_text)

    scored = []
    for product in catalog:
        s = score_product_match(query, product)
        scored.append((s, product))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def get_best_product_link(user_text: str):
    matches = find_best_products(user_text, top_k=5)
    if not matches:
        return None, []

    best_score, best_product = matches[0]
    alternatives = [p for s, p in matches[1:] if s >= 60]

    # soglie più permissive per evitare falsi "non disponibile"
    if best_score >= 78:
        return best_product, alternatives

    # se il nome contiene parole molto indicative accetta anche score medio
    query = extract_candidate_product_text(user_text)
    q_words = [w for w in normalize_text(query).split() if len(w) > 3]
    if q_words:
        hits = sum(1 for w in q_words if w in best_product["norm_name"])
        if hits >= max(1, len(q_words) - 1) and best_score >= 68:
            return best_product, alternatives

    return None, [p for s, p in matches if s >= 58]


# =========================
# STATIC BUSINESS RULES
# =========================
def shipping_reply(user_text: str) -> str:
    return (
        "Per informazioni su spedizione, tracking o stato ordine, "
        f"ti consiglio di contattare direttamente {WHATSAPP_NAME} su WhatsApp al {WHATSAPP_NUMBER}. "
        "In questo modo puoi ricevere assistenza più precisa e veloce sul tuo ordine."
    )

def operator_reply() -> str:
    return (
        f"Per parlare con un operatore ti consiglio di contattare direttamente {WHATSAPP_NAME} "
        f"su WhatsApp al {WHATSAPP_NUMBER}."
    )

def fallback_reply() -> str:
    return (
        f"Per questa richiesta ti consiglio di contattare direttamente {WHATSAPP_NAME} "
        f"su WhatsApp al {WHATSAPP_NUMBER}, così puoi ricevere assistenza più precisa."
    )


# =========================
# LLM PROMPTS
# =========================
STORE_SYSTEM_PROMPT = f"""
Sei l'assistente clienti di MGFishing, ecommerce italiano di articoli da pesca.

Regole fondamentali:
- Se la richiesta riguarda prodotti, devi basarti prima di tutto sui prodotti/link trovati nel catalogo MGFishing forniti nel contesto.
- Non dire mai in modo sbrigativo che un prodotto non è disponibile se nel contesto c'è una possibile corrispondenza.
- Se esiste un link prodotto nel contesto, privilegia SEMPRE il link prodotto diretto rispetto a link categoria.
- Se la richiesta è generale (montature, tecniche, meteo, consigli pesca), rispondi in modo utile, chiaro e naturale.
- Non citare altri siti, non inserire fonti, non dire "secondo il sito X".
- Se l'utente chiede un operatore umano, tracking, spedizione, stato ordine, resi, oppure non sei sicuro della risposta, indirizza a WhatsApp:
  {WHATSAPP_NAME} - {WHATSAPP_NUMBER}
- Scrivi in italiano.
- Tono professionale ma semplice.
- Non inventare disponibilità certa se non è indicata.
- Se hai un prodotto compatibile o molto probabile, proponilo con il suo link.
"""

def call_openai_for_product(user_text: str, matched_product: dict | None, alternatives: list[dict]):
    if not client:
        return None

    context_lines = []

    if matched_product:
        context_lines.append(
            f"PRODOTTO PRINCIPALE TROVATO:\n"
            f"Nome: {matched_product['name']}\n"
            f"Link: {matched_product['url']}"
        )

    if alternatives:
        alt_text = "\n".join([f"- {p['name']} -> {p['url']}" for p in alternatives[:4]])
        context_lines.append(f"ALTERNATIVE TROVATE:\n{alt_text}")

    user_prompt = f"""
Domanda cliente:
{user_text}

Contesto catalogo MGFishing:
{chr(10).join(context_lines) if context_lines else "Nessun prodotto trovato con alta certezza."}

Istruzioni:
- Se il prodotto principale è presente, rispondi usando quel link.
- Se ci sono alternative valide, puoi citarle.
- Non parlare di altri siti.
- Non essere vago.
- Se non sei abbastanza sicuro, invita a contattare {WHATSAPP_NAME} su WhatsApp al {WHATSAPP_NUMBER}.
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": STORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return resp.output_text.strip()
    except Exception:
        return None

def call_openai_general(user_text: str):
    if not client:
        return None

    user_prompt = f"""
Richiesta del cliente:
{user_text}

Rispondi in italiano.
Puoi rispondere in modo utile e pratico su pesca, montature, meteo, tecniche e consigli generali.
Non citare fonti o nomi di altri siti.
Se la richiesta richiede chiaramente assistenza umana o post-vendita, invita a contattare {WHATSAPP_NAME} su WhatsApp al {WHATSAPP_NUMBER}.
"""

    # Prima prova con web_search_preview, poi fallback normale
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": STORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[{"type": "web_search_preview"}],
            temperature=0.4,
        )
        text = resp.output_text.strip()
        if text:
            return text
    except Exception:
        pass

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": STORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return resp.output_text.strip()
    except Exception:
        return None


# =========================
# MAIN RESPONSE LOGIC
# =========================
def generate_reply(user_text: str) -> str:
    t = normalize_text(user_text)

    # saluti molto semplici
    if t in ["ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"]:
        return (
            "Ciao! Benvenuto su MGFishing 🎣\n\n"
            "Posso aiutarti a trovare prodotti sul nostro sito, consigliarti attrezzatura, "
            "oppure darti indicazioni su montature, tecniche e pesca."
        )

    # operatore umano
    if likely_operator_request(user_text):
        return operator_reply()

    # spedizioni / tracking / ordine
    if likely_shipping_request(user_text):
        return shipping_reply(user_text)

    # richieste prodotto
    if likely_product_question(user_text):
        matched_product, alternatives = get_best_product_link(user_text)

        # Se abbiamo un match concreto, usiamolo
        if matched_product:
            ai_reply = call_openai_for_product(user_text, matched_product, alternatives)
            if ai_reply:
                return ai_reply

            # fallback semplice senza AI
            msg = (
                f"Ho trovato questo prodotto sul sito MGFishing:\n\n"
                f"**{matched_product['name']}**\n{matched_product['url']}"
            )
            if alternatives:
                msg += "\n\nPotrebbero interessarti anche:\n"
                for p in alternatives[:3]:
                    msg += f"- {p['name']} → {p['url']}\n"
            return msg

        # Se non c'è match forte ma ci sono alternative plausibili
        if alternatives:
            ai_reply = call_openai_for_product(user_text, None, alternatives)
            if ai_reply:
                return ai_reply

            msg = "Non ho trovato un match perfetto, ma ho trovato questi prodotti simili sul sito MGFishing:\n\n"
            for p in alternatives[:4]:
                msg += f"- **{p['name']}**\n  {p['url']}\n"
            return msg

        return fallback_reply()

    # domande generali pesca / meteo / montature
    if likely_general_question(user_text):
        ai_reply = call_openai_general(user_text)
        if ai_reply:
            return ai_reply
        return fallback_reply()

    # fallback finale con AI generale
    ai_reply = call_openai_general(user_text)
    if ai_reply:
        return ai_reply

    return fallback_reply()


# =========================
# CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ciao! Sono l’assistente MGFishing 🎣\nPosso aiutarti con prodotti, consigli di pesca, montature e informazioni generali."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Scrivi qui la tua domanda...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta migliore..."):
            reply = generate_reply(prompt)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.subheader("Catalogo MGFishing")
    if st.button("Aggiorna catalogo prodotti"):
        build_catalog.clear()
        fetch_sitemap_urls.clear()
        st.success("Cache catalogo svuotata. Al prossimo messaggio verrà ricaricata.")

    try:
        catalog = build_catalog()
        st.caption(f"Prodotti/URL indicizzati: {len(catalog)}")
    except Exception:
        st.caption("Catalogo non disponibile al momento.")

    if not OPENAI_API_KEY:
        st.warning("Manca OPENAI_API_KEY nei secrets o nelle variabili ambiente.")
