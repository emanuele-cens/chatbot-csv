import os
import re
import io
import csv
import json
import unicodedata
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="MGFishing Chatbot", page_icon="🎣", layout="centered")

CSV_PATH = "prodotti.csv"
KNOWLEDGE_PATH = "knowledge.txt"
MEMORY_PATH = "memory.json"
WHATSAPP_NUMBER = "393494166335"
WHATSAPP_LABEL = "MGFishing"
ADMIN_MODE = True  # METTI False QUANDO IL BOT SARA' USATO DAI CLIENTI

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================================================
# UTILS
# =========================================================
def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9àèéìòù\s€.,:/_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9€]+", normalize_text(text)) if len(t) > 1]


def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_float(value):
    if value is None:
        return None
    s = str(value).strip().replace("€", "").replace("eur", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s.lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def whatsapp_link(message: str) -> str:
    from urllib.parse import quote
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(message)}"


def contains_any(text: str, words) -> bool:
    t = normalize_text(text)
    return any(normalize_text(w) in t for w in words)


def ensure_file(path: str, default_content: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_content)


# =========================================================
# DEFAULT FILES
# =========================================================
DEFAULT_MEMORY = {
    "product_links": [
        {
            "name": "Trabucco Xenos SW",
            "url": "",
            "aliases": ["xenos sw", "trabucco xenos sw", "mulinello trabucco xenos sw"]
        }
    ],
    "faq": [
        {
            "question": "quali metodi di pagamento accettate",
            "answer": "Puoi verificare i metodi di pagamento disponibili direttamente durante il checkout sul sito. Per un aiuto immediato puoi contattarci anche su WhatsApp."
        },
        {
            "question": "in quanto spedite",
            "answer": "Gli ordini vengono gestiti rapidamente. Per urgenze o richieste particolari puoi contattarci su WhatsApp."
        }
    ],
    "rules": [
        "Quando l'utente chiede un link prodotto, dare il link esatto del prodotto se presente in memory.json.",
        "Non inventare mai link prodotto.",
        "Se non trovi il prodotto preciso, non dare il link della categoria come se fosse il prodotto.",
        "Per domande su ordini, problemi o casi particolari, invita a contattare WhatsApp.",
        "Rispondi in modo semplice, utile e commerciale."
    ],
    "aliases": [
        {
            "term": "trota lago",
            "synonyms": ["tremarella", "trout area", "pesca trota lago"]
        },
        {
            "term": "mulinello",
            "synonyms": ["reel", "bobina"]
        }
    ]
}

DEFAULT_KNOWLEDGE = """Benvenuto su MGFishing.

Informazioni negozio:
- Per assistenza rapida puoi contattarci su WhatsApp.
- Per ordini particolari, problemi o richieste specifiche il contatto diretto è consigliato.
- Aggiorna questo file con le tue vere informazioni su spedizioni, pagamenti, resi e assistenza.
"""

DEFAULT_CSV = """name,description,price,url,category,brand
Trabucco Xenos SW,Mulinello per spinning potente e pesche gravose,,,Mulinelli,Trabucco
Daiwa Salty Pop 95 F,Esca artificiale topwater popper floating 16.5g 9.5cm,,,Artificiali,Daiwa
"""

ensure_file(MEMORY_PATH, json.dumps(DEFAULT_MEMORY, ensure_ascii=False, indent=2))
ensure_file(KNOWLEDGE_PATH, DEFAULT_KNOWLEDGE)
ensure_file(CSV_PATH, DEFAULT_CSV)

# =========================================================
# LOAD FILES
# =========================================================
def try_read_csv_with_sep(content: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(content),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    )


@st.cache_data(show_spinner=False)
def load_catalog(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()

    best_df = None
    best_cols = 0

    for sep in [",", ";", "\t", "|"]:
        try:
            df = try_read_csv_with_sep(content, sep)
            if df is not None and len(df.columns) > best_cols:
                best_df = df
                best_cols = len(df.columns)
        except Exception:
            pass

    if best_df is None:
        try:
            best_df = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            return pd.DataFrame()

    best_df.columns = [str(c).strip() for c in best_df.columns]
    best_df = best_df.fillna("")

    def find_col(possibles):
        cols = list(best_df.columns)
        for p in possibles:
            for c in cols:
                if c.lower().strip() == p.lower().strip():
                    return c
        for p in possibles:
            for c in cols:
                if p.lower().strip() in c.lower().strip():
                    return c
        return None

    col_name = find_col(["name", "nome", "title", "titolo", "product_name", "nome prodotto"])
    col_desc = find_col(["description", "descrizione", "short_description", "desc"])
    col_price = find_col(["price", "prezzo", "final_price", "prezzo finale", "prezzo iva incl"])
    col_url = find_col(["url", "link", "product_url", "permalink", "href"])
    col_cat = find_col(["category", "categoria", "cat"])
    col_brand = find_col(["brand", "marca", "manufacturer"])

    if not col_name and len(best_df.columns) > 0:
        col_name = best_df.columns[0]

    std = pd.DataFrame()
    std["name"] = best_df[col_name] if col_name in best_df.columns else ""
    std["description"] = best_df[col_desc] if col_desc in best_df.columns else ""
    std["price"] = best_df[col_price] if col_price in best_df.columns else ""
    std["url"] = best_df[col_url] if col_url in best_df.columns else ""
    std["category"] = best_df[col_cat] if col_cat in best_df.columns else ""
    std["brand"] = best_df[col_brand] if col_brand in best_df.columns else ""

    std = std.fillna("")
    std = std[std["name"].astype(str).str.strip() != ""].copy()

    std["name_norm"] = std["name"].apply(normalize_text)
    std["desc_norm"] = std["description"].apply(normalize_text)
    std["category_norm"] = std["category"].apply(normalize_text)
    std["brand_norm"] = std["brand"].apply(normalize_text)
    std["all_text"] = (
        std["name_norm"] + " " + std["desc_norm"] + " " + std["category_norm"] + " " + std["brand_norm"]
    ).str.strip()
    std["price_num"] = std["price"].apply(safe_float)

    std = std[std["name_norm"].str.len() > 2].copy()
    std = std.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)
    return std


def load_knowledge(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_memory(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return DEFAULT_MEMORY
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ["product_links", "faq", "rules", "aliases"]:
            if key not in data or not isinstance(data[key], list):
                data[key] = []
        return data
    except Exception:
        return DEFAULT_MEMORY


def save_memory(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


catalog_df = load_catalog(CSV_PATH)
knowledge_text = load_knowledge(KNOWLEDGE_PATH)
memory_data = load_memory(MEMORY_PATH)

# =========================================================
# KEYWORDS / INTENT
# =========================================================
GREETING_WORDS = {"ciao", "salve", "buongiorno", "buonasera", "hey"}
LINK_WORDS = {
    "link", "mandami il link", "inviami il link", "url", "pagina prodotto",
    "scheda prodotto", "apri prodotto", "dove lo trovo", "dammi il link"
}
STORE_INFO_WORDS = {
    "spedizione", "spedizioni", "tracking", "tracciamento", "consegna", "consegne",
    "ordine", "stato ordine", "tempi di spedizione", "resi", "reso",
    "pagamento", "pagamenti", "pagare", "come posso pagare", "metodi di pagamento",
    "contrassegno", "carta", "paypal", "bonifico", "assistenza", "whatsapp",
    "tempo di consegna", "quanto costa la spedizione", "costo spedizione"
}
PRODUCT_HINTS = {
    "canna", "canne", "mulinello", "mulinelli", "trecciato", "monofilo", "fluorocarbon",
    "bombarda", "galleggiante", "guadino", "artificiale", "artificiali", "esca", "esche",
    "jig", "popper", "minnow", "feeder", "surfcasting", "spinning", "trota", "bolognese",
    "eging", "bolentino", "ledgering", "carpfishing", "ami", "amo", "girelle", "piombi",
    "kit", "combo"
}
WEB_ADVICE_WORDS = {
    "meteo", "vento", "mare", "onde", "marea", "pressione", "pioggia", "temperatura",
    "montatura", "montature", "trave", "finale", "terminale", "terminali", "innesco",
    "come pescare", "quando pescare", "pesca", "orario migliore", "luna", "corrente",
    "spiaggia", "scaduta", "acqua velata", "acqua torbida", "mareggiata"
}
STOPWORDS = {
    "mi", "puoi", "potresti", "vorrei", "voglio", "consigliami", "consiglia", "consiglio",
    "una", "uno", "dei", "delle", "per", "da", "di", "su", "con", "e", "o", "il", "lo",
    "la", "i", "gli", "le", "un", "nel", "nella", "dello", "della", "del", "delle",
    "fascia", "media", "medio", "economica", "economico", "top", "migliore", "migliori",
    "circa", "sui", "sul", "euro", "prezzo", "budget", "adatta", "adatto", "cerco", "cercando"
}

# =========================================================
# MEMORY SEARCH
# =========================================================
def apply_aliases(query: str, aliases: List[Dict[str, Any]]) -> str:
    q = normalize_text(query)
    for item in aliases:
        term = normalize_text(item.get("term", ""))
        synonyms = [normalize_text(x) for x in item.get("synonyms", [])]
        for syn in synonyms:
            if syn and syn in q:
                q = q.replace(syn, term)
    return q


def score_text_match(query: str, target: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(target))
    if not q_tokens or not t_tokens:
        return 0.0
    common = q_tokens.intersection(t_tokens)
    score = len(common) * 10
    if normalize_text(target) in normalize_text(query):
        score += 30
    return float(score)


def find_memory_product_link(query: str, memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = apply_aliases(query, memory.get("aliases", []))
    best_item = None
    best_score = 0.0

    for item in memory.get("product_links", []):
        name = item.get("name", "")
        aliases = item.get("aliases", [])
        url = str(item.get("url", "")).strip()
        if not url:
            continue

        targets = [name] + aliases
        item_score = max(score_text_match(q, t) for t in targets if str(t).strip()) if targets else 0.0
        if item_score > best_score:
            best_score = item_score
            best_item = item

    return best_item if best_score >= 20 else None


def find_memory_faq(query: str, memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = apply_aliases(query, memory.get("aliases", []))
    best_item = None
    best_score = 0.0

    for item in memory.get("faq", []):
        question = item.get("question", "")
        item_score = score_text_match(q, question)
        if item_score > best_score:
            best_score = item_score
            best_item = item

    return best_item if best_score >= 20 else None

# =========================================================
# PRODUCT SEARCH
# =========================================================
def extract_budget(query: str):
    q = normalize_text(query)
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*€",
        r"(\d+(?:[.,]\d+)?)\s*euro",
        r"entro\s+(\d+(?:[.,]\d+)?)",
        r"sui\s+(\d+(?:[.,]\d+)?)",
        r"max\s+(\d+(?:[.,]\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                pass
    return None


def expand_query_tokens(query: str) -> List[str]:
    toks = [t for t in tokenize(query) if t not in STOPWORDS]
    out = list(toks)
    synonyms = {
        "trota": ["trout", "area", "tremarella"],
        "surfcasting": ["surf", "beach", "ledgering"],
        "spinning": ["spin"],
        "bolognese": ["bolo"],
        "canna": ["canne", "rod"],
        "canne": ["canna", "rod"],
        "mulinello": ["mulinelli", "reel"],
        "mulinelli": ["mulinello", "reel"],
        "artificiale": ["artificiali", "minnow", "popper", "jig"],
        "artificiali": ["artificiale", "minnow", "popper", "jig"],
    }
    for t in list(toks):
        out.extend(synonyms.get(t, []))
    return unique_preserve(out)


def score_product(row: pd.Series, query: str, tokens: List[str], budget=None) -> float:
    name = row["name_norm"]
    text = row["all_text"]
    score = 0.0

    for tok in tokens:
        if tok in name:
            score += 10
        elif tok in text:
            score += 4

    qn = normalize_text(query)
    if qn and qn in text:
        score += 20

    price = row.get("price_num")
    if budget is not None and price is not None and pd.notna(price):
        if price <= budget:
            score += 8
        elif price <= budget * 1.10:
            score += 2
        else:
            score -= 6

    if "canna" in qn or "canne" in qn:
        if re.search(r"\bcanna\b|\bcanne\b|\brod\b", text):
            score += 10
        else:
            score -= 20

    if "mulinello" in qn or "mulinelli" in qn:
        if re.search(r"\bmulinello\b|\bmulinelli\b|\breel\b", text):
            score += 10
        else:
            score -= 20

    return score


def search_products(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tokens = expand_query_tokens(query)
    budget = extract_budget(query)

    temp = df.copy()
    temp["score"] = temp.apply(lambda row: score_product(row, query, tokens, budget), axis=1)

    qn = normalize_text(query)

    if "canna" in qn or "canne" in qn:
        temp = temp[temp["all_text"].str.contains(r"\bcanna\b|\bcanne\b|\brod\b", regex=True, na=False)].copy()

    if "mulinello" in qn or "mulinelli" in qn:
        temp = temp[temp["all_text"].str.contains(r"\bmulinello\b|\bmulinelli\b|\breel\b", regex=True, na=False)].copy()

    if budget is not None:
        under = temp[(temp["price_num"].notna()) & (temp["price_num"] <= budget * 1.10)].copy()
        if not under.empty:
            temp = under

    temp = temp.sort_values(by="score", ascending=False)
    temp = temp[temp["score"] > 0]
    return temp.head(top_k).copy()

# =========================================================
# OPENAI
# =========================================================
def ask_openai(system_prompt: str, user_prompt: str, temperature: float = 0.25) -> str:
    if client is None:
        return "Al momento il servizio AI non è disponibile."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Al momento non riesco a generare una risposta corretta."
    except Exception:
        return "Al momento non riesco a generare una risposta corretta."

# =========================================================
# RESPONSES
# =========================================================
def greeting_response() -> str:
    return (
        "Ciao! Sono l’assistente MGFishing 🎣\n\n"
        "Posso aiutarti con:\n"
        "- consigli sui prodotti del catalogo\n"
        "- link prodotti salvati in memoria\n"
        "- pagamenti, spedizioni e info negozio\n"
        "- montature e consigli di pesca"
    )


def product_link_response(query: str, memory: Dict[str, Any], df: pd.DataFrame) -> str:
    mem_match = find_memory_product_link(query, memory)
    if mem_match:
        return f"Ecco il link del prodotto:\n\n{mem_match['url']}"

    results = search_products(query, df, top_k=3)
    if not results.empty:
        exact_urls = []
        for _, row in results.iterrows():
            url = str(row.get("url", "")).strip()
            name = str(row.get("name", "")).strip()
            if url:
                exact_urls.append(f"- **{name}**\n{url}")

        if exact_urls:
            return "Ho trovato questi link prodotto pertinenti:\n\n" + "\n\n".join(exact_urls)

    return (
        "Non ho trovato un link prodotto preciso salvato in memoria.\n\n"
        f"Per il link esatto ti consiglio di scrivere su WhatsApp:\n\n"
        f"{whatsapp_link(f'Ciao, mi serve il link preciso di questo prodotto: {query}')}"
    )


def store_info_response(query: str, knowledge: str, memory: Dict[str, Any]) -> str:
    faq_match = find_memory_faq(query, memory)
    if faq_match:
        return str(faq_match.get("answer", "")).strip()

    if not knowledge.strip():
        return (
            f"Per assistenza diretta ti consiglio di contattarci su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao, avrei bisogno di informazioni sul negozio.')}"
        )

    system_prompt = (
        "Sei l'assistente clienti di MGFishing. "
        "Usa SOLO le informazioni fornite. "
        "Se l'informazione non è presente, invita a contattare WhatsApp. "
        "Non inventare nulla."
    )
    user_prompt = f"Knowledge:\n{knowledge}\n\nDomanda utente:\n{query}"
    reply = ask_openai(system_prompt, user_prompt, temperature=0.1)

    if not reply or "non riesco" in normalize_text(reply):
        return (
            f"Per questa informazione ti consiglio di contattarci su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao, avrei bisogno di informazioni su: {query}')}"
        )
    return reply


def product_response(query: str, df: pd.DataFrame, memory: Dict[str, Any]) -> str:
    memory_link = find_memory_product_link(query, memory)
    if memory_link:
        return f"Ho trovato il prodotto in memoria. Ecco il link corretto:\n\n{memory_link['url']}"

    if df.empty:
        return (
            "Al momento non riesco a leggere il catalogo prodotti.\n\n"
            f"Per un aiuto immediato puoi contattarci su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao, mi serve un consiglio su un prodotto.')}"
        )

    results = search_products(query, df, top_k=5)
    if results.empty:
        return (
            "Non sono riuscito a trovare nel catalogo prodotti davvero pertinenti alla tua richiesta.\n\n"
            f"Per un consiglio più preciso ti consiglio di contattarci su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao, sto cercando questo prodotto: {query}')}"
        )

    catalog_context = []
    for _, row in results.iterrows():
        catalog_context.append({
            "nome": row.get("name", ""),
            "prezzo": row.get("price", ""),
            "categoria": row.get("category", ""),
            "marca": row.get("brand", ""),
            "descrizione": row.get("description", ""),
            "url": row.get("url", ""),
        })

    rules = memory.get("rules", [])
    system_prompt = (
        "Sei un assistente esperto di pesca per MGFishing. "
        "Devi consigliare SOLO prodotti presenti nel catalogo fornito. "
        "Non inventare prodotti, categorie, marchi, prezzi o caratteristiche. "
        "Se l'utente chiede un link e c'è un url nel catalogo, puoi riportarlo. "
        "Se i risultati sono pochi o non perfetti, dillo con onestà. "
        "Niente riferimenti esterni.\n\n"
        f"Regole aggiuntive:\n- " + "\n- ".join(rules)
    )
    user_prompt = (
        f"Richiesta cliente: {query}\n\n"
        f"Prodotti trovati nel catalogo:\n{json.dumps(catalog_context, ensure_ascii=False, indent=2)}\n\n"
        "Rispondi in modo chiaro, commerciale e utile."
    )
    return ask_openai(system_prompt, user_prompt, temperature=0.25)


def web_advice_response(query: str) -> str:
    system_prompt = (
        "Sei un assistente esperto di pesca di MGFishing. "
        "Rispondi a domande su montature, tecniche, terminali, inneschi e consigli pratici. "
        "Non citare siti esterni. Se mancano dati, dillo e dai comunque una linea guida utile."
    )
    return ask_openai(system_prompt, f"Domanda cliente: {query}", temperature=0.35)


def detect_intent(query: str) -> str:
    q = normalize_text(query)
    if q in GREETING_WORDS:
        return "greeting"
    if contains_any(q, LINK_WORDS):
        return "product_link"
    if contains_any(q, STORE_INFO_WORDS):
        return "store_info"
    if contains_any(q, WEB_ADVICE_WORDS):
        return "web_advice"
    if contains_any(q, PRODUCT_HINTS):
        return "product_advice"
    return "generic"


def generic_response(query: str, df: pd.DataFrame, knowledge: str, memory: Dict[str, Any]) -> str:
    faq_match = find_memory_faq(query, memory)
    if faq_match:
        return str(faq_match.get("answer", "")).strip()

    memory_link = find_memory_product_link(query, memory)
    if memory_link:
        return f"Ho trovato questo prodotto in memoria. Ecco il link corretto:\n\n{memory_link['url']}"

    qn = normalize_text(query)
    if contains_any(qn, STORE_INFO_WORDS):
        return store_info_response(query, knowledge, memory)

    maybe_products = search_products(query, df, top_k=3)
    if not maybe_products.empty:
        return product_response(query, df, memory)

    return web_advice_response(query)


def generate_response(query: str) -> str:
    memory = load_memory(MEMORY_PATH)
    knowledge = load_knowledge(KNOWLEDGE_PATH)
    catalog = load_catalog(CSV_PATH)
    intent = detect_intent(query)

    if intent == "greeting":
        return greeting_response()
    if intent == "product_link":
        return product_link_response(query, memory, catalog)
    if intent == "store_info":
        return store_info_response(query, knowledge, memory)
    if intent == "product_advice":
        return product_response(query, catalog, memory)
    if intent == "web_advice":
        return web_advice_response(query)
    return generic_response(query, catalog, knowledge, memory)

# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
    .subtitle { color: #666; margin-bottom: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎣 MGFishing Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Chatbot base con memoria controllata da te</div>',
    unsafe_allow_html=True,
)

if ADMIN_MODE:
    with st.sidebar:
        st.markdown("## Modalità Training")
        st.caption("Questa modalità serve solo a te per insegnare nuovi link e risposte al bot.")

        with st.expander("Aggiungi link prodotto"):
            product_name = st.text_input("Nome prodotto")
            product_url = st.text_input("URL prodotto")
            product_aliases = st.text_area("Alias separati da virgola", placeholder="xenos sw, trabucco xenos sw")
            if st.button("Salva link prodotto"):
                mem = load_memory(MEMORY_PATH)
                mem["product_links"].append({
                    "name": product_name.strip(),
                    "url": product_url.strip(),
                    "aliases": [x.strip() for x in product_aliases.split(",") if x.strip()],
                })
                save_memory(MEMORY_PATH, mem)
                st.cache_data.clear()
                st.success("Link prodotto salvato in memory.json")

        with st.expander("Aggiungi FAQ"):
            faq_q = st.text_input("Domanda FAQ")
            faq_a = st.text_area("Risposta FAQ")
            if st.button("Salva FAQ"):
                mem = load_memory(MEMORY_PATH)
                mem["faq"].append({"question": faq_q.strip(), "answer": faq_a.strip()})
                save_memory(MEMORY_PATH, mem)
                st.cache_data.clear()
                st.success("FAQ salvata in memory.json")

        with st.expander("Aggiungi alias"):
            alias_term = st.text_input("Termine principale")
            alias_syn = st.text_area("Sinonimi separati da virgola")
            if st.button("Salva alias"):
                mem = load_memory(MEMORY_PATH)
                mem["aliases"].append({
                    "term": alias_term.strip(),
                    "synonyms": [x.strip() for x in alias_syn.split(",") if x.strip()],
                })
                save_memory(MEMORY_PATH, mem)
                st.cache_data.clear()
                st.success("Alias salvato in memory.json")

        with st.expander("Vedi memoria attuale"):
            st.json(load_memory(MEMORY_PATH))

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono l’assistente MGFishing 🎣\n\n"
                "Posso aiutarti con prodotti, link, info negozio e consigli di pesca."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Scrivi qui la tua domanda...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sto elaborando la risposta..."):
            answer = generate_response(user_query)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
