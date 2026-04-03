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
WHATSAPP_LABEL = "Emanuele"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================================================
# DEFAULT MEMORY
# =========================================================
DEFAULT_MEMORY = {
    "global_rules": [
        "Non proporre prodotti con prezzo uguale a 0",
        "Non proporre prodotti con categoria Home",
        "Non proporre prodotti non pertinenti alla richiesta",
        "Se l'utente chiede un link prodotto, cercare prima il prodotto esatto e non la categoria",
        "Se non sei sicuro del prodotto esatto, dillo chiaramente"
    ],
    "bad_examples": [],
    "link_rules": [
        "Preferire sempre il link del prodotto esatto se disponibile",
        "Non dare il link categoria se esiste il prodotto specifico"
    ],
    "manual_faq": [],
    "training_notes": []
}


# =========================================================
# UTILS
# =========================================================
def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9àèéìòù\s€.,:/_+-]", " ", text)
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


def short_text(text: str, max_len: int = 300) -> str:
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# =========================================================
# MEMORY
# =========================================================
def ensure_memory_file():
    if not os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_MEMORY, f, ensure_ascii=False, indent=2)


def load_memory() -> Dict[str, Any]:
    ensure_memory_file()
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_MEMORY.copy()
        for k, v in DEFAULT_MEMORY.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return DEFAULT_MEMORY.copy()


def save_memory(data: Dict[str, Any]):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_training_note(user_request: str, wrong_reply: str, lesson: str):
    memory = load_memory()
    memory.setdefault("bad_examples", []).append({
        "user_request": user_request.strip(),
        "wrong_reply": wrong_reply.strip(),
        "lesson": lesson.strip()
    })
    save_memory(memory)


def add_global_rule(rule: str):
    rule = rule.strip()
    if not rule:
        return
    memory = load_memory()
    if rule not in memory.setdefault("global_rules", []):
        memory["global_rules"].append(rule)
    save_memory(memory)


def add_manual_faq(question: str, answer: str):
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return
    memory = load_memory()
    memory.setdefault("manual_faq", []).append({
        "question": question,
        "answer": answer
    })
    save_memory(memory)


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


@st.cache_data(show_spinner=False)
def load_knowledge(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


catalog_df = load_catalog(CSV_PATH)
knowledge_text = load_knowledge(KNOWLEDGE_PATH)

# =========================================================
# KEYWORDS / INTENT
# =========================================================
GREETING_WORDS = {"ciao", "salve", "buongiorno", "buonasera", "hey"}

LINK_WORDS = {
    "link", "mandami il link", "inviami il link", "url", "pagina prodotto",
    "scheda prodotto", "apri prodotto", "dove lo trovo", "dammi il link"
}

PRODUCT_HINTS = {
    "canna", "canne", "mulinello", "mulinelli", "trecciato", "monofilo", "fluorocarbon",
    "bombarda", "galleggiante", "guadino", "artificiale", "artificiali", "esca", "esche",
    "jig", "popper", "minnow", "feeder", "surfcasting", "spinning", "trota", "bolognese",
    "eging", "bolentino", "ledgering", "carpfishing", "ami", "amo", "girelle", "piombi",
    "kit", "combo", "totanara", "totanare", "polpara", "polpare"
}

STORE_INFO_WORDS = {
    "spedizione", "spedizioni", "tracking", "tracciamento", "consegna", "consegne",
    "ordine", "stato ordine", "tempi di spedizione", "resi", "reso",
    "pagamento", "pagamenti", "pagare", "come posso pagare", "metodi di pagamento",
    "contrassegno", "carta", "paypal", "bonifico", "assistenza", "whatsapp",
    "tempo di consegna", "quanto costa la spedizione", "costo spedizione"
}

WEB_ADVICE_WORDS = {
    "meteo", "vento", "mare", "onde", "marea", "pressione", "pioggia", "temperatura",
    "montatura", "montature", "trave", "finale", "terminale", "terminali", "innesco",
    "come pescare", "quando pescare", "pesca", "orario migliore", "luna", "corrente",
    "spiaggia", "scaduta", "acqua velata", "acqua torbida", "mareggiata" 
    "meteo", "vento", "mare", "onde", "marea", "pressione", "pioggia", "temperatura",
    "montatura", "montature", "trave", "finale", "terminale", "terminali", "innesco",
    "come pescare", "quando pescare", "pesca", "orario migliore", "luna", "corrente",
    "spiaggia", "scaduta", "acqua velata", "acqua torbida", "mareggiata",
    "previsioni", "previsione", "swell", "moto ondoso", "umidita", "umidità"
}

STOPWORDS = {
    "mi", "puoi", "potresti", "vorrei", "voglio", "consigliami", "consiglia", "consiglio",
    "una", "uno", "dei", "delle", "per", "da", "di", "su", "con", "e", "o", "il", "lo",
    "la", "i", "gli", "le", "un", "nel", "nella", "dello", "della", "del", "delle",
    "fascia", "media", "medio", "economica", "economico", "top", "migliore", "migliori",
    "circa", "sui", "sul", "euro", "prezzo", "budget", "adatta", "adatto", "cerco", "cercando"
}


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

def web_advice_response(query: str) -> str:
    manual = check_manual_faq(query)
    if manual:
        return manual

    if client is None:
        return "Al momento il servizio AI non è disponibile."

    system_prompt = (
        "Sei un assistente esperto di pesca di MGFishing.\n"
        "Per domande su meteo, vento, mare, onde, pressione, luna e condizioni pesca, "
        "usa la ricerca web per ottenere informazioni aggiornate.\n"
        "Interpreta sempre la richiesta in ottica pesca pratica.\n"
        "Non citare siti esterni per nome nella risposta finale.\n"
        "Non dire che hai usato strumenti o ricerca web.\n"
        "Rispondi in italiano, in modo chiaro e utile.\n"
        "Se mancano dettagli come località o giorno, dallo notare ma prova comunque a dare una risposta utile."
    )

    user_prompt = f"Domanda cliente: {query}"

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=user_prompt,
            tools=[{"type": "web_search"}]
        )
        return response.output_text.strip()
    except Exception as e:
        return f"Al momento non riesco a recuperare informazioni aggiornate dal web. Errore: {str(e)}"


# =========================================================
# CATALOG SEARCH
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
        "totanara": ["totanare", "egi", "squid"],
        "totanare": ["totanara", "egi", "squid"]
    }

    for t in list(toks):
        out.extend(synonyms.get(t, []))

    return unique_preserve(out)


def is_valid_product_row(row: pd.Series) -> bool:
    category = normalize_text(row.get("category", ""))
    price_num = row.get("price_num", None)
    name = normalize_text(row.get("name", ""))

    if not name:
        return False
    if category == "home":
        return False
    if price_num is not None and pd.notna(price_num) and float(price_num) <= 0:
        return False
    return True


def exact_name_match(query: str, df: pd.DataFrame) -> pd.DataFrame:
    qn = normalize_text(query)
    if not qn or df.empty:
        return pd.DataFrame()

    temp = df.copy()
    temp = temp[temp.apply(is_valid_product_row, axis=1)].copy()

    exact = temp[temp["name_norm"] == qn].copy()
    if not exact.empty:
        return exact.head(5)

    contains = temp[temp["name_norm"].str.contains(re.escape(qn), na=False)].copy()
    if not contains.empty:
        contains["score"] = 100
        return contains.sort_values("name_norm").head(5)

    return pd.DataFrame()


def score_product(row: pd.Series, query: str, tokens: List[str], budget=None) -> float:
    if not is_valid_product_row(row):
        return -9999

    name = row["name_norm"]
    text = row["all_text"]
    desc = row["desc_norm"]
    score = 0.0

    for tok in tokens:
        if tok in name:
            score += 14
        elif tok in desc:
            score += 6
        elif tok in text:
            score += 4

    qn = normalize_text(query)

    if qn and qn == name:
        score += 100
    elif qn and qn in name:
        score += 35
    elif qn and qn in text:
        score += 15

    matched_tokens = sum(1 for tok in tokens if tok in text)
    score += matched_tokens * 2

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
            score -= 25

    if "mulinello" in qn or "mulinelli" in qn:
        if re.search(r"\bmulinello\b|\bmulinelli\b|\breel\b", text):
            score += 10
        else:
            score -= 25

    return score


def search_products(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    exact = exact_name_match(query, df)
    if not exact.empty:
        return exact.head(top_k).copy()

    tokens = expand_query_tokens(query)
    budget = extract_budget(query)

    temp = df.copy()
    temp = temp[temp.apply(is_valid_product_row, axis=1)].copy()
    if temp.empty:
        return pd.DataFrame()

    temp["score"] = temp.apply(lambda row: score_product(row, query, tokens, budget), axis=1)

    qn = normalize_text(query)

    if "canna" in qn or "canne" in qn:
        filtered = temp[temp["all_text"].str.contains(r"\bcanna\b|\bcanne\b|\brod\b", regex=True, na=False)].copy()
        if not filtered.empty:
            temp = filtered

    if "mulinello" in qn or "mulinelli" in qn:
        filtered = temp[temp["all_text"].str.contains(r"\bmulinello\b|\bmulinelli\b|\breel\b", regex=True, na=False)].copy()
        if not filtered.empty:
            temp = filtered

    if budget is not None:
        under = temp[(temp["price_num"].notna()) & (temp["price_num"] <= budget * 1.10)].copy()
        if not under.empty:
            temp = under

    temp = temp.sort_values(by="score", ascending=False)
    temp = temp[temp["score"] > 8]

    return temp.head(top_k).copy()


def format_product_cards(results: pd.DataFrame, show_links: bool = False) -> str:
    parts = []
    for _, row in results.iterrows():
        txt = f"**{row.get('name', '')}**\n"
        if str(row.get("price", "")).strip():
            txt += f"Prezzo: {row.get('price', '')}\n"
        if str(row.get("category", "")).strip():
            txt += f"Categoria: {row.get('category', '')}\n"
        if str(row.get("brand", "")).strip():
            txt += f"Marca: {row.get('brand', '')}\n"
        desc = short_text(row.get("description", ""), 180)
        if desc:
            txt += f"{desc}\n"
        if show_links and str(row.get("url", "")).strip():
            txt += f"[Apri prodotto]({row.get('url', '')})\n"
        parts.append(txt.strip())
    return "\n\n---\n\n".join(parts)


# =========================================================
# OPENAI
# =========================================================
def ask_openai(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Al momento non riesco a generare una risposta corretta. Errore: {str(e)}"


# =========================================================
# MEMORY HELPERS
# =========================================================
def check_manual_faq(query: str) -> Optional[str]:
    memory = load_memory()
    qn = normalize_text(query)

    for item in memory.get("manual_faq", []):
        qq = normalize_text(item.get("question", ""))
        if qq and (qq == qn or qq in qn or qn in qq):
            return item.get("answer", "").strip()

    return None


def memory_rules_text() -> str:
    memory = load_memory()
    lines = []

    rules = memory.get("global_rules", [])
    if rules:
        lines.append("REGOLE GLOBALI:")
        for r in rules:
            lines.append(f"- {r}")

    bads = memory.get("bad_examples", [])[-10:]
    if bads:
        lines.append("\nERRORI DA NON RIPETERE:")
        for b in bads:
            lines.append(f"- Richiesta: {b.get('user_request', '')}")
            lines.append(f"  Errore: {b.get('wrong_reply', '')}")
            lines.append(f"  Lezione: {b.get('lesson', '')}")

    link_rules = memory.get("link_rules", [])
    if link_rules:
        lines.append("\nREGOLE LINK:")
        for r in link_rules:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()


# =========================================================
# RESPONSES
# =========================================================
def greeting_response() -> str:
    return (
        "Ciao! Sono l’assistente MGFishing 🎣\n\n"
        "Posso aiutarti con:\n"
        "- consigli sui prodotti del catalogo\n"
        "- link ai prodotti quando disponibili\n"
        "- spedizioni, pagamenti, tracking e info negozio\n"
        "- montature e consigli di pesca"
    )


def product_link_response(query: str, df: pd.DataFrame) -> str:
    results = search_products(query, df, top_k=3)

    if results.empty:
        msg = f"Ciao Emanuele, sto cercando questo prodotto ma non trovo il link preciso: {query}"
        return (
            "Non sono riuscito a trovare con certezza il prodotto esatto nel catalogo.\n\n"
            f"Per verifica rapida scrivi direttamente a {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link(msg)}"
        )

    with_url = results[results["url"].astype(str).str.strip() != ""].copy()

    if with_url.empty:
        msg = f"Ciao Emanuele, sto cercando il link del prodotto: {query}"
        return (
            "Ho trovato dei prodotti pertinenti, ma nel catalogo non vedo un link prodotto disponibile.\n\n"
            + format_product_cards(results.head(3), show_links=False)
            + f"\n\nPer avere il link corretto scrivi a {WHATSAPP_LABEL}:\n\n{whatsapp_link(msg)}"
        )

    if len(with_url) == 1:
        row = with_url.iloc[0]
        return (
            f"Ho trovato il prodotto più pertinente:\n\n"
            f"**{row.get('name', '')}**\n"
            f"[Apri prodotto]({row.get('url', '')})"
        )

    return (
        "Ho trovato questi prodotti pertinenti. Apri quello corretto:\n\n"
        + format_product_cards(with_url.head(3), show_links=True)
    )


def store_info_response(query: str, knowledge: str) -> str:
    manual = check_manual_faq(query)
    if manual:
        return manual

    if not knowledge.strip():
        return (
            f"Per assistenza diretta ti consiglio di contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao Emanuele, avrei bisogno di informazioni sul negozio.')}"
        )

    system_prompt = (
        "Sei l'assistente clienti di MGFishing.\n"
        "Rispondi usando SOLO le informazioni presenti nel testo fornito.\n"
        "Riconosci anche domande formulate in modo diverso ma con lo stesso significato.\n"
        "Se il testo contiene la risposta, rispondi in modo diretto e naturale.\n"
        "Se il testo NON contiene la risposta, invita a contattare Emanuele su WhatsApp al 3494166335.\n"
        "Non inventare nulla."
    )

    user_prompt = (
        f"TESTO KNOWLEDGE:\n{knowledge}\n\n"
        f"DOMANDA UTENTE:\n{query}"
    )

    reply = ask_openai(system_prompt, user_prompt, temperature=0.1)
    return reply


def product_response(query: str, df: pd.DataFrame) -> str:
    manual = check_manual_faq(query)
    if manual:
        return manual

    if df.empty:
        return (
            "Al momento non riesco a leggere il catalogo prodotti.\n\n"
            f"Per un aiuto immediato puoi contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao Emanuele, mi serve un consiglio su un prodotto.')}"
        )

    results = search_products(query, df, top_k=5)

    if results.empty:
        return (
            "Non sono riuscito a trovare nel catalogo prodotti davvero pertinenti alla tua richiesta.\n\n"
            f"Per un consiglio più preciso ti consiglio di contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao Emanuele, sto cercando questo prodotto: {query}')}"
        )

    catalog_context = []
    for _, row in results.iterrows():
        catalog_context.append({
            "nome": row.get("name", ""),
            "prezzo": row.get("price", ""),
            "categoria": row.get("category", ""),
            "marca": row.get("brand", ""),
            "descrizione": short_text(row.get("description", ""), 250),
            "url": row.get("url", "")
        })

    system_prompt = (
        "Sei un assistente esperto di pesca per MGFishing.\n"
        "Devi consigliare SOLO prodotti presenti nel catalogo fornito.\n"
        "Non inventare prodotti, categorie, marchi, prezzi o caratteristiche.\n"
        "Non proporre prodotti con prezzo 0.\n"
        "Non proporre prodotti con categoria Home.\n"
        "Non proporre prodotti poco pertinenti.\n"
        "Se la richiesta parla di canne, non consigliare mulinelli salvo richiesta esplicita.\n"
        "Se la richiesta parla di mulinelli, non consigliare canne salvo richiesta esplicita.\n"
        "Se i risultati non sono perfetti, dichiaralo chiaramente.\n"
        "Risposta chiara, commerciale, concreta.\n"
        "Non citare fonti esterne.\n\n"
        f"{memory_rules_text()}"
    )

    user_prompt = (
        f"Richiesta cliente: {query}\n\n"
        f"Prodotti trovati nel catalogo:\n{json.dumps(catalog_context, ensure_ascii=False, indent=2)}\n\n"
        "Consiglia solo prodotti davvero coerenti. Se uno non è coerente, ignoralo."
    )

    reply = ask_openai(system_prompt, user_prompt, temperature=0.2)

    cards = format_product_cards(results.head(3), show_links=False)
    return f"{reply}\n\n---\n\n{cards}"


def web_advice_response(query: str) -> str:
    manual = check_manual_faq(query)
    if manual:
        return manual

    system_prompt = (
        "Sei un assistente esperto di pesca di MGFishing.\n"
        "Rispondi a domande su montature, tecniche, terminali, inneschi, spot, orari e consigli pratici.\n"
        "Non citare siti esterni o fonti esterne.\n"
        "Se mancano dati come zona, specie o stagione, dillo e dai comunque una linea guida utile.\n\n"
        f"{memory_rules_text()}"
    )
    user_prompt = f"Domanda cliente: {query}"
    return ask_openai(system_prompt, user_prompt, temperature=0.35)


def generic_response(query: str, df: pd.DataFrame, knowledge: str) -> str:
    manual = check_manual_faq(query)
    if manual:
        return manual

    qn = normalize_text(query)

    if contains_any(qn, STORE_INFO_WORDS):
        return store_info_response(query, knowledge)

    maybe_products = search_products(query, df, top_k=3)
    if not maybe_products.empty and contains_any(qn, PRODUCT_HINTS):
        return product_response(query, df)

    return web_advice_response(query)


def generate_response(query: str, training_mode: bool = False) -> str:
    intent = detect_intent(query)

    if intent == "greeting":
        return greeting_response()
    if intent == "product_link":
        return product_link_response(query, catalog_df)
    if intent == "store_info":
        return store_info_response(query, knowledge_text)
    if intent == "product_advice":
        return product_response(query, catalog_df)
    if intent == "web_advice":
        return web_advice_response(query)

    return generic_response(query, catalog_df, knowledge_text)


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.2rem;
    }
    .small-note {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎣 MGFishing Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Consigli prodotti, link prodotto, info negozio, montature e supporto training</div>',
    unsafe_allow_html=True
)

ensure_memory_file()

with st.sidebar:
    st.header("Impostazioni")
    training_mode = st.toggle("Training Mode", value=False)
    if training_mode:
        st.markdown("**Modalità training attiva**: le correzioni possono essere salvate.")
    else:
        st.markdown("**Modalità cliente**: il chatbot legge la memoria ma non impara.")

    st.divider()

    st.subheader("Aggiungi Regola")
    new_rule = st.text_area("Nuova regola globale", height=100, key="new_rule")
    if st.button("Salva regola"):
        if training_mode and new_rule.strip():
            add_global_rule(new_rule)
            st.success("Regola salvata in memory.json")
        elif not training_mode:
            st.warning("Attiva prima il Training Mode.")

    st.divider()

    st.subheader("Salva FAQ Manuale")
    faq_q = st.text_input("Domanda", key="faq_q")
    faq_a = st.text_area("Risposta", height=120, key="faq_a")
    if st.button("Salva FAQ"):
        if training_mode and faq_q.strip() and faq_a.strip():
            add_manual_faq(faq_q, faq_a)
            st.success("FAQ salvata in memory.json")
        elif not training_mode:
            st.warning("Attiva prima il Training Mode.")

    st.divider()

    st.subheader("Salva Errore da Non Ripetere")
    wrong_user_request = st.text_input("Richiesta utente errata", key="wrong_user_request")
    wrong_reply = st.text_area("Risposta sbagliata del chatbot", height=120, key="wrong_reply")
    wrong_lesson = st.text_area("Lezione / correzione", height=120, key="wrong_lesson")
    if st.button("Salva errore"):
        if training_mode and wrong_user_request.strip() and wrong_reply.strip() and wrong_lesson.strip():
            add_training_note(wrong_user_request, wrong_reply, wrong_lesson)
            st.success("Errore salvato in memory.json")
        elif not training_mode:
            st.warning("Attiva prima il Training Mode.")

    st.divider()

    if st.button("Ricarica catalogo"):
        load_catalog.clear()
        load_knowledge.clear()
        st.success("Cache svuotata. Ricarica la pagina.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono l’assistente MGFishing 🎣\n\n"
                "Posso aiutarti con consigli sui prodotti, link prodotto, info su pagamenti e spedizioni, "
                "montature e supporto pesca."
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
            answer = generate_response(user_query, training_mode=training_mode)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
