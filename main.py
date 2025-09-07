import os
import re
import time
import json
import difflib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# ---------- ENV ----------
# =========================
INVENTORY_CSV_URL = os.getenv("INVENTORY_CSV_URL", "").strip()
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
DEALERSHIP_NAME = os.getenv("DEALERSHIP_NAME", "Your Dealership")
DEALERSHIP_ADDRESS = os.getenv("DEALERSHIP_ADDRESS", "").strip()  # optional

DEALERSHIP_CALENDAR_ID = os.getenv("DEALERSHIP_CALENDAR_ID", "").strip()
SALESPEOPLE_JSON = os.getenv("SALESPEOPLE_JSON", "[]")
APPT_DEFAULT_SLOTS_JSON = os.getenv("APPT_DEFAULT_SLOTS_JSON", '["Tomorrow 10 AM", "Thursday 2 PM"]')
HOURS_JSON = os.getenv("HOURS_JSON", "")  # e.g. {"mon":[9,18],...} (optional)

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
FROM_EMAIL = os.getenv("FROM_EMAIL", "").strip()
ANALYTICS_WEBHOOK_URL = os.getenv("ANALYTICS_WEBHOOK_URL", "").strip()

VERSION = "0.3"

# ============================
# ---------- STATE ----------
# ============================
SESSIONS: Dict[str, Dict[str, Any]] = {}
_round_robin_idx = 0

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "state": "GREETING",
            "intent": None,
            "make": None,
            "model": None,
            "vehicle": None,
            "salesperson": None,
            "proposed": [],
            "chosen": None,
            "phone": None,
            "email": None,
            "created_at": time.time(),
        }
    return SESSIONS[session_id]

# =================================
# ---------- INVENTORY -----------
# =================================
_inv_cache: List[Dict[str, Any]] = []
_inv_age = 0

def _split_csv(line: str) -> List[str]:
    out, cur, q = [], "", False
    for ch in line:
        if ch == '"' and not q:
            q = True; continue
        if ch == '"' and q:
            q = False; continue
        if ch == ',' and not q:
            out.append(cur); cur = ""; continue
        cur += ch
    out.append(cur)
    return out

def load_inventory() -> List[Dict[str, Any]]:
    """
    Load CSV inventory from Google Sheets (or any CSV URL).
    Cached for 60s to keep responses fast.
    """
    global _inv_cache, _inv_age
    if _inv_cache and time.time() - _inv_age < 60:
        return _inv_cache
    rows: List[Dict[str, Any]] = []
    if not INVENTORY_CSV_URL:
        _inv_cache = rows; _inv_age = time.time(); return rows
    try:
        with httpx.Client(timeout=12.0) as c:
            r = c.get(INVENTORY_CSV_URL)
            r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        if not lines:
            _inv_cache = []; _inv_age = time.time(); return []
        headers = [h.strip() for h in _split_csv(lines[0])]
        for ln in lines[1:]:
            cells = [x.strip() for x in _split_csv(ln)]
            rows.append({headers[i]: (cells[i] if i < len(cells) else "") for i in range(len(headers))})
    except Exception as e:
        print("ERROR fetching inventory CSV:", e)
    _inv_cache = rows; _inv_age = time.time(); return rows

# Canonicalization (extend over time)
MAKE_SYNONYMS = {
    "chevy": "chevrolet",
    "vw": "volkswagen",
    "merc": "mercedes-benz",
    "benz": "mercedes-benz",
    "bimmer": "bmw",
    "tesla": "tesla",
}

MODEL_SYNONYMS = {
    "civics": "civic",
    "accords": "accord",
    "rav 4": "rav4",
    "rav-4": "rav4",
    "rav4s": "rav4",
    "f150": "f-150",
    "f-150": "f-150",
    "model3": "model 3",
    "model-y": "model y",
}

def _canon_make(s: str) -> str:
    return MAKE_SYNONYMS.get(s, s)

def _canon_model(s: str) -> str:
    return MODEL_SYNONYMS.get(s, s)

def _closest(token: str, options: List[str], cutoff: float = 0.84) -> Optional[str]:
    match = difflib.get_close_matches(token, options, n=1, cutoff=cutoff)
    return match[0] if match else None

def extract_make_model(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Forgiving extractor: handles plurals, hyphens, common synonyms, and mild typos.
    """
    if not text:
        return None, None
    inv = load_inventory()
    makes = sorted({(r.get("Make") or "").strip().lower() for r in inv if r.get("Make")})
    models_raw = [(r.get("Model") or "").strip() for r in inv if r.get("Model")]
    models_norm = sorted({m.lower().replace("-", "").strip() for m in models_raw})

    words = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    mk = md = None

    for w in words:
        wn = w.replace("-", "")
        candidates = {wn}
        if wn.endswith("ies"): candidates.add(wn[:-3] + "y")
        if wn.endswith("es"):  candidates.add(wn[:-2])
        if wn.endswith("s"):   candidates.add(wn[:-1])
        can_makes = {_canon_make(c) for c in candidates}
        can_models = {_canon_model(c) for c in candidates}

        if not mk and any(c in makes for c in can_makes):
            mk = next(c for c in can_makes if c in makes)

        if not md and any(c.replace("-", "") in models_norm for c in can_models):
            md = next(c for c in can_models if c.replace("-", "") in models_norm)

    # fuzzy fallback for model if still missing
    if not md:
        for w in words:
            wn = _canon_model(w.replace("-", ""))
            guess = _closest(wn, models_norm, cutoff=0.82)
            if guess:
                md = guess
                break

    # fuzzy fallback for make if still missing
    if not mk:
        for w in words:
            w2 = _canon_make(w)
            guess = _closest(w2, makes, cutoff=0.82)
            if guess:
                mk = guess
                break

    return mk, md

# ---------- Additional lightweight filters ----------
BODY_STYLES = {"suv","sedan","truck","coupe","hatchback","van","minivan","convertible","crossover"}
COLOR_WORDS = {"black","white","silver","gray","grey","red","blue","green","brown","beige","gold","yellow","orange"}

def parse_budget(text: str) -> Optional[int]:
    """
    Extract a rough max budget from phrases like:
    - 'under 25k', 'below 30,000', '$22,500', 'around 20k'
    Returns an integer dollar amount or None.
    """
    t = text.lower().replace(",", "")
    # explicit $amount
    m = re.search(r"\$?\s*(\d{4,6})\b", t)
    if m:
        val = int(m.group(1))
        # treat 20000..200000 as dollars; 2000..9999 could be year, allow but keep
        if 10000 <= val <= 200000:
            return val
    # 25k style
    m = re.search(r"(\d{2,3})\s*k\b", t)
    if m:
        return int(m.group(1)) * 1000
    # under/below/around X
    m = re.search(r"(under|below|less than|around|about)\s+(\d{2,3})\s*k", t)
    if m:
        return int(m.group(2)) * 1000
    return None

def parse_year(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if m:
        return m.group(1)
    return None

def parse_body_style(text: str) -> Optional[str]:
    t = text.lower()
    for w in BODY_STYLES:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w
    return None

def parse_color(text: str) -> Optional[str]:
    t = text.lower()
    for c in COLOR_WORDS:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c
    return None

def parse_filters(text: str) -> Dict[str, Any]:
    mk, md = extract_make_model(text)
    year = parse_year(text)
    body = parse_body_style(text)
    color = parse_color(text)
    budget = parse_budget(text)
    return {"make": mk, "model": md, "year": year, "body": body, "color": color, "budget": budget}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _price_to_int(s: str) -> Optional[int]:
    if not s: return None
    try:
        s2 = re.sub(r"[^\d]", "", s)
        return int(s2) if s2 else None
    except Exception:
        return None

def search_inventory(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return vehicles matching any provided filters.
    We apply AND across provided fields; each field uses a tolerant contains match.
    """
    inv = load_inventory()
    mk = (filters.get("make") or "").lower()
    md = (filters.get("model") or "").lower().replace("-", "")
    yr = (filters.get("year") or "").lower()
    body = (filters.get("body") or "").lower()
    color = (filters.get("color") or "").lower()
    budget = filters.get("budget")

    results: List[Dict[str, Any]] = []
    for r in inv:
        rmk = _norm(r.get("Make"))
        rmd = _norm(r.get("Model")).replace("-", "")
        ryr = _norm(str(r.get("Year")))
        rbody = _norm(r.get("Body") or r.get("Type") or r.get("Category"))
        rcolor = _norm(r.get("Color"))
        rprice = _price_to_int(str(r.get("Price")))

        # AND across provided filters
        if mk and mk not in rmk: continue
        if md and md not in rmd: continue
        if yr and yr not in ryr: continue
        if body and body not in rbody: continue
        if color and color not in rcolor: continue
        if budget and rprice and rprice > budget: continue

        results.append(r)
    return results

def vehicle_blurb(v: Dict[str, Any]) -> str:
    year = str(v.get("Year", "")).strip()
    make = v.get("Make", "").strip()
    model = v.get("Model", "").strip()
    color = v.get("Color", "").strip()
    cond  = v.get("Condition", "").strip()
    price = v.get("Price", "").strip()
    bits = []
    if year and model: bits.append(f"{year} {make} {model}")
    if color: bits.append(color)
    if cond:  bits.append(cond)
    if price: bits.append(f"${price}")
    return " · ".join(bits)

def compose_inventory_reply(matches: List[Dict[str, Any]], filters: Dict[str, Any]) -> str:
    """
    Natural one-liner that answers the user's question first.
    """
    if not matches:
        want = []
        if filters.get("make"): want.append(filters["make"].title())
        if filters.get("model"): want.append(filters["model"].title())
        if filters.get("body"): want.append(filters["body"])
        if filters.get("year"): want.append(filters["year"])
        if filters.get("color"): want.append(filters["color"])
        if filters.get("budget"): want.append(f"under ${filters['budget']:,}")
        wish = " ".join(want) if want else "that exact vehicle"
        return f"I’m not seeing {wish} right now. We can look at close alternatives if you’d like."

    top = matches[:3]
    listed = ", ".join(vehicle_blurb(v) for v in top)
    extra = len(matches) - len(top)
    suffix = f" (plus {extra} more)" if extra > 0 else ""
    return f"Yes — we have {listed}{suffix}."

# =================================
# ---------- CALENDAR ------------
# =================================
def _google_service():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        return None
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_service_account_info(
            json.loads(GOOGLE_SERVICE_ACCOUNT_JSON),
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        print("Google Calendar init error:", e)
        return None

def _freebusy(service, cal_ids: List[str], start, end, tz):
    try:
        body = {
            "timeMin": start.astimezone(ZoneInfo("UTC")).isoformat(),
            "timeMax": end.astimezone(ZoneInfo("UTC")).isoformat(),
            "timeZone": tz,
            "items": [{"id": cid} for cid in cal_ids]
        }
        fb = service.freebusy().query(body=body).execute()
        return {k: v.get("busy", []) for k, v in fb.get("calendars", {}).items()}
    except Exception as e:
        print("freebusy error:", e)
        return {cid: [] for cid in cal_ids}

def _is_free(start, end, busy):
    s = start.astimezone(ZoneInfo("UTC"))
    e = end.astimezone(ZoneInfo("UTC"))
    for b in busy:
        bs = datetime.fromisoformat(b["start"].replace("Z", "+00:00"))
        be = datetime.fromisoformat(b["end"].replace("Z", "+00:00"))
        if not (e <= bs or s >= be):
            return False
    return True

def _parse_hours():
    try:
        return json.loads(HOURS_JSON) if HOURS_JSON else {}
    except Exception:
        return {}

def propose_slots(sales_cal_id: Optional[str]) -> List[str]:
    """
    Propose up to 2 slots within 5 days.
    If Google Calendar is configured, check both dealership and salesperson calendars.
    Otherwise fall back to APPT_DEFAULT_SLOTS_JSON.
    """
    tz = ZoneInfo(TIMEZONE)
    service = _google_service()
    hours = _parse_hours()
    if service and DEALERSHIP_CALENDAR_ID and sales_cal_id:
        now = datetime.now(tz)
        horizon = now + timedelta(days=5)
        fb = _freebusy(service, [DEALERSHIP_CALENDAR_ID, sales_cal_id], now, horizon, TIMEZONE)
        dealer_busy = fb.get(DEALERSHIP_CALENDAR_ID, [])
        sales_busy = fb.get(sales_cal_id, [])
        slots = []
        d = now
        while d < horizon and len(slots) < 2:
            wd = ["mon","tue","wed","thu","fri","sat","sun"][d.weekday()]
            start_hour, end_hour = (hours.get(wd, [9, 18]) if isinstance(hours.get(wd, []), list) else [9, 18])
            for hour in [start_hour, max(start_hour+2, 11), max(start_hour+4, 13), min(end_hour-3, 15)]:
                s = d.replace(hour=hour, minute=0, second=0, microsecond=0)
                e = s + timedelta(minutes=30)
                if s < now: continue
                if _is_free(s, e, dealer_busy) and _is_free(s, e, sales_busy):
                    slots.append(s.strftime("%A %I:%M %p"))
                    if len(slots) == 2: break
            d = (d + timedelta(days=1)).replace(hour=start_hour, minute=0)
        if slots:
            return slots
    try:
        return json.loads(APPT_DEFAULT_SLOTS_JSON)[:2]
    except Exception:
        return ["Tomorrow 10 AM", "Thursday 2 PM"]

def create_calendar_event(sp, customer_email, customer_phone, start_str, title, description):
    service = _google_service()
    if not (service and DEALERSHIP_CALENDAR_ID and sp and sp.get("calendar_id")):
        print("[BOOK] skipping calendar (not configured)")
        return None
    tz = ZoneInfo(TIMEZONE)
    try:
        now = datetime.now(tz)
        label = (start_str or "").lower()
        if "tomorrow" in label:
            base = now + timedelta(days=1)
            hour = 10 if "10" in label else (14 if "2" in label else 10)
            start = base.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            wd_names = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            target = next((i for i,n in enumerate(wd_names) if n in label), None)
            d = now
            for _ in range(8):
                if target is None or d.weekday() == target:
                    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", label)
                    hh = int(m.group(1)) if m else 10
                    mm = int(m.group(2)) if (m and m.group(2)) else 0
                    ap = (m.group(3) if m else "am").lower()
                    if ap == "pm" and hh < 12: hh += 12
                    start = d.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if start > now: break
                d = d + timedelta(days=1)
        end = start + timedelta(minutes=30)
    except Exception as e:
        print("parse slot error:", e)
        return None

    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start.astimezone(ZoneInfo("UTC")).isoformat(), "timeZone": TIMEZONE},
        "end":   {"dateTime": end.astimezone(ZoneInfo("UTC")).isoformat(), "timeZone": TIMEZONE},
        "attendees": [{"email": sp.get("email")}] + ([{"email": customer_email}] if customer_email else []),
    }
    try:
        created = service.events().insert(
            calendarId=DEALERSHIP_CALENDAR_ID, body=event, sendUpdates="all"
        ).execute()
        return created.get("htmlLink")
    except Exception as e:
        print("calendar create error:", e)
        return None

# =================================
# ---------- NOTIFY --------------
# =================================
def send_sms(to: str, body: str):
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and to):
        return
    try:
        from twilio.rest import Client
        Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
            to=to, from_=TWILIO_FROM_NUMBER, body=body
        )
    except Exception as e:
        print("twilio error:", e)

def send_email(to: str, subject: str, text: str):
    if not (SENDGRID_API_KEY and FROM_EMAIL and to):
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        msg = Mail(from_email=FROM_EMAIL, to_emails=to, subject=subject, plain_text_content=text)
        SendGridAPIClient(SENDGRID_API_KEY).send(msg)
    except Exception as e:
        print("sendgrid error:", e)

def post_analytics(payload: Dict[str, Any]):
    if not ANALYTICS_WEBHOOK_URL:
        return
    try:
        safe = json.loads(json.dumps(payload, ensure_ascii=False))
        # light PII scrubbing for analytics
        if "payload" in safe and isinstance(safe["payload"], dict):
            p = safe["payload"]
            if "phone" in p: p["phone"] = "***REDACTED***"
            if "email" in p: p["email"] = "***REDACTED***"
        with httpx.Client(timeout=6.0) as c:
            c.post(ANALYTICS_WEBHOOK_URL, json=safe)
    except Exception as e:
        print("analytics post error:", e)

# =================================
# ---------- HELPERS -------------
# =================================
def extract_contact(text: str) -> Tuple[Optional[str], Optional[str]]:
    phone = email = None
    if text:
        m = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text)
        if m:
            phone = re.sub(r"[^\d+]", "", m.group(1))
            if len(phone) < 10: phone = None
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if m:
            email = m.group(0)
    return phone, email

# =================================
# ---------- INTENTS -------------
# =================================
INTENTS: Dict[str, List[str]] = {
    "sales_inquiry": [
        r"\b(test[-\s]?drive|book|appointment)\b",
        r"\b(look(ing)? for|interested in|inventory|available|in stock|have any|got any)\b",
        r"\b(civic|accord|camry|corolla|rav[\s-]?4|f[-\s]?150|model\s?[3syx])\b",
        r"\b(suv|sedan|truck|hatchback|coupe)\b"
    ],
    "service_request": [
        r"\b(service|repair|maintenance|oil change|check engine|brake|tire|diagnostic)\b"
    ],
    "hours_location": [
        r"\b(hours?|open|close|closing|address|location|directions|where|time)\b"
    ],
    "smalltalk": [
        r"\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b"
    ],
}

def classify_intent(text: str) -> str:
    t = (text or "").lower()
    for name, patterns in INTENTS.items():
        for p in patterns:
            if re.search(p, t):
                return name
    return "sales_inquiry"  # default path

# =================================
# ---------- API -----------------
# =================================
app = FastAPI(title="DealerBrain", version=VERSION)

@app.get("/health")
def health():
    try:
        inv_ready = bool(INVENTORY_CSV_URL)
        calendar_ready = bool(GOOGLE_SERVICE_ACCOUNT_JSON and DEALERSHIP_CALENDAR_ID and load_salespeople())
        sms_ready = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER)
        email_ready = bool(SENDGRID_API_KEY and FROM_EMAIL)
        return {
            "ok": True,
            "version": VERSION,
            "sheet": inv_ready,
            "calendar_ready": calendar_ready,
            "sms_ready": sms_ready,
            "email_ready": email_ready,
            "timezone": TIMEZONE,
            "dealer": DEALERSHIP_NAME,
            "now": datetime.now(ZoneInfo(TIMEZONE)).isoformat()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

class VoiceTurn(BaseModel):
    session_id: str
    transcript: Optional[str] = ""
    caller_phone: Optional[str] = None

# SALESPEOPLE utils
def load_salespeople() -> List[Dict[str, Any]]:
    try:
        return [a for a in json.loads(SALESPEOPLE_JSON or "[]") if a.get("name")]
    except Exception:
        return []

def pick_salesperson_by_name(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").lower()
    for sp in load_salespeople():
        if sp.get("name","").lower() in t:
            return sp
    return None

def pick_salesperson_round_robin() -> Optional[Dict[str, Any]]:
    global _round_robin_idx
    sps = load_salespeople()
    if not sps: return None
    sp = sps[_round_robin_idx % len(sps)]
    _round_robin_idx += 1
    return sp

# ---------- Affirmation helpers ----------
YES_WORDS = {"yes","yeah","yep","sure","sounds good","ok","okay","let's do it","book","schedule","set it up","come by","come in","test drive"}
NO_WORDS  = {"no","not now","maybe later","just looking","browsing"}

def _wants_booking(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in YES_WORDS) or "test drive" in t or "see it" in t

def _declines_booking(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in NO_WORDS)

# =================================
# ---------- VOICE WEBHOOK -------
# =================================
@app.post("/webhooks/voice")
async def webhooks_voice(turn: VoiceTurn):
    # Debug trace for observability
    print("TURN:", {"session_id": turn.session_id, "transcript": turn.transcript, "caller_phone": turn.caller_phone})

    s = get_session(turn.session_id)
    text = (turn.transcript or "").strip()
    if not s.get("intent"):
        s["intent"] = classify_intent(text)

    # -------- GREETING / ROUTER --------
    if s["state"] == "GREETING":
        intent = s["intent"] or classify_intent(text)
        if intent == "service_request":
            s["state"] = "SERVICE_GREETING"
            post_analytics({"type": "intent", "payload": {"intent": "service_request"}})
            return {"reply": "Happy to help with service. What vehicle and issue are you experiencing?", "handoff": False, "end_call": False}
        if intent == "hours_location":
            s["state"] = "INFO"
            addr = f" Our address is {DEALERSHIP_ADDRESS}." if DEALERSHIP_ADDRESS else ""
            return {"reply": f"We’re open most days 9 AM to 6 PM.{addr} Would you like to book a visit?", "handoff": False, "end_call": False}
        # smalltalk & default → sales
        s["state"] = "QUALIFY"
        return {"reply": f"Thanks for calling {DEALERSHIP_NAME}! Which car are you interested in? You can say make, model, or even a color/budget.", "handoff": False, "end_call": False}

    # -------- SALES FLOW (Flexible) --------
    if s["state"] == "QUALIFY":
        # Merge new cues with prior context
        filters = parse_filters(text)
        if filters.get("make"):  s["make"]  = filters["make"]
        if filters.get("model"): s["model"] = filters["model"]

        # If we have no filters at all, gently guide with popular options
        if not any(filters.values()) and not (s.get("make") or s.get("model")):
            inv = load_inventory()
            seen_makes, top_makes = set(), []
            seen_models, top_models = set(), []
            for r in inv:
                mk = (r.get("Make") or "").strip()
                md = (r.get("Model") or "").strip()
                if mk and mk.lower() not in seen_makes:
                    seen_makes.add(mk.lower()); top_makes.append(mk)
                if md and md.lower() not in seen_models:
                    seen_models.add(md.lower()); top_models.append(md)
                if len(top_makes) >= 3 and len(top_models) >= 3: break
            prompt = "Any particular make or model you had in mind"
            examples = f"— for example: {', '.join(top_models[:3])} or {', '.join(top_makes[:3])}"
            return {"reply": f"Got it. {prompt}{examples}?", "handoff": False, "end_call": False}

        # Build a combined filter with context memory
        combined = {
            "make":  s.get("make")  or filters.get("make"),
            "model": s.get("model") or filters.get("model"),
            "year":  filters.get("year"),
            "body":  filters.get("body"),
            "color": filters.get("color"),
            "budget":filters.get("budget"),
        }

        matches = search_inventory(combined)
        reply = compose_inventory_reply(matches, combined)

        # If we found something, move toward booking naturally
        if matches:
            # Store the "primary" vehicle for downstream (first best match)
            s["vehicle"] = matches[0]
            sp = pick_salesperson_by_name(text) or pick_salesperson_round_robin()
            s["salesperson"] = sp

            # If caller explicitly wants to come in, jump straight to SLOTS
            if _wants_booking(text):
                s["proposed"] = propose_slots(sp.get("calendar_id") if sp else None)
                s["state"] = "SLOTS"
                return {"reply": f"{reply} I can set up a quick visit. {', '.join(s['proposed'])} work for you?",
                        "handoff": False, "end_call": False}

            # Otherwise answer first, then nudge
            s["proposed"] = propose_slots(sp.get("calendar_id") if sp else None)
            nudge = "Want to pop in for a quick look or test drive?"
            if s["proposed"]:
                nudge = f"Want to pop in for a quick look or test drive? I have {', '.join(s['proposed'])}."
            return {"reply": f"{reply} {nudge}", "handoff": False, "end_call": False}

        # No matches → suggest close alternatives by model
        inv = load_inventory()
        models_norm = sorted({(r.get("Model") or "").strip().lower().replace("-", "") for r in inv if r.get("Model")})
        wanted = (combined.get("model") or "") or (combined.get("body") or "")
        alts = difflib.get_close_matches(wanted, models_norm, n=2, cutoff=0.6) if wanted else []
        alt_text = f" Maybe {', '.join(alts)}?" if alts else ""
        post_analytics({"type": "inventory_miss", "payload": {"query": combined}})
        return {"reply": f"{reply}{alt_text} Tell me a must-have—make, model, body style, color, or budget—and I’ll narrow it down.",
                "handoff": False, "end_call": False}

    # -------- SLOT SELECTION --------
    if s["state"] == "SLOTS":
        # Allow direct selection or generic yes
        if _wants_booking(text) and not s.get("proposed"):
            sp = s.get("salesperson") or pick_salesperson_round_robin()
            s["salesperson"] = sp
            s["proposed"] = propose_slots(sp.get("calendar_id") if sp else None)

        chosen = None
        for sl in (s.get("proposed") or []):
            if sl and sl.lower() in text.lower():
                chosen = sl; break
        if not chosen:
            if re.search(r"\bfirst\b", text.lower()) and s.get("proposed"):
                chosen = s["proposed"][0]
            elif re.search(r"\bsecond\b", text.lower()) and s.get("proposed") and len(s["proposed"]) > 1:
                chosen = s["proposed"][1]

        if not chosen:
            # If user declines, drop back to QUALIFY without pressure
            if _declines_booking(text):
                s["state"] = "QUALIFY"
                return {"reply": "No problem. What else can I look up for you—another model, color, or budget?",
                        "handoff": False, "end_call": False}
            return {"reply": f"Which time works best: {', '.join(s.get('proposed') or [])}?",
                    "handoff": False, "end_call": False}

        s["chosen"] = chosen
        s["state"] = "CONTACT"
        return {"reply": "Great—what’s the best phone number and email for confirmation?",
                "handoff": False, "end_call": False}

    # -------- CONTACT / CONFIRM --------
    if s["state"] == "CONTACT":
        ph, em = extract_contact(text)
        if turn.caller_phone and not s["phone"]: s["phone"] = turn.caller_phone
        if ph: s["phone"] = ph
        if em: s["email"] = em
        if not (s.get("phone") and s.get("email")):
            return {"reply": "I’ll need both a phone number and an email to confirm. Can you share both?",
                    "handoff": False, "end_call": False}

        sp = s.get("salesperson") or {}
        v  = s.get("vehicle") or {}
        slot = s.get("chosen") or ""
        title = f"Showroom Appointment — {vehicle_blurb(v)}"
        desc = (
            f"Customer: {s.get('email')} / {s.get('phone')}\n"
            f"Salesperson: {sp.get('name','')} <{sp.get('email','')}>\n"
            f"Vehicle: {json.dumps(v, ensure_ascii=False)}\n"
            f"Booked via {DEALERSHIP_NAME} AI receptionist."
        )
        link = create_calendar_event(sp, s.get("email"), s.get("phone"), slot, title, desc)

        msg = f"{DEALERSHIP_NAME}: Confirmed {slot} with {sp.get('name','our advisor')} — {vehicle_blurb(v)}."
        send_sms(s.get("phone"), msg)
        send_email(s.get("email"), f"{DEALERSHIP_NAME} appointment confirmed", msg + (f"\nCalendar: {link}" if link else ""))

        outcome = {
            "slot": slot,
            "vehicle": v,
            "salesperson": sp,
            "phone": s.get("phone"),
            "email": s.get("email"),
            "calendar_link": link
        }
        print("[BOOKED]", json.dumps(outcome, ensure_ascii=False))
        post_analytics({"type": "booking", "payload": outcome})

        s["state"] = "CLOSE"
        return {"reply": "Booked! I’ll send your confirmation shortly. Anything else I can help with?",
                "handoff": False, "end_call": False}

    # -------- SERVICE (MVP) --------
    if s["state"] == "SERVICE_GREETING":
        if "issue" not in s or not s.get("issue"):
            s["issue"] = text
            return {"reply": "Thanks. I’ll get a service advisor to follow up. What’s the best phone and email?", "handoff": False, "end_call": False}
        ph, em = extract_contact(text)
        if turn.caller_phone and not s.get("phone"): s["phone"] = turn.caller_phone
        if ph: s["phone"] = ph
        if em: s["email"] = em
        if not (s.get("phone") and s.get("email")):
            return {"reply": "I’ll need both a phone number and an email for the service advisor to contact you.",
                    "handoff": False, "end_call": False}
        post_analytics({"type": "service_lead", "payload": {"issue": s.get("issue"), "phone": s.get("phone"), "email": s.get("email")}})
        s["state"] = "CLOSE"
        return {"reply": "Thanks! Our service team will reach out shortly. Anything else I can help with?",
                "handoff": False, "end_call": False}

    # -------- INFO / SMALLTALK --------
    if s["state"] == "INFO":
        s["state"] = "GREETING"
        return {"reply": "Anything else I can help with today—sales or service?", "handoff": False, "end_call": False}

    # default fall-through / end
    return {"reply": "Thanks for calling. Have a great day!", "handoff": False, "end_call": True}
