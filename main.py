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

GOAL_FIELDS = ["make", "model", "price_max", "year_min", "mileage_max", "condition", "color"]

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "state": "GREETING",
            "intent": "sales_inquiry",
            # target fields
            "make": None,
            "model": None,
            "price_max": None,
            "year_min": None,
            "mileage_max": None,
            "condition": None,
            "color": None,
            # runtime
            "vehicle_hits": [],
            "salesperson": None,
            "proposed": [],
            "chosen": None,
            "phone": None,
            "email": None,
            "no_new_field_turns": 0,   # turns since a new field was captured
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
    """Load CSV inventory from Google Sheets (or any CSV URL). Cached for 60s."""
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

# --- normalization helpers ---
MAKE_SYNONYMS = {
    "chevy": "chevrolet",
    "vw": "volkswagen",
    "merc": "mercedes-benz",
    "benz": "mercedes-benz",
    "bimmer": "bmw",
}
MODEL_SYNONYMS = {
    "civics": "civic",
    "accords": "accord",
    "rav 4": "rav4",
    "rav-4": "rav4",
    "f150": "f-150",
    "model3": "model 3",
    "model-y": "model y",
}
CONDITION_WORDS = {
    "new": "New",
    "brand new": "New",
    "used": "Used",
    "preowned": "Used",
    "pre-owned": "Used",
    "certified": "Certified",
}

COLOR_WORDS = {
    "black":"Black","white":"White","silver":"Silver","gray":"Gray","grey":"Gray","blue":"Blue","red":"Red",
    "green":"Green","yellow":"Yellow","gold":"Gold","orange":"Orange","brown":"Brown","beige":"Beige"
}

def _canon_make(s: str) -> str:
    return MAKE_SYNONYMS.get(s, s)

def _canon_model(s: str) -> str:
    return MODEL_SYNONYMS.get(s, s)

# =================================
# ---------- EXTRACTION ----------
# =================================
PRICE_RE = re.compile(r"(?:under|below|less than|<=?|<|\$)\s*\$?\s*([0-9]{2,6})", re.I)
YEAR_RE  = re.compile(r"(?:from|after|since|>=?)\s*(20\d{2}|19\d{2})|(?:year|model year|my)\s*(20\d{2}|19\d{2})|(?:\b(20\d{2}|19\d{2})\b)", re.I)
MILES_RE = re.compile(r"(?:under|below|less than|<=?|<)\s*([0-9]{2,6})\s*(?:miles?|mi\b)", re.I)

def extract_make_model(text: str, inv: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Forgiving extractor from free text."""
    if not text: return None, None
    inv = inv or load_inventory()
    makes = sorted({(r.get("Make") or "").strip().lower() for r in inv if r.get("Make")})
    models_raw = [(r.get("Model") or "").strip() for r in inv if r.get("Model")]
    models_norm = sorted({m.lower().replace("-", "").strip() for m in models_raw})

    words = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    mk = md = None
    for w in words:
        wn = _canon_model(w.replace("-", ""))
        if not md and wn in models_norm: md = wn
        wm = _canon_make(w)
        if not mk and wm in makes: mk = wm
    return mk, md

def extract_price_max(text: str) -> Optional[int]:
    m = PRICE_RE.search(text or "")
    try:
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None

def extract_year_min(text: str) -> Optional[int]:
    for m in YEAR_RE.finditer(text or ""):
        for g in m.groups():
            if g and g.isdigit():
                y = int(g)
                if 1980 <= y <= 2100:
                    return y
    return None

def extract_mileage_max(text: str) -> Optional[int]:
    m = MILES_RE.search(text or "")
    try:
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None

def extract_condition(text: str) -> Optional[str]:
    t = (text or "").lower()
    for k, v in CONDITION_WORDS.items():
        if k in t: return v
    return None

def extract_color(text: str) -> Optional[str]:
    t = (text or "").lower()
    for k, v in COLOR_WORDS.items():
        if re.search(rf"\b{k}\b", t): return v
    return None

def fill_fields_from_text(s: Dict[str, Any], text: str) -> int:
    """Try to fill any of the 7 fields from this turn. Returns number of newly filled fields."""
    inv = load_inventory()
    before = sum(1 for f in GOAL_FIELDS if s.get(f))
    mk, md = extract_make_model(text, inv)
    price_max = extract_price_max(text)
    year_min = extract_year_min(text)
    miles = extract_mileage_max(text)
    cond = extract_condition(text)
    color = extract_color(text)

    if mk and not s.get("make"): s["make"] = mk
    if md and not s.get("model"): s["model"] = md
    if price_max and not s.get("price_max"): s["price_max"] = price_max
    if year_min and not s.get("year_min"): s["year_min"] = year_min
    if miles and not s.get("mileage_max"): s["mileage_max"] = miles
    if cond and not s.get("condition"): s["condition"] = cond
    if color and not s.get("color"): s["color"] = color

    after = sum(1 for f in GOAL_FIELDS if s.get(f))
    return max(0, after - before)

# =================================
# ---------- SEARCH --------------
# =================================
def _as_int(x: Any) -> Optional[int]:
    try:
        s = str(x).replace(",", "").strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None

def matches_filters(v: Dict[str, Any], s: Dict[str, Any]) -> Tuple[bool, int]:
    """Return (match, score). Score rewards matched fields."""
    score = 0
    # make/model
    if s.get("make"):
        if s["make"] not in (v.get("Make","") or "").lower(): return (False, 0)
        score += 1
    if s.get("model"):
        if s["model"].replace("-", "") not in (v.get("Model","") or "").lower().replace("-", ""): return (False, 0)
        score += 1
    # price
    if s.get("price_max"):
        price_i = _as_int(v.get("Price"))
        if price_i is None or price_i > int(s["price_max"]): return (False, 0)
        score += 1
    # year
    if s.get("year_min"):
        vy = _as_int(v.get("Year"))
        if vy is None or vy < int(s["year_min"]): return (False, 0)
        score += 1
    # mileage
    if s.get("mileage_max"):
        miles_i = _as_int(v.get("Mileage"))
        if miles_i is None or miles_i > int(s["mileage_max"]): return (False, 0)
        score += 1
    # condition
    if s.get("condition"):
        if (v.get("Condition","") or "").lower() != s["condition"].lower(): return (False, 0)
        score += 1
    # color (loose contains)
    if s.get("color"):
        if s["color"].lower() not in (v.get("Color","") or "").lower(): return (False, 0)
        score += 1
    return (True, score)

def search_inventory(s: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    inv = load_inventory()
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for v in inv:
        ok, sc = matches_filters(v, s)
        if ok:
            scored.append((sc, v))
    # If nothing matches strict filters, relax gradually (drop least important fields)
    if not scored:
        relaxed_order = ["color", "condition", "mileage_max", "year_min", "price_max", "model", "make"]
        tmp = dict(s)
        for f in relaxed_order:
            if tmp.get(f):
                tmp[f] = None
                scored = []
                for v in inv:
                    ok, sc = matches_filters(v, tmp)
                    if ok:
                        scored.append((sc, v))
                if scored:  # stop at first level that yields hits
                    s["_relaxed"] = f
                    break
    scored.sort(key=lambda t: (-t[0], (t[1].get("Price") or "")))
    return [v for _, v in scored[:limit]]

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
        if not (e <= bs or s >= be): return False
    return True

def _parse_hours():
    try:
        return json.loads(HOURS_JSON) if HOURS_JSON else {}
    except Exception:
        return {}

def propose_slots(sales_cal_id: Optional[str]) -> List[str]:
    tz = ZoneInfo(TIMEZONE)
    service = _google_service()
    hours = _parse_hours()
    if service and DEALERSHIP_CALENDAR_ID and sales_cal_id:
        now = datetime.now(tz); horizon = now + timedelta(days=5)
        fb = _freebusy(service, [DEALERSHIP_CALENDAR_ID, sales_cal_id], now, horizon, TIMEZONE)
        dealer_busy = fb.get(DEALERSHIP_CALENDAR_ID, []); sales_busy = fb.get(sales_cal_id, [])
        slots = []; d = now
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
        if slots: return slots
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
        "end":   {"dateTime": end.astimezone(ZoneInfo("UTC")).isoformat(),   "timeZone": TIMEZONE},
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
        r"\b(test[-\s]?drive|book|appointment|schedule)\b",
        r"\b(look(ing)? for|interested in|inventory|available|in stock|have any|show me)\b",
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
    return "sales_inquiry"

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

# --- salespeople utils ---
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

# =================================
# ---------- VOICE FLOW ----------
# =================================
@app.post("/webhooks/voice")
async def webhooks_voice(turn: VoiceTurn):
    print("TURN:", {"session_id": turn.session_id, "transcript": turn.transcript, "caller_phone": turn.caller_phone})

    s = get_session(turn.session_id)
    text = (turn.transcript or "").strip()

    # classify once, but keep sales as default
    if s.get("intent") is None:
        s["intent"] = classify_intent(text)

    # ---- GREETING ----
    if s["state"] == "GREETING":
        s["state"] = "COLLECT"
        return {"reply": f"Thanks for calling {DEALERSHIP_NAME}! Tell me what you have in mind—make, model, budget, year, mileage, condition or even color, and I’ll find the best matches.",
                "handoff": False, "end_call": False}

    # ---- PRESENT (already showed options) ----
    if s["state"] == "PRESENT":
        # If they say "first/second" or name one, jump to slots
        choice = None
        if re.search(r"\bfirst\b", text.lower()):
            choice = 0
        elif re.search(r"\bsecond\b", text.lower()):
            choice = 1
        elif re.search(r"\bthird\b", text.lower()):
            choice = 2
        else:
            # try model+year match
            for i, v in enumerate(s.get("vehicle_hits") or []):
                m = (v.get("Model","") or "").lower()
                y = str(v.get("Year","") or "")
                if m and m in text.lower() or (y and y in text):
                    choice = i; break

        if choice is not None and (s.get("vehicle_hits") and 0 <= choice < len(s["vehicle_hits"])):
            chosen_v = s["vehicle_hits"][choice]
            s["chosen_vehicle"] = chosen_v
            sp = pick_salesperson_by_name(text) or pick_salesperson_round_robin()
            s["salesperson"] = sp
            s["proposed"] = propose_slots(sp.get("calendar_id") if sp else None)
            s["state"] = "SLOTS"
            return {"reply": f"Great choice — {vehicle_blurb(chosen_v)}. I can set up a quick test drive. Times: {', '.join(s['proposed'])}. What works for you?",
                    "handoff": False, "end_call": False}

        # Otherwise keep collecting (maybe they added more filters)
        s["state"] = "COLLECT"

    # ---- COLLECT (natural filling of fields) ----
    if s["state"] == "COLLECT":
        new_count = fill_fields_from_text(s, text)
        if new_count > 0:
            s["no_new_field_turns"] = 0
        else:
            s["no_new_field_turns"] += 1

        filled = [f for f in GOAL_FIELDS if s.get(f)]
        # trigger search if enough signals OR user explicitly asked
        enough = len(filled) >= 4 or re.search(r"\b(show|any|have|available|inventory|do you have)\b", text.lower())

        if enough:
            hits = search_inventory(s, limit=3)
            s["vehicle_hits"] = hits
            if hits:
                s["state"] = "PRESENT"
                top = "; ".join(vehicle_blurb(v) for v in hits)
                extra = ""
                if s.get("_relaxed"):
                    extra = f" (I relaxed '{s['_relaxed']}' to show options)."
                return {"reply": f"Here are some good matches{extra}: {top}. Which one should I set up a test drive for—first, second, or third?",
                        "handoff": False, "end_call": False}
            else:
                # no hits yet—suggest a gentle narrowing
                s["state"] = "COLLECT"
                return {"reply": "I didn’t see anything with those exact filters. Want to adjust the budget, year, or color a bit and I’ll re-check?",
                        "handoff": False, "end_call": False}

        # If not enough yet and we’ve had 2 turns with no new info, ask the next missing field
        if s["no_new_field_turns"] >= 2:
            for f in GOAL_FIELDS:
                if not s.get(f):
                    s["no_new_field_turns"] = 0
                    prompts = {
                        "make": "Do you have a make in mind—like Honda, Toyota, or Chevy?",
                        "model": "Any model you’re leaning toward—Civic, Camry, Accord?",
                        "price_max": "What’s a comfortable budget to stay under?",
                        "year_min": "Is there a model year you’d prefer—from what year and newer?",
                        "mileage_max": "Any mileage cap you want to stay under?",
                        "condition": "New, used, or certified?",
                        "color": "Any color preferences?"
                    }
                    return {"reply": prompts[f], "handoff": False, "end_call": False}

        # Otherwise, acknowledge and keep listening—no forced question
        ack_bits = []
        for key, label in [("make","make"),("model","model"),("price_max","budget"),
                           ("year_min","year"),("mileage_max","mileage"),("condition","condition"),("color","color")]:
            if s.get(key): ack_bits.append(label)
        if ack_bits:
            return {"reply": f"Got it on {', '.join(ack_bits)}. Anything else you want me to factor in—like year, mileage, condition or color?",
                    "handoff": False, "end_call": False}
        else:
            return {"reply": "Tell me anything you care about—make, model, price, year, mileage, condition or color—and I’ll pull matches.",
                    "handoff": False, "end_call": False}

    # ---- SLOTS ----
    if s["state"] == "SLOTS":
        chosen = None
        txt = text.lower()
        # allow “morning/afternoon” mapping
        if "morning" in txt and s["proposed"]: chosen = s["proposed"][0]
        if "afternoon" in txt and len(s["proposed"]) > 1: chosen = s["proposed"][1]
        # ordinal shortcuts
        if re.search(r"\bfirst\b", txt) and s["proposed"]: chosen = s["proposed"][0]
        if re.search(r"\bsecond\b", txt) and len(s["proposed"])>1: chosen = s["proposed"][1]
        # exact include
        for sl in s["proposed"]:
            if sl.lower() in txt: chosen = sl; break

        if not chosen:
            return {"reply": f"Which time works best: {', '.join(s['proposed'])}?",
                    "handoff": False, "end_call": False}

        s["chosen"] = chosen
        s["state"] = "CONTACT"
        return {"reply": "Perfect. What’s the best phone number and email for confirmation?",
                "handoff": False, "end_call": False}

    # ---- CONTACT ----
    if s["state"] == "CONTACT":
        ph, em = extract_contact(text)
        if turn.caller_phone and not s.get("phone"): s["phone"] = turn.caller_phone
        if ph: s["phone"] = ph
        if em: s["email"] = em

        if not (s.get("phone") or s.get("email")):
            return {"reply": "I’ll need a phone or an email to send the confirmation—whichever you prefer.",
                    "handoff": False, "end_call": False}

        sp = s.get("salesperson") or {}
        v  = s.get("chosen_vehicle") or (s.get("vehicle_hits")[0] if s.get("vehicle_hits") else {})
        slot = s.get("chosen") or ""
        title = f"Showroom Appointment — {vehicle_blurb(v)}" if v else "Showroom Appointment"
        desc = (
            f"Customer: {s.get('email','')} / {s.get('phone','')}\n"
            f"Salesperson: {sp.get('name','')} <{sp.get('email','')}>\n"
            f"Vehicle: {json.dumps(v, ensure_ascii=False)}\n"
            f"Booked via {DEALERSHIP_NAME} AI receptionist."
        )
        link = create_calendar_event(sp, s.get("email"), s.get("phone"), slot, title, desc)

        msg = f"{DEALERSHIP_NAME}: Confirmed {slot} — {vehicle_blurb(v)}." if v else f"{DEALERSHIP_NAME}: Confirmed {slot}."
        if s.get("phone"): send_sms(s["phone"], msg)
        if s.get("email"): send_email(s["email"], f"{DEALERSHIP_NAME} appointment confirmed", msg + (f"\nCalendar: {link}" if link else ""))

        outcome = {"slot": slot, "vehicle": v, "salesperson": sp, "phone": s.get("phone"), "email": s.get("email"), "calendar_link": link}
        print("[BOOKED]", json.dumps(outcome, ensure_ascii=False))
        post_analytics({"type": "booking", "payload": outcome})

        s["state"] = "CLOSE"
        return {"reply": "Booked! I’ll send your confirmation. Anything else I can help with?",
                "handoff": False, "end_call": False}

    # ---- FALLTHROUGH / CLOSE ----
    return {"reply": "Thanks for calling. Have a great day!",
            "handoff": False, "end_call": True}
