import os
import re
import time
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- ENV ----------
INVENTORY_CSV_URL = os.getenv("INVENTORY_CSV_URL", "").strip()
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
DEALERSHIP_NAME = os.getenv("DEALERSHIP_NAME", "Your Dealership")

DEALERSHIP_CALENDAR_ID = os.getenv("DEALERSHIP_CALENDAR_ID", "").strip()
SALESPEOPLE_JSON = os.getenv("SALESPEOPLE_JSON", "[]")
APPT_DEFAULT_SLOTS_JSON = os.getenv("APPT_DEFAULT_SLOTS_JSON", '["Tomorrow 10 AM", "Thursday 2 PM"]')

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
FROM_EMAIL = os.getenv("FROM_EMAIL", "").strip()
ANALYTICS_WEBHOOK_URL = os.getenv("ANALYTICS_WEBHOOK_URL", "").strip()

# ---------- RUNTIME STATE ----------
SESSIONS: Dict[str, Dict[str, Any]] = {}
_round_robin_idx = 0


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "state": "GREETING",
            "make": None,
            "model": None,
            "vehicle": None,
            "salesperson": None,
            "proposed": [],
            "chosen": None,
            "phone": None,
            "email": None
        }
    return SESSIONS[session_id]


# ---------- SHEET LOADING ----------
_inv_cache: List[Dict[str, Any]] = []
_inv_age = 0


def _split_csv(line: str) -> List[str]:
    out, cur, q = [], "", False
    for ch in line:
        if ch == '"' and not q:
            q = True
            continue
        if ch == '"' and q:
            q = False
            continue
        if ch == ',' and not q:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    out.append(cur)
    return out


def load_inventory() -> List[Dict[str, Any]]:
    global _inv_cache, _inv_age
    if _inv_cache and time.time() - _inv_age < 60:
        return _inv_cache
    rows: List[Dict[str, Any]] = []
    if not INVENTORY_CSV_URL:
        _inv_cache = rows
        _inv_age = time.time()
        return rows
    try:
        with httpx.Client(timeout=12.0) as c:
            r = c.get(INVENTORY_CSV_URL)
            r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        headers = [h.strip() for h in _split_csv(lines[0])]
        for ln in lines[1:]:
            cells = [x.strip() for x in _split_csv(ln)]
            rows.append({headers[i]: (cells[i] if i < len(cells) else "") for i in range(len(headers))})
    except Exception as e:
        print("ERROR fetching inventory CSV:", e)
    _inv_cache = rows
    _inv_age = time.time()
    return rows


# ---------- SALESPEOPLE ----------
def load_salespeople() -> List[Dict[str, Any]]:
    try:
        return [a for a in json.loads(SALESPEOPLE_JSON or "[]") if a.get("name")]
    except Exception:
        return []


def pick_salesperson_by_name(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").lower()
    for sp in load_salespeople():
        if sp["name"].lower() in t:
            return sp
    return None


def pick_salesperson_round_robin() -> Optional[Dict[str, Any]]:
    global _round_robin_idx
    sps = load_salespeople()
    if not sps:
        return None
    sp = sps[_round_robin_idx % len(sps)]
    _round_robin_idx += 1
    return sp


# ---------- EXTRACTION / MATCH ----------
def extract_make_model(text: str) -> (Optional[str], Optional[str]):
    if not text:
        return None, None
    words = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    inv = load_inventory()
    makes = {(r.get("Make", "") or "").lower() for r in inv if r.get("Make")}
    models = {(r.get("Model", "") or "").lower().replace("-", "") for r in inv if r.get("Model")}
    mk = md = None
    for w in words:
        wn = w.replace("-", "")
        if not mk and w in makes:
            mk = w
        if not md and wn in models:
            md = wn
    return mk, md


def find_vehicle(make: Optional[str], model: Optional[str]) -> Optional[Dict[str, Any]]:
    inv = load_inventory()
    mk = (make or "").lower()
    md = (model or "").lower()
    for r in inv:
        rmk = (r.get("Make", "") or "").lower()
        rmd = (r.get("Model", "") or "").lower().replace("-", "")
        if (not md or md in rmd) and (not mk or mk in rmk):
            return r
    return None


def vehicle_blurb(v: Dict[str, Any]) -> str:
    year = str(v.get("Year", "")).strip()
    make = v.get("Make", "").strip()
    model = v.get("Model", "").strip()
    color = v.get("Color", "").strip()
    cond = v.get("Condition", "").strip()
    price = v.get("Price", "").strip()
    bits = []
    if year and model:
        bits.append(f"{year} {make} {model}")
    if color:
        bits.append(color)
    if cond:
        bits.append(cond)
    if price:
        bits.append(f"${price}")
    return " · ".join(bits)


# ---------- GOOGLE CALENDAR ----------
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


def propose_slots(sales_cal_id: Optional[str]) -> List[str]:
    tz = ZoneInfo(TIMEZONE)
    service = _google_service()
    if service and DEALERSHIP_CALENDAR_ID and sales_cal_id:
        now = datetime.now(tz)
        horizon = now + timedelta(days=5)
        fb = _freebusy(service, [DEALERSHIP_CALENDAR_ID, sales_cal_id], now, horizon, TIMEZONE)
        dealer_busy = fb.get(DEALERSHIP_CALENDAR_ID, [])
        sales_busy = fb.get(sales_cal_id, [])
        slots = []
        day = now
        while day < horizon and len(slots) < 2:
            for hour in [9, 11, 13, 15]:
                s = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                e = s + timedelta(minutes=30)
                if s < now:
                    continue
                if _is_free(s, e, dealer_busy) and _is_free(s, e, sales_busy):
                    slots.append(s.strftime("%A %I:%M %p"))
                    if len(slots) == 2:
                        break
            day = (day + timedelta(days=1)).replace(hour=9, minute=0)
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
        label = start_str.lower()
        if "tomorrow" in label:
            base = now + timedelta(days=1)
            hour = 10 if "10" in label else (14 if "2" in label else 10)
            start = base.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            wd_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target = next((i for i, n in enumerate(wd_names) if n in label), None)
            d = now
            for _ in range(8):
                if target is None or d.weekday() == target:
                    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", label)
                    hh = int(m.group(1)) if m else 10
                    mm = int(m.group(2)) if (m and m.group(2)) else 0
                    ap = (m.group(3) if m else "am").lower()
                    if ap == "pm" and hh < 12:
                        hh += 12
                    start = d.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if start > now:
                        break
                d = d + timedelta(days=1)
        end = start + timedelta(minutes=30)
    except Exception as e:
        print("parse slot error:", e)
        return None

    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start.astimezone(ZoneInfo("UTC")).isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end.astimezone(ZoneInfo("UTC")).isoformat(), "timeZone": TIMEZONE},
        "attendees": [{"email": sp.get("email")}] + ([{"email": customer_email}] if customer_email else []),
    }
    try:
        created = service.events().insert(
            calendarId=DEALERSHIP_CALENDAR_ID,
            body=event,
            sendUpdates="all"
        ).execute()
        return created.get("htmlLink")
    except Exception as e:
        print("calendar create error:", e)
        return None


# ---------- NOTIFICATIONS ----------
def send_sms(to: str, body: str):
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and to):
        return
    try:
        from twilio.rest import Client
        Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
            to=to,
            from_=TWILIO_FROM_NUMBER,
            body=body
        )
    except Exception as e:
        print("twilio error:", e)


def send_email(to: str, subject: str, text: str):
    if not (SENDGRID_API_KEY and FROM_EMAIL and to):
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        msg = Mail(
            from_email=FROM_EMAIL,
            to_emails=to,
            subject=subject,
            plain_text_content=text
        )
        SendGridAPIClient(SENDGRID_API_KEY).send(msg)
    except Exception as e:
        print("sendgrid error:", e)


def post_analytics(payload: Dict[str, Any]):
    if not ANALYTICS_WEBHOOK_URL:
        return
    try:
        with httpx.Client(timeout=6.0) as c:
            c.post(ANALYTICS_WEBHOOK_URL, json=payload)
    except Exception as e:
        print("analytics post error:", e)


# ---------- HELPERS ----------
def extract_contact(text: str):
    phone = email = None
    if text:
        m = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text)
        if m:
            phone = re.sub(r"[^\d+]", "", m.group(1))
            if len(phone) < 10:
                phone = None
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if m:
            email = m.group(0)
    return phone, email


# ---------- API ----------
app = FastAPI(title="DealerBrain", version="0.1")


@app.get("/health")
def health():
    return {
        "ok": True,
        "sheet": bool(INVENTORY_CSV_URL),
        "calendar_ready": bool(GOOGLE_SERVICE_ACCOUNT_JSON and DEALERSHIP_CALENDAR_ID and load_salespeople()),
        "sms_ready": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER),
        "email_ready": bool(SENDGRID_API_KEY and FROM_EMAIL),
        "timezone": TIMEZONE
    }


class VoiceTurn(BaseModel):
    session_id: str
    transcript: Optional[str] = ""
    caller_phone: Optional[str] = None


@app.post("/webhooks/voice")
async def webhooks_voice(turn: VoiceTurn):
    s = get_session(turn.session_id)
    text = (turn.transcript or "").strip()

    if s["state"] == "GREETING":
        s["state"] = "QUALIFY"
        return {
            "reply": f"Thanks for calling {DEALERSHIP_NAME}! Which car are you interested in? You can say make and model.",
            "handoff": False,
            "end_call": False
        }

    if s["state"] == "QUALIFY":
        mk, md = extract_make_model(text)
        if mk:
            s["make"] = mk
        if md:
            s["model"] = md
        if not s["model"]:
            return {
                "reply": "Got it. Which model are you looking for?",
                "handoff": False,
                "end_call": False
            }
        v = find_vehicle(s["make"], s["model"])
        if v:
            s["vehicle"] = v
            sp = pick_salesperson_by_name(text) or pick_salesperson_round_robin()
            s["salesperson"] = sp
            s["proposed"] = propose_slots(sp.get("calendar_id") if sp else None)
            s["state"] = "SLOTS"
            return {
                "reply": f"We have {vehicle_blurb(v)} available. Times: {', '.join(s['proposed'])}. What works for you?",
                "handoff": False,
                "end_call": False
            }
        else:
            return {
                "reply": "I couldn’t find that exact vehicle in our inventory. Want to try a different make or model?",
                "handoff": False,
                "end_call": False
            }

    if s["state"] == "SLOTS":
        chosen = None
        for sl in s["proposed"]:
            if sl and sl.lower() in text.lower():
                chosen = sl
                break
        if not chosen:
            if re.search(r"\bfirst\b", text.lower()):
                chosen = s["proposed"][0] if s["proposed"] else None
            if re.search(r"\bsecond\b", text.lower()) and len(s["proposed"]) > 1:
                chosen = s["proposed"][1]
        if not chosen:
            return {
                "reply": f"Which works best: {', '.join(s['proposed'])}?",
                "handoff": False,
                "end_call": False
            }
        s["chosen"] = chosen
        s["state"] = "CONTACT"
        return {
            "reply": "Great—what’s the best phone number and email for confirmation?",
            "handoff": False,
            "end_call": False
        }

    if s["state"] == "CONTACT":
        ph, em = extract_contact(text)
        if turn.caller_phone and not s["phone"]:
            s["phone"] = turn.caller_phone
        if ph:
            s["phone"] = ph
        if em:
            s["email"] = em
        if not (s.get("phone") and s.get("email")):
            return {
                "reply": "I’ll need both a phone number and an email to confirm. Can you share both?",
                "handoff": False,
                "end_call": False
            }

        sp = s.get("salesperson") or {}
        v = s.get("vehicle") or {}
        slot = s.get("chosen") or ""
        title = f"Showroom Appointment — {vehicle_blurb(v)}"
        desc = (
            f"Customer: {s.get('email')} / {s.get('phone')}\n"
            f"Salesperson: {sp.get('name')} <{sp
