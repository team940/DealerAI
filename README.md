# Dealer AI Brain (FastAPI) — ElevenLabs + Render

A tiny webhook “brain” for a dealership voice agent. ElevenLabs sends each caller turn here; the brain checks your Google Sheets inventory, assigns a salesperson, proposes two times, captures contact info, books a Google Calendar event, and sends SMS/email confirmations.

## Environment variables (set these on Render)
Required:
- INVENTORY_CSV_URL = https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv
- TIMEZONE = America/New_York
- DEALERSHIP_NAME = Your Dealership

Recommended (enable real booking + notifications):
- SALESPEOPLE_JSON = JSON array, e.g.
  [
    {"name":"Alex Rivera","email":"alex@dealer.com","calendar_id":"alex@dealer.com","phone":"+15551110001"},
    {"name":"Brooke Tan","email":"brooke@dealer.com","calendar_id":"brooke@dealer.com","phone":"+15551110002"}
  ]
- DEALERSHIP_CALENDAR_ID = dealership@example.com
- GOOGLE_SERVICE_ACCOUNT_JSON = full JSON of Google service account with Calendar scope
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
- SENDGRID_API_KEY, FROM_EMAIL
- ANALYTICS_WEBHOOK_URL (optional)
- APPT_DEFAULT_SLOTS_JSON (optional fallback), e.g. ["Tomorrow 10 AM","Thursday 2 PM"]

## Deploy on Render
Build command:
pip install -r requirements.txt

Start command:
uvicorn main:app --host 0.0.0.0 --port $PORT

## Health check
Once deployed, open your browser and go to:
https://YOUR-RENDER-URL/health
You should see:
{"ok":true, ...}

## ElevenLabs (workflow webhook)
- URL: https://YOUR-RENDER-URL/webhooks/voice
- Method: POST
- Header: Content-Type=application/json
- Body params (Dynamic Variables):
  - session_id ← conversation_id (Required)
  - transcript ← last_user_message (Required)
  - caller_phone ← phone variable if available (Optional)
- After tool, subagent prompt:
  If {{YourWebhook.end_call}} is true, politely say goodbye and end the call.
  Otherwise, speak exactly the text in {{YourWebhook.reply}}.
- Loop back to the webhook with an “Always true” condition.

## Curl test (optional)
You can test your webhook manually from a terminal:
curl -X POST "https://YOUR-RENDER-URL/webhooks/voice" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"t1","transcript":"hi","caller_phone":"+16135551234"}'
