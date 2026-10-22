# Baby Detect

Every check interval this app fetches a snapshot from a UniFi Protect camera,
runs a YOLOv8n person detector on it (CPU, single thread), and POSTs
`{"detected": true|false, "confidence": 0.53}` to a Home Assistant webhook when
the debounced state changes, plus a periodic heartbeat.

The Home Assistant automation on that webhook must set the state from
`trigger.json.detected` (turn on/off), not toggle it.

## Webhook URL

From inside Home Assistant the app can reach Core directly at
`http://homeassistant:8123/api/webhook/<webhook_id>`. This avoids certificate
issues and passes the webhook trigger's `local_only` check.

## Region of interest

Set `roi` to the crib's position as fractions of the image, e.g.
`0.25,0.30,0.75,0.95`, so people outside the crib are ignored.
