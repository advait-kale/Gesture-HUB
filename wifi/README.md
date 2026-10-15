# Gesture-HUB Wi-Fi API

This Arduino sketch runs gesture inference on the ESP32 and exposes the latest
result over HTTP on the same Wi-Fi network.

## Setup

1. Open the `wifi` folder as an Arduino sketch (`wifi.ino`).
2. Edit `wifi_config.h` with the 2.4 GHz network name and password.
3. Select **ESP32 Dev Module** and upload over the USB-to-UART COM port.
4. Open Serial Monitor at **115200 baud** and note the `API ready at http://...`
   address.
5. Connect the phone/computer making API requests to the same Wi-Fi network.
6. Open `dashboard.html` in a browser, enter the printed IP if it differs from
   the default, and select **Connect**. The page polls `/api/result` once per
   second and shows recent results it has observed.

The API listens on port 80. It keeps waiting for a gesture, classifies each
captured motion, and updates the latest result. It does not need a request to
start a capture.

## Endpoints

### `GET /api/status`

Returns model readiness, Wi-Fi connection state, IP address, and the most
recent event ID.

### `GET /api/result`

Returns the most recent result. `event_id` increments for each captured gesture;
clients can poll this endpoint and act only when the ID changes.

`GET /api/gesture/latest` is also available as an alias.

Example response:

```json
{
  "available": true,
  "event_id": 1,
  "gesture": "left_twist",
  "confidence": 0.91,
  "accepted": true,
  "timestamp_ms": 18234
}
```

`accepted` is false when confidence is below 0.70. Before the first gesture,
the endpoint returns `{"available":false,...}`.

### `GET /`

Returns a small JSON description of the API.

These endpoints are unauthenticated and intended for use on a trusted local
network. The sketch includes permissive CORS headers for browser clients.
