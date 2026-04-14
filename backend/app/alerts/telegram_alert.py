"""
AI Sentinel Lite - Telegram Alert System
------------------------------------------
Sends anomaly screenshots and event notifications to your
Telegram account via a bot. Fully async, non-blocking.

Setup:
  1. Message @BotFather on Telegram -> /newbot -> copy the token
  2. Message @userinfobot on Telegram -> copy your Chat ID
  3. Put both into sentinel_config.json
"""

import os
import json
import time
import threading
import requests


CONFIG_PATH = "sentinel_config.json"

DEFAULT_CONFIG = {
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "alert_cooldown_seconds": 30,
    "alert_on_unknown_person": True,
    "alert_on_zone_intrusion": True,
    "alert_on_loitering": True,
    "alert_on_fast_movement": True,
}


class TelegramAlerter:
    def __init__(self):
        self.config = self._load_config()
        self.token = self.config.get("telegram_bot_token", "")
        self.chat_id = self.config.get("telegram_chat_id", "")
        self.cooldown = self.config.get("alert_cooldown_seconds", 30)
        self.last_alert_time = 0
        self.enabled = bool(self.token and self.chat_id)

        if self.enabled:
            print(f"[TELEGRAM] Bot connected. Alerts will be sent to chat {self.chat_id}.")
        else:
            print("[TELEGRAM] Not configured. Edit sentinel_config.json to enable alerts.")

    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        else:
            # Create default config file
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            print(f"[TELEGRAM] Created {CONFIG_PATH}. Edit it with your bot token and chat ID.")
            return DEFAULT_CONFIG

    def _send_message(self, text):
        """Send a text message to Telegram."""
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:
            print(f"[TELEGRAM] Failed to send message: {e}")

    def _send_photo(self, photo_path, caption=""):
        """Send a photo with caption to Telegram."""
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        try:
            with open(photo_path, "rb") as photo:
                requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=15,
                )
        except Exception as e:
            print(f"[TELEGRAM] Failed to send photo: {e}")

    def alert(self, reason, photo_path=None):
        """
        Send an alert if cooldown has elapsed. Non-blocking (runs in thread).
        """
        now = time.time()
        if now - self.last_alert_time < self.cooldown:
            return
        self.last_alert_time = now

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚨 SENTINEL ALERT\n\n{reason}\n\n🕐 {timestamp}"

        def _send():
            if photo_path and os.path.exists(photo_path):
                self._send_photo(photo_path, caption=message)
                print(f"[TELEGRAM] Alert sent with photo: {reason}")
            else:
                self._send_message(message)
                print(f"[TELEGRAM] Alert sent: {reason}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()
