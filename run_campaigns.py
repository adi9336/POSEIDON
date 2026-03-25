
"""
Standalone campaign runner.
Run alongside app.py or independently for autonomous monitoring.
Usage: python run_campaigns.py
"""
import logging
import time
from src.campaigns.manager import CampaignManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if __name__ == "__main__":
    print("POSEIDON Campaign System starting...")
    manager = CampaignManager()
    manager.start()
    print("All campaigns running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down campaigns...")
        manager.stop()
        print("Done.")
