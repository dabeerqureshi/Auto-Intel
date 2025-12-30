"""
Scheduler Script - Phase 5
Automate periodic data collection using Python scheduling
"""

import time
import schedule
from datetime import datetime
import subprocess
import sys
import os

class DataScheduler:
    def __init__(self, interval_minutes=30):
        self.interval_minutes = interval_minutes
        self.scraper_script = os.path.join(os.path.dirname(__file__), '..', 'scraper', 'simple_scraper.py')

    def run_scraper(self):
        """Execute the scraper script"""
        print(f"[{datetime.now()}] Starting scheduled scraper run...")

        try:
            # Run the scraper
            result = subprocess.run([
                sys.executable, self.scraper_script
            ], capture_output=True, text=True, cwd=os.path.dirname(self.scraper_script))

            if result.returncode == 0:
                print(f"[{datetime.now()}] Scraper completed successfully")
                # Here you could add logic to import new data to database
                self.import_new_data()
            else:
                print(f"[{datetime.now()}] Scraper failed: {result.stderr}")

        except Exception as e:
            print(f"[{datetime.now()}] Error running scraper: {e}")

    def import_new_data(self):
        """Import newly scraped data to database"""
        try:
            # Import to database
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from database.models import DatabaseManager

            db = DatabaseManager()
            imported = db.import_csv_data('data/sample_listings.csv')
            print(f"[{datetime.now()}] Imported {imported} listings to database")

        except Exception as e:
            print(f"[{datetime.now()}] Error importing data: {e}")

    def start_scheduler(self):
        """Start the periodic scheduler"""
        print(f"Starting AutoIntel data scheduler (every {self.interval_minutes} minutes)")

        # Schedule the job
        schedule.every(self.interval_minutes).minutes.do(self.run_scraper)

        # Run once immediately for testing
        print("Running initial scraper...")
        self.run_scraper()

        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")

def main():
    """Main scheduler function"""
    import argparse

    parser = argparse.ArgumentParser(description='AutoIntel Data Scheduler')
    parser.add_argument('--interval', type=int, default=30,
                       help='Interval in minutes between scraper runs (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Run scraper once and exit (for testing)')

    args = parser.parse_args()

    scheduler = DataScheduler(interval_minutes=args.interval)

    if args.once:
        # Run once for testing
        scheduler.run_scraper()
    else:
        # Start continuous scheduler
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()
