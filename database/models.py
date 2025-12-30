"""
Database Models - Phase 4
SQLite database with listings and price_history tables
"""

import sqlite3
from datetime import datetime
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create listings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id TEXT UNIQUE,
                    title TEXT NOT NULL,
                    price TEXT,
                    city TEXT,
                    url TEXT UNIQUE,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    mileage TEXT,
                    seller_type TEXT,
                    time_on_market TEXT,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create price_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id TEXT,
                    price TEXT,
                    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (listing_id) REFERENCES listings(listing_id)
                )
            ''')

            conn.commit()

    def extract_listing_info(self, title, url):
        """Extract make, model, year from title and URL"""
        make = ""
        model = ""
        year = None

        # Try to extract from title
        title_lower = title.lower()

        # Common makes
        makes = ['toyota', 'honda', 'suzuki', 'daihatsu', 'hyundai', 'nissan', 'mercedes', 'bmw', 'audi', 'kia', 'mg', 'changan', 'faw']

        for m in makes:
            if m in title_lower:
                make = m.title()
                break

        # Extract year (4 digits)
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        if year_match:
            year = int(year_match.group())

        # Extract model (words after make)
        if make:
            make_pos = title_lower.find(make.lower())
            if make_pos >= 0:
                after_make = title[make_pos + len(make):].strip()
                # Take first word as model
                model = after_make.split()[0] if after_make else ""

        return make, model, year

    def insert_or_update_listing(self, title, price, city, url):
        """Insert new listing or update existing one"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Extract additional info
            make, model, year = self.extract_listing_info(title, url)

            # Generate listing_id from URL
            listing_id = url.split('-')[-1] if '-' in url else str(hash(url))

            # Check if listing exists
            cursor.execute('SELECT id, price FROM listings WHERE listing_id = ?', (listing_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing listing
                old_price = existing[1]
                cursor.execute('''
                    UPDATE listings
                    SET price = ?, city = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE listing_id = ?
                ''', (price, city, listing_id))

                # If price changed, add to history
                if old_price != price:
                    cursor.execute('''
                        INSERT INTO price_history (listing_id, price)
                        VALUES (?, ?)
                    ''', (listing_id, price))
            else:
                # Insert new listing
                cursor.execute('''
                    INSERT INTO listings (listing_id, title, price, city, url, make, model, year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (listing_id, title, price, city, url, make, model, year))

                # Add initial price to history
                cursor.execute('''
                    INSERT INTO price_history (listing_id, price)
                    VALUES (?, ?)
                ''', (listing_id, price))

            conn.commit()

    def import_csv_data(self, csv_path):
        """Import data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            imported_count = 0

            for _, row in df.iterrows():
                try:
                    self.insert_or_update_listing(
                        row['title'],
                        row['price'],
                        row['city'],
                        row['url']
                    )
                    imported_count += 1
                except Exception as e:
                    print(f"Error importing row: {e}")
                    continue

            print(f"Imported {imported_count} listings from CSV")
            return imported_count

        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}")
            return 0

    def get_listing_count(self):
        """Get total number of listings"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM listings')
            return cursor.fetchone()[0]

    def get_recent_listings(self, limit=10):
        """Get recently added listings"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT title, price, city, first_seen
                FROM listings
                ORDER BY first_seen DESC
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

    def get_price_changes(self, days=7):
        """Get listings with price changes in last N days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT l.title, l.price, ph.price as old_price, ph.recorded_at
                FROM listings l
                JOIN price_history ph ON l.listing_id = ph.listing_id
                WHERE ph.recorded_at >= datetime('now', '-{} days')
                AND ph.price != l.price
                ORDER BY ph.recorded_at DESC
            '''.format(days))
            return cursor.fetchall()

def main():
    """Test database functionality"""
    db = DatabaseManager()

    # Import CSV data
    imported = db.import_csv_data('data/sample_listings.csv')
    print(f"Database initialized with {imported} listings")

    # Show stats
    total = db.get_listing_count()
    print(f"Total listings in database: {total}")

    # Show recent listings
    recent = db.get_recent_listings(5)
    print("\nRecent listings:")
    for listing in recent:
        print(f"- {listing[0]}: {listing[1]} ({listing[2]})")

if __name__ == "__main__":
    main()
