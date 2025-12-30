"""
Intelligence Features - Phase 6
Basic analytics and alerts for car listings
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import statistics

class MarketIntelligence:
    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path

    def get_listings_dataframe(self):
        """Get all listings as pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM listings", conn)
        return df

    def calculate_average_prices(self):
        """Calculate average prices by make, model, and city"""
        df = self.get_listings_dataframe()

        # Clean price data (remove PKR and commas)
        df['price_clean'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
        df['price_numeric'] = pd.to_numeric(df['price_clean'], errors='coerce')

        # Group by make and calculate averages
        make_avg = df.groupby('make')['price_numeric'].agg(['mean', 'count', 'min', 'max']).round(0)

        # Group by city and calculate averages
        city_avg = df.groupby('city')['price_numeric'].agg(['mean', 'count', 'min', 'max']).round(0)

        return {
            'by_make': make_avg,
            'by_city': city_avg,
            'overall_avg': df['price_numeric'].mean()
        }

    def detect_underpriced_vehicles(self, threshold_percentile=25):
        """Detect potentially underpriced vehicles"""
        df = self.get_listings_dataframe()

        # Clean price data
        df['price_clean'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
        df['price_numeric'] = pd.to_numeric(df['price_clean'], errors='coerce')

        # Calculate price percentiles by make
        underpriced = []

        for make in df['make'].unique():
            if pd.isna(make) or make == '':
                continue

            make_data = df[df['make'] == make].copy()
            if len(make_data) < 3:  # Need at least 3 data points
                continue

            # Calculate threshold (e.g., 25th percentile)
            threshold = make_data['price_numeric'].quantile(threshold_percentile / 100)

            # Find underpriced vehicles
            cheap_ones = make_data[make_data['price_numeric'] <= threshold]

            for _, vehicle in cheap_ones.iterrows():
                underpriced.append({
                    'title': vehicle['title'],
                    'price': vehicle['price'],
                    'city': vehicle['city'],
                    'make': vehicle['make'],
                    'threshold': f"PKR {threshold:,.0f}",
                    'savings_potential': f"PKR {(threshold - vehicle['price_numeric']):,.0f}"
                })

        return underpriced

    def get_price_alerts(self):
        """Generate alerts for price changes and opportunities"""
        alerts = []

        # Get average prices
        averages = self.calculate_average_prices()

        # Get underpriced vehicles
        underpriced = self.detect_underpriced_vehicles()

        # Create alerts
        for vehicle in underpriced[:5]:  # Top 5 underpriced
            alerts.append({
                'type': 'underpriced',
                'title': f"Potentially underpriced {vehicle['make']}",
                'message': f"{vehicle['title']} in {vehicle['city']} at {vehicle['price']} (threshold: {vehicle['threshold']})",
                'priority': 'high'
            })

        # Add market summary
        if 'by_make' in averages and not averages['by_make'].empty:
            top_make = averages['by_make'].iloc[0]
            alerts.append({
                'type': 'market_info',
                'title': 'Market Average Prices',
                'message': f"Overall average car price: PKR {averages['overall_avg']:,.0f}",
                'priority': 'info'
            })

        return alerts

    def get_recent_price_changes(self, days=7):
        """Get listings with recent price changes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT l.title, l.price, ph.price as old_price,
                       ph.recorded_at, l.city, l.make
                FROM listings l
                JOIN price_history ph ON l.listing_id = ph.listing_id
                WHERE ph.recorded_at >= datetime('now', '-{} days')
                AND ph.price != l.price
                ORDER BY ph.recorded_at DESC
                LIMIT 10
            '''.format(days))

            changes = []
            for row in cursor.fetchall():
                changes.append({
                    'title': row[0],
                    'current_price': row[1],
                    'old_price': row[2],
                    'changed_at': row[3],
                    'city': row[4],
                    'make': row[5]
                })

            return changes

    def generate_market_report(self):
        """Generate a comprehensive market report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'averages': self.calculate_average_prices(),
            'underpriced_vehicles': self.detect_underpriced_vehicles(),
            'price_changes': self.get_recent_price_changes(),
            'alerts': self.get_price_alerts()
        }

        return report

def print_market_report():
    """Print a formatted market report"""
    intelligence = MarketIntelligence()

    print("=== AutoIntel Market Intelligence Report ===\n")

    # Average prices
    print("📊 AVERAGE PRICES:")
    averages = intelligence.calculate_average_prices()

    print(f"Overall Average: PKR {averages['overall_avg']:,.0f}")

    if 'by_make' in averages and not averages['by_make'].empty:
        print("\nBy Make:")
        for make, data in averages['by_make'].iterrows():
            if pd.notna(make):
                print(f"  {make}: PKR {data['mean']:,.0f} (avg), {int(data['count'])} listings")

    if 'by_city' in averages and not averages['by_city'].empty:
        print("\nBy City:")
        for city, data in averages['by_city'].iterrows():
            if pd.notna(city):
                print(f"  {city}: PKR {data['mean']:,.0f} (avg), {int(data['count'])} listings")

    # Underpriced vehicles
    print("\n🎯 POTENTIALLY UNDERPRICED VEHICLES:")
    underpriced = intelligence.detect_underpriced_vehicles()
    for vehicle in underpriced[:5]:  # Show top 5
        print(f"  • {vehicle['title']} - {vehicle['price']} in {vehicle['city']}")
        print(f"    Savings potential: {vehicle['savings_potential']}")

    # Price changes
    print("\n📈 RECENT PRICE CHANGES:")
    changes = intelligence.get_recent_price_changes()
    if changes:
        for change in changes[:3]:  # Show last 3
            print(f"  • {change['title']}: {change['old_price']} → {change['current_price']}")
    else:
        print("  No recent price changes")

    # Alerts
    print("\n🚨 ALERTS:")
    alerts = intelligence.get_price_alerts()
    for alert in alerts:
        emoji = "🔴" if alert['priority'] == 'high' else "ℹ️"
        print(f"  {emoji} {alert['title']}: {alert['message']}")

    print(f"\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main intelligence function"""
    print_market_report()

if __name__ == "__main__":
    main()
