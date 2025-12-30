"""
Simple Scraper using requests and BeautifulSoup - Phase 3 Alternative
Extract listings from saved HTML file
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_from_html():
    """
    Extract car listings from saved HTML file
    """
    listings = []

    try:
        with open('data/page_content.html', 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find car listings in featured section
        featured_section = soup.find('div', class_='car-featured-used-home')
        if featured_section:
            cards = featured_section.find_all('div', class_='cards')
            print(f"Found {len(cards)} featured cards")

            for i, card in enumerate(cards[:10]):  # Take first 10
                try:
                    # Extract title
                    title_elem = card.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else "Title not found"

                    # Extract price
                    price_elem = card.find('div', class_='generic-green')
                    price = price_elem.get_text(strip=True) if price_elem else "Price not found"

                    # Extract city
                    city_elem = card.find('div', class_='generic-gray')
                    city = city_elem.get_text(strip=True) if city_elem else "City not found"

                    # Extract URL
                    link_elem = card.find('a')
                    href = link_elem.get('href') if link_elem else None
                    url = f"https://www.pakwheels.com{href}" if href and href.startswith('/') else href or "URL not found"

                    listings.append({
                        'title': title,
                        'price': price,
                        'city': city,
                        'url': url
                    })

                except Exception as e:
                    print(f"Error extracting listing {i}: {e}")
                    continue

        else:
            print("Featured section not found")

        # If no listings from featured, try other sections
        if not listings:
            all_cards = soup.find_all('div', class_='cards')
            print(f"Found {len(all_cards)} total cards")

            for i, card in enumerate(all_cards[:10]):
                try:
                    title_elem = card.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else "Title not found"

                    price_elem = card.find('div', class_='generic-green')
                    price = price_elem.get_text(strip=True) if price_elem else "Price not found"

                    city_elem = card.find('div', class_='generic-gray')
                    city = city_elem.get_text(strip=True) if city_elem else "City not found"

                    link_elem = card.find('a')
                    href = link_elem.get('href') if link_elem else None
                    url = f"https://www.pakwheels.com{href}" if href and href.startswith('/') else href or "URL not found"

                    listings.append({
                        'title': title,
                        'price': price,
                        'city': city,
                        'url': url
                    })

                except Exception as e:
                    print(f"Error extracting listing {i}: {e}")
                    continue

    except FileNotFoundError:
        print("HTML file not found. Please run the Playwright scraper first.")

    return listings

def main():
    print("Starting simple HTML scraper...")
    listings = scrape_from_html()

    print(f"Extracted {len(listings)} listings")

    if listings:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(listings)
        df.to_csv('data/sample_listings.csv', index=False)
        print("Data saved to data/sample_listings.csv")
        print("\nFirst few listings:")
        print(df.head())
    else:
        print("No listings extracted from HTML.")

if __name__ == "__main__":
    main()
