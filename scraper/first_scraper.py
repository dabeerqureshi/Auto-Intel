"""
First Scraper Implementation - Phase 3
Limited scope: Karachi, Toyota Corolla, max 10 listings
Extract: title, price, city, url
Output: CSV format
"""

import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re

async def scrape_pakwheels():
    """
    Scrape PakWheels used cars page for Toyota Corolla in Karachi
    """
    listings = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )

        # Navigate to used cars page
        await page.goto("https://www.pakwheels.com/used-cars/", timeout=60000)

        # Wait for page to load
        await page.wait_for_load_state('networkidle')

        # Wait for car listings to appear (they load dynamically)
        await page.wait_for_selector('.car-featured-used-home .cards', timeout=10000)

        # Scroll down to load more content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)

        # Debug: Save page content for analysis
        content = await page.content()
        with open('data/page_content.html', 'w', encoding='utf-8') as f:
            f.write(content)
        print("Page content saved to data/page_content.html")

        # Print page title
        title = await page.title()
        print(f"Page title: {title}")

        # Try to filter by city (Karachi) - this might require clicking
        # For now, we'll scrape from main page and filter later
        # TODO: Implement city filtering

        # Try to search for Toyota Corolla
        # Look for search input
        try:
            search_input = page.locator('input[placeholder*="search"]').first
            await search_input.fill("Toyota Corolla")
            await search_input.press("Enter")
            await page.wait_for_load_state('networkidle')
        except:
            print("Search input not found, proceeding with main page")

        # Extract listings - look for car listing containers
        # Based on typical structure, listings are in divs with car info

        # Try different selectors for car listings
        selectors = [
            '.car-listing',
            '.search-results .car-item',
            '.car-listing-item',
            'article.car-listing',
            '.car-box'
        ]

        car_elements = None
        for selector in selectors:
            car_elements = page.locator(selector)
            count = await car_elements.count()
            if count > 0:
                print(f"Found {count} listings with selector: {selector}")
                break

        if car_elements is None:
            print("No car listings found with known selectors")
            # Try to get all links that look like car listings
            all_links = page.locator('a[href*="/used-cars/"]')
            count = await all_links.count()
            print(f"Found {count} potential car links")

            # Take first 10
            for i in range(min(10, count)):
                link = all_links.nth(i)
                href = await link.get_attribute('href')
                text = await link.inner_text()
                if href and 'used-cars' in href and len(text.strip()) > 10:
                    # Extract basic info from text
                    title = text.strip()
                    price = "Price not extracted"
                    city = "City not extracted"
                    url = f"https://www.pakwheels.com{href}" if href.startswith('/') else href

                    listings.append({
                        'title': title,
                        'price': price,
                        'city': city,
                        'url': url
                    })
        else:
            # Extract from found elements
            count = await car_elements.count()
            for i in range(min(10, count)):
                element = car_elements.nth(i)

                try:
                    # Try to extract title
                    title_selectors = ['h3', '.car-title', '.title', 'a']
                    title = ""
                    for ts in title_selectors:
                        try:
                            title_elem = element.locator(ts).first
                            title = await title_elem.inner_text()
                            if title:
                                break
                        except:
                            continue

                    # Try to extract price
                    price_selectors = ['.price', '.car-price', '[class*="price"]']
                    price = ""
                    for ps in price_selectors:
                        try:
                            price_elem = element.locator(ps).first
                            price = await price_elem.inner_text()
                            if price:
                                break
                        except:
                            continue

                    # Try to extract city
                    city_selectors = ['.city', '.location', '[class*="city"]']
                    city = ""
                    for cs in city_selectors:
                        try:
                            city_elem = element.locator(cs).first
                            city = await city_elem.inner_text()
                            if city:
                                break
                        except:
                            continue

                    # Extract URL
                    url = ""
                    try:
                        link = element.locator('a').first
                        href = await link.get_attribute('href')
                        url = f"https://www.pakwheels.com{href}" if href and href.startswith('/') else href
                    except:
                        pass

                    if title or url:
                        listings.append({
                            'title': title.strip() if title else "Title not found",
                            'price': price.strip() if price else "Price not found",
                            'city': city.strip() if city else "City not found",
                            'url': url if url else "URL not found"
                        })

                except Exception as e:
                    print(f"Error extracting listing {i}: {e}")
                    continue

        await browser.close()

    return listings

def main():
    print("Starting PakWheels scraper...")
    listings = asyncio.run(scrape_pakwheels())

    print(f"Extracted {len(listings)} listings")

    if listings:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(listings)
        df.to_csv('data/sample_listings.csv', index=False)
        print("Data saved to data/sample_listings.csv")
        print("\nFirst few listings:")
        print(df.head())
    else:
        print("No listings extracted. Check selectors or page structure.")

if __name__ == "__main__":
    main()
