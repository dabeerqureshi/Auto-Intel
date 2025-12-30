"""
AutoIntel Dashboard - Phase 7
Streamlit web interface for market intelligence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.models import DatabaseManager
from scripts.intelligence import MarketIntelligence

# Page configuration
st.set_page_config(
    page_title="AutoIntel - Car Market Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize database and intelligence components"""
    db = DatabaseManager()
    intelligence = MarketIntelligence()
    return db, intelligence

db, intelligence = init_components()

def main():
    """Main dashboard application"""
    st.title("🚗 AutoIntel - Car Market Intelligence")
    st.markdown("**Real-time Pakistani used car market analytics and insights**")

    # Sidebar
    st.sidebar.header("📊 Market Overview")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_listings = db.get_listing_count()
        st.metric("Total Listings", f"{total_listings:,}")

    with col2:
        averages = intelligence.calculate_average_prices()
        avg_price = f"PKR {averages['overall_avg']:,.0f}"
        st.metric("Average Price", avg_price)

    with col3:
        underpriced = len(intelligence.detect_underpriced_vehicles())
        st.metric("Underpriced Deals", underpriced)

    with col4:
        changes = len(intelligence.get_recent_price_changes())
        st.metric("Recent Changes", changes)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Market Overview",
        "🎯 Deal Alerts",
        "📋 Listings",
        "📊 Analytics",
        "🔮 Price Predictor",
        "📈 Advanced Tools"
    ])

    with tab1:
        show_market_overview()

    with tab2:
        show_deal_alerts()

    with tab3:
        show_listings_table()

    with tab4:
        show_analytics_charts()

    with tab5:
        show_price_predictor()

    with tab6:
        show_advanced_tools()

@st.cache_data(ttl=300)
def show_market_overview():
    """Display market overview with averages and trends"""
    st.header("📈 Market Overview")

    averages = intelligence.calculate_average_prices()

    # Average prices by make
    if 'by_make' in averages and not averages['by_make'].empty:
        st.subheader("Average Prices by Make")

        # Prepare data for chart
        make_data = averages['by_make'].reset_index()
        make_data.columns = ['Make', 'Average Price', 'Count', 'Min Price', 'Max Price']
        make_data = make_data.sort_values('Average Price', ascending=False)

        # Bar chart
        fig = px.bar(
            make_data.head(10),
            x='Make',
            y='Average Price',
            title='Top 10 Car Makes by Average Price',
            labels={'Average Price': 'Price (PKR)'},
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.dataframe(make_data[['Make', 'Average Price', 'Count']].head(10), use_container_width=True)

    # Average prices by city
    if 'by_city' in averages and not averages['by_city'].empty:
        st.subheader("Average Prices by City")

        city_data = averages['by_city'].reset_index()
        city_data.columns = ['City', 'Average Price', 'Count', 'Min Price', 'Max Price']
        city_data = city_data.sort_values('Average Price', ascending=False)

        # Pie chart
        fig = px.pie(
            city_data,
            values='Count',
            names='City',
            title='Listings Distribution by City'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(city_data, use_container_width=True)

@st.cache_data(ttl=300)
def show_deal_alerts():
    """Display deal alerts and underpriced vehicles"""
    st.header("🎯 Deal Alerts & Opportunities")

    # Underpriced vehicles
    st.subheader("🔥 Potentially Underpriced Vehicles")

    underpriced = intelligence.detect_underpriced_vehicles()
    if underpriced:
        # Create dataframe for display
        deals_df = pd.DataFrame(underpriced)
        deals_df['Savings'] = deals_df['savings_potential'].str.replace('PKR', '').str.replace(',', '').astype(float)

        # Sort by potential savings
        deals_df = deals_df.sort_values('Savings', ascending=False)

        # Display top deals
        for _, deal in deals_df.head(5).iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**{deal['title']}**")
                    st.caption(f"📍 {deal['city']}")

                with col2:
                    st.markdown(f"💰 {deal['price']}")
                    st.caption(f"Make: {deal['make']}")

                with col3:
                    st.success(f"💸 Save {deal['savings_potential']}")

                st.divider()
    else:
        st.info("No underpriced vehicles detected at this time.")

    # Price change alerts
    st.subheader("📈 Recent Price Changes")

    changes = intelligence.get_recent_price_changes()
    if changes:
        for change in changes[:10]:  # Show last 10 changes
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**{change['title']}**")
                    st.caption(f"📍 {change['city']}")

                with col2:
                    st.markdown(f"🔄 {change['old_price']} → {change['current_price']}")

                with col3:
                    st.caption(f"📅 {change['changed_at'][:19]}")

                st.divider()
    else:
        st.info("No recent price changes detected.")

@st.cache_data(ttl=300)
def show_listings_table():
    """Display interactive listings table"""
    st.header("📋 Car Listings Database")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        make_filter = st.selectbox(
            "Filter by Make",
            ["All"] + sorted(intelligence.get_listings_dataframe()['make'].dropna().unique().tolist())
        )

    with col2:
        city_filter = st.selectbox(
            "Filter by City",
            ["All"] + sorted(intelligence.get_listings_dataframe()['city'].dropna().unique().tolist())
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First", "Price: Low to High", "Price: High to Low"]
        )

    # Get filtered data
    df = intelligence.get_listings_dataframe()

    # Apply filters
    if make_filter != "All":
        df = df[df['make'] == make_filter]

    if city_filter != "All":
        df = df[df['city'] == city_filter]

    # Apply sorting
    if sort_by == "Newest First":
        df = df.sort_values('first_seen', ascending=False)
    elif sort_by == "Oldest First":
        df = df.sort_values('first_seen', ascending=True)
    elif sort_by == "Price: Low to High":
        df = df.sort_values('price_numeric', na_position='last')
    elif sort_by == "Price: High to Low":
        df = df.sort_values('price_numeric', ascending=False, na_position='last')

    # Clean up display columns
    display_df = df[['title', 'price', 'city', 'make', 'model', 'year', 'first_seen']].copy()
    display_df['first_seen'] = pd.to_datetime(display_df['first_seen']).dt.strftime('%Y-%m-%d %H:%M')
    display_df.columns = ['Title', 'Price', 'City', 'Make', 'Model', 'Year', 'First Seen']

    # Display table
    st.dataframe(display_df, use_container_width=True)

    st.caption(f"Showing {len(display_df)} listings")

@st.cache_data(ttl=300)
def show_analytics_charts():
    """Display detailed analytics and charts"""
    st.header("📊 Market Analytics")

    df = intelligence.get_listings_dataframe()

    if df.empty:
        st.warning("No data available for analytics.")
        return

    # Price distribution
    st.subheader("Price Distribution")

    # Clean price data
    df['price_numeric'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    df = df.dropna(subset=['price_numeric'])

    if not df.empty:
        fig = px.histogram(
            df,
            x='price_numeric',
            nbins=30,
            title='Car Price Distribution',
            labels={'price_numeric': 'Price (PKR)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Price by make box plot
        if 'make' in df.columns and df['make'].notna().any():
            st.subheader("Price Distribution by Make")

            # Filter to top makes
            top_makes = df['make'].value_counts().head(5).index
            filtered_df = df[df['make'].isin(top_makes)]

            fig = px.box(
                filtered_df,
                x='make',
                y='price_numeric',
                title='Price Distribution by Top 5 Makes',
                labels={'price_numeric': 'Price (PKR)', 'make': 'Make'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Market intelligence summary
    st.subheader("Intelligence Summary")

    report = intelligence.generate_market_report()

    # Display key insights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Vehicles Analyzed", len(df))

    with col2:
        underpriced_count = len(report['underpriced_vehicles'])
        st.metric("Potential Deals Found", underpriced_count)

    with col3:
        alerts_count = len(report['alerts'])
        st.metric("Active Alerts", alerts_count)

    # Recent activity
    st.subheader("Recent Activity")

    if report['price_changes']:
        changes_df = pd.DataFrame(report['price_changes'])
        changes_df['changed_at'] = pd.to_datetime(changes_df['changed_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(changes_df[['title', 'current_price', 'old_price', 'changed_at']], use_container_width=True)
    else:
        st.info("No recent price changes")

def show_price_predictor():
    """Interactive price prediction tool"""
    st.header("🔮 AI Price Predictor")
    st.markdown("Get accurate price predictions for any vehicle using our advanced ML models")

    # Import AI models
    from scripts.ai_models import PricePredictionModel
    predictor = PricePredictionModel()

    # Check if model is available
    if not predictor.load_model():
        st.warning("⚠️ AI model not trained yet. Train the model first using the training button below.")

        if st.button("🚀 Train AI Model", type="primary"):
            with st.spinner("Training AI model... This may take a few moments."):
                if predictor.train_model():
                    st.success("✅ AI model trained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to train model. Check console for details.")
        return

    # Prediction form
    st.subheader("Vehicle Details")

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox(
            "Car Make",
            ["Toyota", "Honda", "Suzuki", "Hyundai", "KIA", "Nissan", "Mercedes", "BMW", "Audi", "MG", "Other"],
            help="Select the car manufacturer"
        )

        model = st.text_input(
            "Car Model",
            placeholder="e.g., Corolla, Civic, Cultus",
            help="Enter the specific model name"
        )

        year = st.slider(
            "Manufacturing Year",
            min_value=1990,
            max_value=datetime.now().year,
            value=2020,
            help="Year the vehicle was manufactured"
        )

    with col2:
        city = st.selectbox(
            "City",
            ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta"],
            help="City where the vehicle is located"
        )

        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=5000,
            help="Current mileage of the vehicle"
        )

        condition = st.selectbox(
            "Condition",
            ["Excellent", "Good", "Fair", "Poor"],
            help="Overall condition of the vehicle"
        )

    # Prediction button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data and predicting price..."):
            try:
                # Get prediction
                predicted_price = predictor.predict_price(make, model, year, city, mileage)

                if predicted_price:
                    # Display results
                    st.success("✅ Price prediction completed!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Predicted Market Value",
                            f"PKR {predicted_price:,.0f}",
                            help="Estimated fair market value based on AI analysis"
                        )

                    with col2:
                        # Calculate price range
                        lower_bound = predicted_price * 0.9
                        upper_bound = predicted_price * 1.1
                        st.metric(
                            "Expected Range",
                            f"PKR {lower_bound:,.0f} - {upper_bound:,.0f}",
                            help="90% confidence interval"
                        )

                    with col3:
                        # Compare with market average
                        averages = intelligence.calculate_average_prices()
                        market_avg = averages.get('overall_avg', 0)
                        diff_pct = ((predicted_price - market_avg) / market_avg) * 100 if market_avg > 0 else 0

                        st.metric(
                            "vs Market Average",
                            f"{diff_pct:+.1f}%",
                            delta=f"PKR {predicted_price - market_avg:,.0f}",
                            help="Comparison with overall market average"
                        )

                    # Additional insights
                    st.subheader("📊 Prediction Insights")

                    # Market comparison
                    market_comparison = intelligence.calculate_average_prices()
                    if 'by_make' in market_comparison and make in market_comparison['by_make'].index:
                        make_avg = market_comparison['by_make'].loc[make, 'mean']
                        make_diff = ((predicted_price - make_avg) / make_avg) * 100
                        st.info(f"Compared to other {make} vehicles: {make_diff:+.1f}% {'above' if make_diff > 0 else 'below'} average")

                    # Age factor
                    current_year = datetime.now().year
                    age = current_year - year
                    if age <= 2:
                        st.success("🚗 This is a relatively new vehicle - great for resale value!")
                    elif age <= 5:
                        st.info("📅 Moderately aged vehicle - balanced price consideration")
                    else:
                        st.warning("⏰ Older vehicle - consider maintenance costs in total ownership")

                    # Mileage analysis
                    if mileage < 30000:
                        st.success("🛣️  Low mileage - excellent condition indicator!")
                    elif mileage < 80000:
                        st.info("📊 Moderate mileage - reasonable for age")
                    else:
                        st.warning("🔧 High mileage - factor in potential maintenance costs")

                else:
                    st.error("❌ Unable to generate prediction. Please check your inputs.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("💡 Try adjusting the vehicle details or check if the AI model needs retraining.")

    # Historical predictions (if any)
    st.subheader("📈 Recent Predictions")
    st.info("Recent price predictions will be displayed here for comparison.")

def show_advanced_tools():
    """Advanced analysis tools and features"""
    st.header("📈 Advanced Market Intelligence Tools")

    # Tool selector
    tool = st.selectbox(
        "Select Tool",
        ["Trend Analysis", "Arbitrage Calculator", "Deal Comparator", "Market Scanner", "Export Data"]
    )

    if tool == "Trend Analysis":
        show_trend_analysis_tool()
    elif tool == "Arbitrage Calculator":
        show_arbitrage_calculator()
    elif tool == "Deal Comparator":
        show_deal_comparator()
    elif tool == "Market Scanner":
        show_market_scanner()
    elif tool == "Export Data":
        show_data_export()

def show_trend_analysis_tool():
    """Interactive trend analysis tool"""
    st.subheader("📊 Price Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Select Make", ["All"] + sorted(intelligence.get_listings_dataframe()['make'].dropna().unique().tolist()))
        days = st.slider("Analysis Period (days)", 7, 90, 30)

    with col2:
        model = st.selectbox("Select Model (optional)", ["All"] + sorted(intelligence.get_listings_dataframe()['model'].dropna().unique().tolist()) if make != "All" else ["All"])
        city = st.selectbox("Select City (optional)", ["All"] + sorted(intelligence.get_listings_dataframe()['city'].dropna().unique().tolist()))

    if st.button("📈 Analyze Trends"):
        from scripts.ai_models import PriceTrendForecaster
        forecaster = PriceTrendForecaster()

        trend_data = forecaster.forecast_price_trend(
            make=make if make != "All" else None,
            model=model if model != "All" else None,
            city=city if city != "All" else None,
            forecast_days=14
        )

        if trend_data['trend'] != 'insufficient_data':
            # Display trend results
            col1, col2, col3 = st.columns(3)

            with col1:
                trend_emoji = "📈" if trend_data['trend'] == 'increasing' else "📉" if trend_data['trend'] == 'decreasing' else "➡️"
                st.metric("Trend Direction", f"{trend_emoji} {trend_data['trend'].title()}")

            with col2:
                st.metric("Confidence Level", f"{trend_data['confidence']:.1%}")

            with col3:
                st.metric("Current Average", f"PKR {trend_data['current_avg']:,.0f}")

            # Forecast chart
            if trend_data['forecast']:
                forecast_df = pd.DataFrame(trend_data['forecast'])
                fig = px.line(
                    forecast_df,
                    x='date',
                    y='predicted_price',
                    title=f'Price Forecast for {make} vehicles',
                    labels={'predicted_price': 'Predicted Price (PKR)', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)

            st.info(f"📊 Analysis based on {trend_data['data_points']} data points over {trend_data['period_days']} days")
        else:
            st.warning(trend_data['message'])

def show_arbitrage_calculator():
    """Arbitrage opportunity calculator"""
    st.subheader("💰 Arbitrage Calculator")

    make = st.selectbox("Select Car Make", sorted(intelligence.get_listings_dataframe()['make'].dropna().unique().tolist()))

    if st.button("🔍 Find Arbitrage Opportunities"):
        from scripts.ai_models import ArbitrageAnalyzer
        analyzer = ArbitrageAnalyzer()

        opportunities = analyzer.analyze_city_arbitrage(make)

        if opportunities:
            st.success(f"Found {len(opportunities)} arbitrage opportunities for {make} vehicles!")

            for opp in opportunities:
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"**{opp['buy_city']} → {opp['sell_city']}**")
                        st.caption(f"{opp['make']} {opp['model'] or 'vehicles'}")

                    with col2:
                        st.markdown(f"Buy: {opp['buy_price']}")
                        st.markdown(f"Sell: {opp['sell_price']}")

                    with col3:
                        st.markdown(f"Transport: {opp['transport_cost']}")
                        st.success(f"Profit: {opp['profit_potential']}")

                    with col4:
                        st.metric("Margin", opp['profit_margin'], delta=opp['confidence'].title())

                    st.divider()
        else:
            st.info("No arbitrage opportunities found for the selected make.")

def show_deal_comparator():
    """Compare multiple deals side by side"""
    st.subheader("⚖️ Deal Comparator")

    # Get available deals
    deals = intelligence.detect_underpriced_vehicles()

    if not deals:
        st.info("No deals available for comparison.")
        return

    # Select deals to compare
    deal_titles = [deal['title'] for deal in deals[:10]]  # Top 10 deals

    selected_deals = st.multiselect(
        "Select deals to compare",
        deal_titles,
        max_selections=4,
        help="Choose up to 4 deals to compare side by side"
    )

    if selected_deals:
        # Create comparison table
        comparison_data = []
        for deal in deals:
            if deal['title'] in selected_deals:
                savings = float(deal['savings_potential'].replace('PKR', '').replace(',', ''))
                price = float(deal['price'].replace('PKR', '').replace(',', ''))

                comparison_data.append({
                    'Vehicle': deal['title'],
                    'Price': f"PKR {price:,.0f}",
                    'City': deal['city'],
                    'Make': deal['make'],
                    'Savings': f"PKR {savings:,.0f}",
                    'Value Score': savings / price if price > 0 else 0
                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

            # Best deal highlight
            best_deal = max(comparison_data, key=lambda x: x['Value Score'])
            st.success(f"🏆 Best Value: {best_deal['Vehicle']} - {best_deal['Savings']} savings!")

def show_market_scanner():
    """Advanced market scanning tool"""
    st.subheader("🔍 Market Scanner")

    col1, col2, col3 = st.columns(3)

    with col1:
        price_range = st.slider("Price Range (PKR)", 0, 20000000, (100000, 5000000))
        make_filter = st.selectbox("Make Filter", ["All"] + sorted(intelligence.get_listings_dataframe()['make'].dropna().unique().tolist()))

    with col2:
        year_range = st.slider("Year Range", 1990, datetime.now().year, (2015, datetime.now().year))
        city_filter = st.selectbox("City Filter", ["All"] + sorted(intelligence.get_listings_dataframe()['city'].dropna().unique().tolist()))

    with col3:
        sort_option = st.selectbox("Sort By", ["Price: Low to High", "Price: High to Low", "Newest First", "Best Deals"])
        show_count = st.slider("Show Results", 10, 100, 25)

    if st.button("🔍 Scan Market"):
        df = intelligence.get_listings_dataframe()

        # Apply filters
        df['price_numeric'] = df['price'].str.replace('PKR', '').str.replace(',', '').astype(float)
        df = df[
            (df['price_numeric'] >= price_range[0]) &
            (df['price_numeric'] <= price_range[1]) &
            (df['year'] >= year_range[0]) &
            (df['year'] <= year_range[1])
        ]

        if make_filter != "All":
            df = df[df['make'] == make_filter]

        if city_filter != "All":
            df = df[df['city'] == city_filter]

        # Apply sorting
        if sort_option == "Price: Low to High":
            df = df.sort_values('price_numeric')
        elif sort_option == "Price: High to Low":
            df = df.sort_values('price_numeric', ascending=False)
        elif sort_option == "Newest First":
            df = df.sort_values('first_seen', ascending=False)
        elif sort_option == "Best Deals":
            # Simple deal detection based on price percentiles
            if len(df) > 5:
                price_threshold = df['price_numeric'].quantile(0.25)
                df['is_deal'] = df['price_numeric'] <= price_threshold
                df = df.sort_values(['is_deal', 'price_numeric'], ascending=[False, True])

        # Display results
        if not df.empty:
            st.success(f"Found {len(df)} vehicles matching your criteria")

            display_df = df.head(show_count)[['title', 'price', 'city', 'make', 'model', 'year']].copy()
            display_df.columns = ['Vehicle', 'Price', 'City', 'Make', 'Model', 'Year']

            st.dataframe(display_df, use_container_width=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price", f"PKR {df['price_numeric'].mean():,.0f}")
            with col2:
                st.metric("Price Range", f"PKR {df['price_numeric'].min():,.0f} - {df['price_numeric'].max():,.0f}")
            with col3:
                st.metric("Total Results", len(df))
        else:
            st.info("No vehicles found matching your criteria. Try adjusting the filters.")

def show_data_export():
    """Data export functionality"""
    st.subheader("📤 Export Market Data")

    export_type = st.selectbox(
        "Export Type",
        ["Listings Data", "Price History", "Market Averages", "Deal Analysis"]
    )

    format_type = st.selectbox("Format", ["CSV", "Excel", "JSON"])

    if st.button("📥 Generate Export"):
        try:
            if export_type == "Listings Data":
                df = intelligence.get_listings_dataframe()
            elif export_type == "Price History":
                # Get price history from database
                import sqlite3
                with sqlite3.connect('data/autointel.db') as conn:
                    df = pd.read_sql_query("SELECT * FROM price_history", conn)
            elif export_type == "Market Averages":
                averages = intelligence.calculate_average_prices()
                df = pd.DataFrame(averages)
            elif export_type == "Deal Analysis":
                deals = intelligence.detect_underpriced_vehicles()
                df = pd.DataFrame(deals)

            if not df.empty:
                # Create download button
                if format_type == "CSV":
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"autointel_{export_type.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                elif format_type == "Excel":
                    # Note: This would require openpyxl, but we'll show CSV for now
                    st.info("Excel export requires additional dependencies. Use CSV format instead.")
                elif format_type == "JSON":
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"autointel_{export_type.lower().replace(' ', '_')}.json",
                        mime="application/json"
                    )

                st.success(f"✅ Export ready! {len(df)} records prepared for download.")
                st.dataframe(df.head(), use_container_width=True)
            else:
                st.warning("No data available for export.")

        except Exception as e:
            st.error(f"Export failed: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("*AutoIntel - Transforming car buying with AI-powered market intelligence*")

if __name__ == "__main__":
    main()
