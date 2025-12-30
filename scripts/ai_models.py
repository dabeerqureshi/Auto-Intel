"""
AI Models and Predictive Analytics - Phase 8
Machine learning models for price prediction and market intelligence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PricePredictionModel:
    """Machine learning model for predicting car prices"""

    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'price_numeric'
        self.model_path = 'data/price_prediction_model.pkl'

    def get_training_data(self):
        """Get and prepare training data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM listings", conn)
        except:
            # Fallback to intelligence module
            from scripts.intelligence import MarketIntelligence
            intelligence = MarketIntelligence()
            df = intelligence.get_listings_dataframe()

        if df.empty:
            return pd.DataFrame()

        # Clean and prepare data
        df = df.copy()
        df['price_clean'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
        df['price_numeric'] = pd.to_numeric(df['price_clean'], errors='coerce')
        df = df.dropna(subset=['price_numeric'])

        # Extract year from title if not available
        if 'year' not in df.columns or df['year'].isna().all():
            import re
            df['year'] = df['title'].str.extract(r'\b(19|20)\d{2}\b').astype(float)

        # Fill missing values
        df['year'] = df['year'].fillna(df['year'].median())
        df['mileage'] = df.get('mileage', '0').astype(str).str.extract(r'(\d+)').astype(float).fillna(0)

        # Select features for model
        feature_cols = ['make', 'model', 'year', 'city', 'mileage']
        available_cols = [col for col in feature_cols if col in df.columns]

        df = df[available_cols + [self.target_column]].dropna()

        return df

    def prepare_features(self, df):
        """Prepare features for machine learning"""
        df_processed = df.copy()

        # Encode categorical variables
        categorical_cols = ['make', 'model', 'city']
        for col in categorical_cols:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    # Handle new categories not seen during training
                    try:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                    except ValueError:
                        # For new categories, assign them a default value (e.g., -1)
                        df_processed[col] = -1

        # Scale numerical features
        numerical_cols = ['year', 'mileage']
        available_num_cols = [col for col in numerical_cols if col in df_processed.columns]

        if available_num_cols:
            if self.scaler is None:
                self.scaler = StandardScaler()
            df_processed[available_num_cols] = self.scaler.fit_transform(df_processed[available_num_cols])

        # Filter out unwanted columns (like title which is used for feature extraction)
        exclude_cols = [self.target_column, 'title']  # title is used for extraction but not a feature
        self.feature_columns = [col for col in df_processed.columns if col not in exclude_cols]

        return df_processed

    def train_model(self, use_ensemble=True):
        """Train the price prediction model with enhanced accuracy"""
        df = self.get_training_data()

        if df.empty or len(df) < 5:
            print("Insufficient data for training. Need at least 5 samples.")
            return False

        # Enhanced data preprocessing
        df = self._preprocess_training_data(df)

        if len(df) < 5:
            print("Insufficient data after preprocessing.")
            return False

        # Prepare features
        df_processed = self.prepare_features(df)

        # Split data with stratification if possible
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]

        # Use cross-validation for better evaluation
        from sklearn.model_selection import cross_val_score

        if use_ensemble:
            # Train ensemble of models for better accuracy
            from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
            from sklearn.linear_model import Ridge

            models = {
                'random_forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=0.1)
            }

            best_model = None
            best_score = -float('inf')
            best_name = None

            print("Training ensemble models...")

            for name, model in models.items():
                try:
                    # Cross-validation scoring
                    cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)), scoring='r2')
                    avg_score = cv_scores.mean()

                    print(f"{name}: CV R² = {avg_score:.3f} (+/- {cv_scores.std() * 2:.3f})")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_name = name
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue

            if best_name:
                print(f"\nBest model: {best_name} with CV R² = {best_score:.3f}")
                # Create a fresh instance of the best model
                if best_name == 'random_forest':
                    self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                elif best_name == 'gradient_boosting':
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    self.model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
                elif best_name == 'extra_trees':
                    self.model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                elif best_name == 'ridge':
                    self.model = Ridge(alpha=0.1)

                self.model.fit(X, y)  # Train on full dataset
            else:
                # Fallback to simple model
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X, y)
        else:
            # Simple model training
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)

        # Enhanced evaluation
        evaluation_results = self._evaluate_model(X, y)

        print("\nEnhanced Model Training Results:")
        print(f"MAE: PKR {evaluation_results['mae']:,.0f}")
        print(f"RMSE: PKR {evaluation_results['rmse']:,.0f}")
        print(f"R² Score: {evaluation_results['r2']:.3f}")
        print(f"MAPE: {evaluation_results['mape']:.1f}%")
        print(f"Training samples: {len(X)}")

        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            self._analyze_feature_importance()

        # Save model
        self.save_model()

        return True

    def _preprocess_training_data(self, df):
        """Advanced data preprocessing for better accuracy"""
        df_clean = df.copy()

        # Remove extreme outliers (prices beyond reasonable bounds)
        # Pakistani car market typical range
        df_clean = df_clean[
            (df_clean['price_numeric'] >= 50000) &  # Minimum reasonable price
            (df_clean['price_numeric'] <= 20000000)  # Maximum reasonable price
        ]

        # Remove entries with missing critical information
        critical_cols = ['make', 'year']
        df_clean = df_clean.dropna(subset=critical_cols)

        # Fill mileage with median if missing
        if 'mileage' in df_clean.columns:
            df_clean['mileage'] = df_clean['mileage'].fillna(df_clean['mileage'].median())

        # Extract additional features from title
        df_clean = self._extract_title_features(df_clean)

        # Remove duplicate entries - use available columns
        if 'title' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['title', 'price_numeric'], keep='first')
        else:
            # Use make, model, year, price as unique identifier if title not available
            dedup_cols = ['make', 'model', 'year', 'price_numeric']
            available_dedup_cols = [col for col in dedup_cols if col in df_clean.columns]
            if available_dedup_cols:
                df_clean = df_clean.drop_duplicates(subset=available_dedup_cols, keep='first')

        return df_clean

    def _extract_title_features(self, df):
        """Extract additional features from vehicle titles"""
        df = df.copy()

        # Check if title column exists
        if 'title' not in df.columns:
            # Add default values if no title column
            df['engine_cc'] = 1000.0
            df['likely_petrol'] = 0
            df['likely_diesel'] = 0
            df['likely_hybrid'] = 0
            return df

        # Extract engine capacity (cc)
        import re
        cc_pattern = r'(\d{3,4})\s*cc'
        df['engine_cc'] = df['title'].str.extract(cc_pattern, flags=re.IGNORECASE).astype(float)

        # Extract fuel type hints
        df['likely_petrol'] = df['title'].str.contains(r'\b(petrol|gasoline)\b', case=False).astype(int)
        df['likely_diesel'] = df['title'].str.contains(r'\b(diesel)\b', case=False).astype(int)
        df['likely_hybrid'] = df['title'].str.contains(r'\b(hybrid|electric)\b', case=False).astype(int)

        # Fill missing values
        df['engine_cc'] = df['engine_cc'].fillna(df.groupby('make')['engine_cc'].transform('median'))
        df['engine_cc'] = df['engine_cc'].fillna(1000)  # Default CC

        return df

    def _evaluate_model(self, X, y):
        """Enhanced model evaluation with multiple metrics"""
        from sklearn.model_selection import cross_val_predict

        # Cross-validated predictions
        y_pred = cross_val_predict(self.model, X, y, cv=min(3, len(X)))

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def _analyze_feature_importance(self):
        """Analyze and display feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop Feature Importances:")
            for _, row in importance_df.head(5).iterrows():
                print(".3f")

    def predict_price(self, make, model, year, city, mileage=0):
        """Predict price for a given vehicle"""
        if self.model is None:
            if not self.load_model():
                return None

        # Create input dataframe with all required columns
        input_data = pd.DataFrame({
            'title': [f"{make} {model} {year}"],  # Create a title for feature extraction
            'make': [make],
            'model': [model],
            'year': [year],
            'city': [city],
            'mileage': [mileage]
        })

        # Extract title features (same as training)
        input_data = self._extract_title_features(input_data)

        # Prepare features
        input_processed = self.prepare_features(input_data)

        # Make prediction - only use features that were actually used during training
        available_features = [col for col in self.feature_columns if col in input_processed.columns]
        prediction = self.model.predict(input_processed[available_features])[0]

        return max(0, prediction)  # Ensure non-negative price

    def save_model(self):
        """Save trained model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat()
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoders = model_data['label_encoders']
                self.feature_columns = model_data['feature_columns']

                print(f"Model loaded from {self.model_path}")
                return True
            else:
                print(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class PriceTrendForecaster:
    """Time series forecasting for price trends"""

    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path

    def get_price_history(self, make=None, model=None, city=None, days=90):
        """Get price history data for trend analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT l.make, l.model, l.city, ph.price, ph.recorded_at
                    FROM price_history ph
                    JOIN listings l ON ph.listing_id = l.listing_id
                    WHERE ph.recorded_at >= datetime('now', '-{} days')
                """.format(days)

                if make:
                    query += f" AND l.make = '{make}'"
                if model:
                    query += f" AND l.model = '{model}'"
                if city:
                    query += f" AND l.city = '{city}'"

                query += " ORDER BY ph.recorded_at DESC"

                df = pd.read_sql_query(query, conn)

        except Exception as e:
            print(f"Error fetching price history: {e}")
            return pd.DataFrame()

        # Clean price data
        df['price_clean'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
        df['price_numeric'] = pd.to_numeric(df['price_clean'], errors='coerce')
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])

        return df.dropna(subset=['price_numeric'])

    def forecast_price_trend(self, make, model=None, city=None, forecast_days=30):
        """Forecast price trend using simple moving averages"""
        df = self.get_price_history(make=make, model=model, city=city, days=90)

        if df.empty or len(df) < 7:
            return {
                'trend': 'insufficient_data',
                'confidence': 0,
                'forecast': [],
                'message': 'Not enough historical data for forecasting'
            }

        # Group by date and calculate daily averages
        daily_prices = df.groupby(df['recorded_at'].dt.date)['price_numeric'].mean().reset_index()
        daily_prices = daily_prices.sort_values('recorded_at')

        if len(daily_prices) < 7:
            return {
                'trend': 'insufficient_data',
                'confidence': 0,
                'forecast': [],
                'message': 'Need at least 7 days of data'
            }

        # Calculate trend using linear regression on recent prices
        recent_prices = daily_prices.tail(14)  # Last 2 weeks
        x = np.arange(len(recent_prices))
        y = recent_prices['price_numeric'].values

        if len(y) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

            # Calculate confidence based on data points and trend strength
            confidence = min(0.9, len(recent_prices) / 20) * (1 - abs(slope) / (np.std(y) + 1))

            # Simple forecast for next days
            forecast = []
            last_price = y[-1]
            for i in range(1, forecast_days + 1):
                forecast_price = last_price + slope * i
                forecast.append({
                    'day': i,
                    'predicted_price': max(0, forecast_price),
                    'date': (daily_prices['recorded_at'].max() + timedelta(days=i)).strftime('%Y-%m-%d')
                })
        else:
            trend_direction = 'stable'
            confidence = 0.1
            forecast = []

        return {
            'trend': trend_direction,
            'confidence': round(confidence, 2),
            'current_avg': round(daily_prices['price_numeric'].mean(), 0),
            'forecast': forecast,
            'data_points': len(daily_prices),
            'period_days': (daily_prices['recorded_at'].max() - daily_prices['recorded_at'].min()).days
        }

class ArbitrageAnalyzer:
    """Analyze price differences between cities for arbitrage opportunities"""

    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path
        self.transport_costs = {
            'karachi_lahore': 15000,
            'karachi_islamabad': 20000,
            'lahore_islamabad': 10000,
            'karachi_faisalabad': 12000,
            'lahore_faisalabad': 8000,
        }

    def analyze_city_arbitrage(self, make, model=None, min_profit_margin=0.05):
        """Find arbitrage opportunities between cities"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT make, model, city, price,
                           CAST(REPLACE(REPLACE(price, 'PKR', ''), ',', '') AS INTEGER) as price_numeric
                    FROM listings
                    WHERE make = ?
                """

                params = [make]
                if model:
                    query += " AND model = ?"
                    params.append(model)

                df = pd.read_sql_query(query, conn, params=params)

        except Exception as e:
            print(f"Error fetching arbitrage data: {e}")
            return []

        if df.empty:
            return []

        opportunities = []

        # Calculate average prices by city
        city_avg = df.groupby('city')['price_numeric'].agg(['mean', 'count', 'min', 'max']).round(0)

        # Find arbitrage opportunities
        cities = city_avg.index.tolist()

        for i, city1 in enumerate(cities):
            for city2 in cities[i+1:]:
                if city1 == city2:
                    continue

                price1 = city_avg.loc[city1, 'mean']
                price2 = city_avg.loc[city2, 'mean']

                # Calculate price difference
                price_diff = abs(price1 - price2)
                cheaper_city = city1 if price1 < price2 else city2
                expensive_city = city2 if price1 < price2 else city1
                cheaper_price = min(price1, price2)

                # Get transport cost
                route_key = f"{cheaper_city.lower()}_{expensive_city.lower()}"
                transport_cost = self.transport_costs.get(route_key, 15000)  # Default cost

                # Calculate profit potential
                profit = price_diff - transport_cost

                if profit > 0 and (profit / cheaper_price) > min_profit_margin:
                    opportunities.append({
                        'make': make,
                        'model': model,
                        'buy_city': cheaper_city,
                        'sell_city': expensive_city,
                        'buy_price': f"PKR {cheaper_price:,.0f}",
                        'sell_price': f"PKR {max(price1, price2):,.0f}",
                        'price_difference': f"PKR {price_diff:,.0f}",
                        'transport_cost': f"PKR {transport_cost:,.0f}",
                        'profit_potential': f"PKR {profit:,.0f}",
                        'profit_margin': f"{(profit / cheaper_price * 100):.1f}%",
                        'confidence': 'high' if city_avg.loc[cheaper_city, 'count'] > 5 else 'medium'
                    })

        # Sort by profit potential
        opportunities.sort(key=lambda x: float(x['profit_potential'].replace('PKR', '').replace(',', '')), reverse=True)

        return opportunities[:10]  # Top 10 opportunities

class DealDetectionEngine:
    """Advanced deal detection using ML and statistical methods"""

    def __init__(self, db_path='data/autointel.db'):
        self.db_path = db_path
        self.price_predictor = PricePredictionModel(db_path)

    def find_statistical_deals(self, threshold_percentile=15):
        """Find deals using statistical analysis (underpriced vehicles)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM listings", conn)
        except:
            from scripts.intelligence import MarketIntelligence
            intelligence = MarketIntelligence()
            df = intelligence.get_listings_dataframe()

        if df.empty:
            return []

        # Clean price data
        df['price_clean'] = df['price'].str.replace('PKR', '').str.replace(',', '').str.strip()
        df['price_numeric'] = pd.to_numeric(df['price_clean'], errors='coerce')
        df = df.dropna(subset=['price_numeric'])

        deals = []

        # Analyze by make
        for make in df['make'].dropna().unique():
            make_data = df[df['make'] == make]

            if len(make_data) < 5:  # Need minimum samples
                continue

            # Calculate price percentiles
            price_threshold = make_data['price_numeric'].quantile(threshold_percentile / 100)

            # Find deals below threshold
            deal_vehicles = make_data[make_data['price_numeric'] <= price_threshold]

            for _, vehicle in deal_vehicles.iterrows():
                deals.append({
                    'type': 'statistical',
                    'title': vehicle['title'],
                    'price': vehicle['price'],
                    'city': vehicle['city'],
                    'make': vehicle['make'],
                    'model': vehicle.get('model', ''),
                    'reason': f'Below {threshold_percentile}th percentile for {make}',
                    'savings_potential': f"PKR {(price_threshold - vehicle['price_numeric']):,.0f}",
                    'confidence': 'high',
                    'detection_method': 'statistical_analysis'
                })

        return deals

    def find_ml_deals(self):
        """Find deals using machine learning predictions"""
        if not self.price_predictor.load_model():
            print("ML model not available. Train model first.")
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM listings", conn)
        except:
            from scripts.intelligence import MarketIntelligence
            intelligence = MarketIntelligence()
            df = intelligence.get_listings_dataframe()

        if df.empty:
            return []

        deals = []

        for _, vehicle in df.iterrows():
            # Prepare vehicle data for prediction
            make = vehicle.get('make', '')
            model = vehicle.get('model', '')
            year = vehicle.get('year', vehicle['title'] if 'title' in vehicle else '')
            city = vehicle.get('city', '')
            mileage = vehicle.get('mileage', 0)

            # Extract year if needed
            if not year or str(year) == 'nan':
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', str(vehicle.get('title', '')))
                year = int(year_match.group()) if year_match else 2020

            # Get predicted price
            predicted_price = self.price_predictor.predict_price(make, model, year, city, mileage)

            if predicted_price:
                actual_price = vehicle['price']
                if isinstance(actual_price, str):
                    actual_price = float(actual_price.replace('PKR', '').replace(',', '').strip())

                price_diff = predicted_price - actual_price

                if price_diff > predicted_price * 0.1:  # At least 10% below predicted
                    deals.append({
                        'type': 'ml_prediction',
                        'title': vehicle.get('title', f"{make} {model} {year}"),
                        'price': f"PKR {actual_price:,.0f}",
                        'predicted_price': f"PKR {predicted_price:,.0f}",
                        'city': city,
                        'make': make,
                        'model': model,
                        'reason': 'Significantly below predicted market value',
                        'savings_potential': f"PKR {price_diff:,.0f}",
                        'confidence': 'medium',
                        'detection_method': 'ml_prediction'
                    })

        return deals

    def get_all_deals(self):
        """Get comprehensive deal analysis"""
        statistical_deals = self.find_statistical_deals()
        ml_deals = self.find_ml_deals()

        all_deals = statistical_deals + ml_deals

        # Remove duplicates and sort by savings potential
        seen_titles = set()
        unique_deals = []

        for deal in all_deals:
            title = deal['title']
            if title not in seen_titles:
                seen_titles.add(title)
                unique_deals.append(deal)

        # Sort by savings potential (extract numeric value)
        def get_savings_value(deal):
            savings_str = deal['savings_potential'].replace('PKR', '').replace(',', '')
            try:
                return float(savings_str)
            except:
                return 0

        unique_deals.sort(key=get_savings_value, reverse=True)

        return unique_deals[:20]  # Top 20 deals

# Import sqlite3 at module level
import sqlite3

def main():
    """Demonstrate AI capabilities"""
    print("=== AutoIntel AI Models Demo ===\n")

    # Initialize components
    predictor = PricePredictionModel()
    forecaster = PriceTrendForecaster()
    arbitrage = ArbitrageAnalyzer()
    deal_detector = DealDetectionEngine()

    # Train model if possible
    print("1. Training Price Prediction Model:")
    if predictor.train_model():
        print("✅ Model trained successfully!\n")
    else:
        print("❌ Could not train model - insufficient data\n")

    # Test price prediction
    print("2. Price Prediction Test:")
    if predictor.model:
        test_price = predictor.predict_price('Toyota', 'Corolla', 2020, 'Karachi', 50000)
        if test_price:
            print(f"Predicted price for 2020 Toyota Corolla in Karachi: PKR {test_price:,.0f}")
        else:
            print("Could not make prediction")
    print()

    # Test trend forecasting
    print("3. Price Trend Forecasting:")
    trend = forecaster.forecast_price_trend('Toyota', forecast_days=7)
    print(f"Trend: {trend.get('trend', 'unknown')}")
    print(f"Confidence: {trend.get('confidence', 0)}")
    print(f"Current average: PKR {trend.get('current_avg', 0):,.0f}")
    print(f"Data points: {trend.get('data_points', 0)}")
    print()

    # Test arbitrage analysis
    print("4. Arbitrage Analysis:")
    opportunities = arbitrage.analyze_city_arbitrage('Toyota')
    if opportunities:
        print(f"Found {len(opportunities)} arbitrage opportunities")
        for opp in opportunities[:3]:
            print(f"  {opp['buy_city']} → {opp['sell_city']}: {opp['profit_potential']} profit")
    else:
        print("No arbitrage opportunities found")
    print()

    # Test deal detection
    print("5. Deal Detection:")
    deals = deal_detector.get_all_deals()
    if deals:
        print(f"Found {len(deals)} potential deals")
        for deal in deals[:3]:
            print(f"  {deal['title']}: {deal['price']} ({deal['reason']})")
    else:
        print("No deals detected")
    print()

    print("=== AI Models Demo Complete ===")

if __name__ == "__main__":
    main()
