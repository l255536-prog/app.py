

def train_and_predict(data: pd.DataFrame, days: int) -> PredictionResult:
    prepared = add_features(data)
    features = ["Open", "High", "Low", "Close", "Volume", "MA_7", "MA_21", "Volatility_7"]

    x = prepared[features]
    y = prepared["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    forecast_rows = []
    rolling_data = data.copy()
    last_date = pd.to_datetime(rolling_data["Date"].iloc[-1])

    for step in range(days):
        next_features = add_features(rolling_data).iloc[-1][features]
        predicted_close = float(model.predict(pd.DataFrame([next_features]))[0])

        previous_row = rolling_data.iloc[-1].copy()
        next_date = last_date + timedelta(days=step + 1)
        previous_close = float(previous_row["Close"])

        forecast_rows.append(
            {
                "Date": next_date,
                "Predicted Close": predicted_close,
            }
        )

        synthetic_row = {
            "Date": next_date,
            "Open": previous_close,
            "High": max(previous_close, predicted_close),
            "Low": min(previous_close, predicted_close),
            "Close": predicted_close,
            "Adj Close": predicted_close,
            "Volume": previous_row["Volume"],
        }
        rolling_data = pd.concat(
            [rolling_data, pd.DataFrame([synthetic_row])], ignore_index=True
        )

    return PredictionResult(
        model=model,
        mae=mae,
        rmse=rmse,
        forecast=pd.DataFrame(forecast_rows),
        prepared_data=prepared,
    )


def price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close")
    )
    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"].rolling(20).mean(),
            mode="lines",
            name="20-day average",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"].rolling(50).mean(),
            mode="lines",
            name="50-day average",
        )
    )
    figure.update_layout(
        title=f"{ticker.upper()} historical close price",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    return figure


def forecast_chart(data: pd.DataFrame, forecast: pd.DataFrame, ticker: str) -> go.Figure:
    recent = data.tail(120)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=recent["Date"], y=recent["Close"], mode="lines", name="Actual")
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Predicted Close"],
            mode="lines+markers",
            name="Forecast",
        )
    )
    figure.update_layout(
        title=f"{ticker.upper()} forecast",
        xaxis_title="Date",
        yaxis_title="Predicted price",
        hovermode="x unified",
    )
    return figure


def main() -> None:
    st.title("Stock Prediction Application")
    st.caption("Machine learning forecast using historical Yahoo Finance data")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
        years = st.slider("Historical years", min_value=1, max_value=10, value=5)
        forecast_days = st.slider("Forecast days", min_value=1, max_value=30, value=7)
        run_button = st.button("Run prediction", type="primary")

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    if run_button:
        with st.spinner(f"Downloading {ticker} data and training model..."):
            data = download_stock_data(ticker, start_date, end_date)

            if data.empty or len(data) < 80:
                st.error("Not enough data found. Try another ticker or a longer date range.")
                return

            result = train_and_predict(data, forecast_days)

        latest_close = float(data["Close"].iloc[-1])
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest close", f"${latest_close:,.2f}")
        col2.metric("MAE", f"${result.mae:,.2f}")
        col3.metric("RMSE", f"${result.rmse:,.2f}")

        st.plotly_chart(price_chart(data, ticker), use_container_width=True)
        st.plotly_chart(forecast_chart(data, result.forecast, ticker), use_container_width=True)

        st.subheader("Forecast")
        st.dataframe(
            result.forecast.assign(
                Date=result.forecast["Date"].dt.strftime("%Y-%m-%d"),
                **{"Predicted Close": result.forecast["Predicted Close"].map("${:,.2f}".format)},
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Model details"):
            st.write(
                "The model uses open, high, low, close, volume, moving averages, "
                "and recent volatility to predict the next closing price."
            )
            st.write(
                "This is an educational forecast, not financial advice. Real stock "
                "prices can change suddenly because of news and market conditions."
            )
    else:
        st.info("Choose a ticker and press Run prediction.")


if __name__ == "__main__":
    main()

Latest turn








