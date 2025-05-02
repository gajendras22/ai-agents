from mcp.server.fastmcp import FastMCP

from weather_api import *

# Initialize FastMCP server
weather_mcp_server = FastMCP("weather")


@weather_mcp_server.tool()
async def fetch_weather_alerts(state_code: str) -> str:
    """
    Fetch weather alerts for a given US state.

    Args:
        state_code: Two-letter US state code (e.g., CA, NY).

    Returns:
        A string containing formatted weather alerts or an error message.
    """
    alerts_url = f"{NWS_API_BASE_URL}/alerts/active/area/{state_code}"
    alert_data = await fetch_nws_data(alerts_url)

    if not alert_data or "features" not in alert_data:
        return "Unable to fetch alerts or no alerts found."

    if not alert_data["features"]:
        return "No active alerts for this state."

    formatted_alerts = [format_weather_alert(feature) for feature in alert_data["features"]]
    return "\n---\n".join(formatted_alerts)

@weather_mcp_server.tool()
async def fetch_weather_forecast(lat: float, lon: float) -> str:
    """
    Fetch weather forecast for a specific location.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        A string containing the weather forecast or an error message.
    """
    # Fetch the forecast grid endpoint
    points_url = f"{NWS_API_BASE_URL}/points/{lat},{lon}"
    points_data = await fetch_nws_data(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Extract the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await fetch_nws_data(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the forecast periods into a readable string
    forecast_periods = forecast_data["properties"]["periods"]
    formatted_forecasts = []

    for period in forecast_periods[:5]:  # Limit to the next 5 periods
        formatted_forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        formatted_forecasts.append(formatted_forecast)

    return "\n---\n".join(formatted_forecasts)

if __name__ == "__main__":
    # Start the FastMCP server
    weather_mcp_server.run(transport='stdio')