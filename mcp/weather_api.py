from typing import Any
import httpx


NWS_API_BASE_URL = "https://api.weather.gov"
USER_AGENT_HEADER = "weather-app/1.0"

async def fetch_nws_data(api_url: str) -> dict[str, Any] | None:
    """
    Fetch data from the National Weather Service (NWS) API with error handling.

    Args:
        api_url: The URL to fetch data from.

    Returns:
        A dictionary containing the API response, or None if an error occurs.
    """
    headers = {
        "User-Agent": USER_AGENT_HEADER,
        "Accept": "application/geo+json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_weather_alert(alert_feature: dict) -> str:
    """
    Format a weather alert feature into a readable string.

    Args:
        alert_feature: A dictionary containing alert properties.

    Returns:
        A formatted string describing the alert.
    """
    properties = alert_feature["properties"]
    return f"""
Event: {properties.get('event', 'Unknown')}
Area: {properties.get('areaDesc', 'Unknown')}
Severity: {properties.get('severity', 'Unknown')}
Description: {properties.get('description', 'No description available')}
Instructions: {properties.get('instruction', 'No specific instructions provided')}
"""