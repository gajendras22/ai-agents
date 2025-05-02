import logging
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, EmbeddedResource
from typing import Any, Sequence
from weather_api import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sse-mcp-weather-server")


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

class WeatherServer:
    def __init__(self):
        logger.debug("Initializing WeatherServer")
        self.app = Server("weather-mcp-server")
        self.setup_tools()

    def setup_tools(self):
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="fetch_weather_alerts",
                    description="Fetch weather alerts for a given US state",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "state_code": {
                                "type": "string",
                                "description": "Two-letter US state code (e.g., CA, NY)"
                            }
                        },
                        "required": ["state_code"]
                    }
                ),
                Tool(
                    name="fetch_weather_forecast",
                    description="Fetch weather forecast for a specific location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Latitude of the location"
                            },
                            "lon": {
                                "type": "number",
                                "description": "Longitude of the location"
                            }
                        },
                        "required": ["lat", "lon"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | EmbeddedResource]:
            if name not in ["fetch_weather_alerts", "fetch_weather_forecast"]:
                logger.error(f"Unknown tool: {name}")
                raise ValueError(f"Unknown tool: {name}")

            if not isinstance(arguments, dict):
                logger.error(f"Invalid arguments: {arguments} is not a 'dict'")
                raise ValueError(f"Invalid arguments: {arguments} is not a 'dict'")

            try:
                if name == "fetch_weather_alerts":
                    state_code = arguments["state_code"]
                    logger.debug(f"Fetching weather alerts for state: {state_code}")
                    alerts = await fetch_weather_alerts(state_code)
                    return [TextContent(type="text", text=alerts)]

                elif name == "fetch_weather_forecast":
                    lat = arguments["lat"]
                    lon = arguments["lon"]
                    logger.debug(f"Fetching weather forecast for location: ({lat}, {lon})")
                    forecast = await fetch_weather_forecast(lat, lon)
                    return [TextContent(type="text", text=forecast)]

            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                raise RuntimeError(f"Weather API error: {str(e)}")


def create_app():
    weather_server = WeatherServer()
    sse = SseServerTransport("/weather")

    class HandleSSE:
        def __init__(self, sse, weather_server):
            self.sse = sse
            self.weather_server = weather_server

        async def __call__(self, scope, receive, send):
            async with self.sse.connect_sse(scope, receive, send) as streams:
                await self.weather_server.app.run(
                    streams[0],
                    streams[1],
                    self.weather_server.app.create_initialization_options()
                )

    class HandleMessages:
        def __init__(self, sse):
            self.sse = sse

        async def __call__(self, scope, receive, send):
            await self.sse.handle_post_message(scope, receive, send)

    routes = [
        Route("/sse", endpoint=HandleSSE(sse, weather_server), methods=["GET"]),
        Route("/weather", endpoint=HandleMessages(sse), methods=["POST"])
    ]

    return Starlette(routes=routes)


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=3001)
 