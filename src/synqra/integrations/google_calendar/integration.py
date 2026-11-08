"""Google Calendar integration connector for SYNQRA.

Encapsulates calendar listing, event retrieval, event scheduling, event updates,
event deletions, and free/busy search via Composio's session-based API.
"""

import logging

from fastmcp import FastMCP

from synqra.core.config import Settings, get_settings
from synqra.integrations.base import BaseIntegration
from synqra.integrations.google_calendar.client import GoogleCalendarClient
from synqra.integrations.google_calendar.schemas import (
    CalendarEventDetails,
    CreateCalendarEventOutput,
    DeleteCalendarEventOutput,
    FindFreeTimeOutput,
    ListCalendarEventsOutput,
    ListCalendarsOutput,
    UpdateCalendarEventOutput,
)
from synqra.integrations.google_calendar.tools import (
    create_calendar_event,
    delete_calendar_event,
    find_free_time,
    get_calendar_event,
    list_calendar_events,
    list_calendars,
    update_calendar_event,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)

logger = logging.getLogger("synqra.integrations.google_calendar")


class GoogleCalendarIntegration(BaseIntegration):
    """Production integration connector for Google Calendar via Composio."""

    def __init__(
        self,
        client: GoogleCalendarClient | None = None,
        settings: Settings | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize the Google Calendar integration instance.

        Args:
            client: Optional pre-configured GoogleCalendarClient instance.
            settings: Optional application settings.
            user_id: Optional user identifier for Composio session scoping.
        """
        super().__init__(
            metadata=IntegrationMetadata(
                name="google_calendar",
                display_name="Google Calendar",
                description=(
                    "List calendars, retrieve events, schedule meetings, update appointments, "
                    "delete events with confirmation, and find available free time slots."
                ),
                version="0.1.0",
                auth_required=True,
                auth_type="composio",
                documentation_url="https://developers.google.com/calendar",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or GoogleCalendarClient(
            settings=self._settings,
            user_id=user_id or self._settings.composio_user_id,
        )

    def is_configured(self) -> bool:
        """Check whether Composio credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.composio_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that a valid, active Google Calendar connection exists through Composio."""
        if not self.is_configured():
            return False
        return await self.client.check_connection_async()

    async def health_check(self) -> IntegrationStatus:
        """Return operational health status, providing an authorization URL if unauthenticated."""
        if not self.is_configured():
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.UNCONFIGURED,
                is_configured=False,
                is_connected=False,
                registered_tools=[],
                error_message="Composio API key is not configured.",
                last_checked_at=utc_now(),
            )

        try:
            is_connected = await self.test_connection()
            if is_connected:
                return IntegrationStatus(
                    name=self.name,
                    display_name=self.display_name,
                    state=IntegrationState.CONNECTED,
                    is_configured=True,
                    is_connected=True,
                    registered_tools=self.registered_tools,
                    last_checked_at=utc_now(),
                )

            # Not connected: attempt to generate authorization link
            auth_url: str | None = None
            try:
                auth_url = await self.client.authorize_async()
            except Exception as auth_exc:
                logger.debug("Could not generate Google Calendar auth URL: %s", auth_exc)

            auth_msg = (
                f"Google Calendar OAuth not connected. Authorize at: {auth_url}"
                if auth_url
                else "Google Calendar OAuth not connected. Connect account in Composio."
            )

            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message=auth_msg,
                last_checked_at=utc_now(),
            )
        except Exception as exc:
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message=f"Health probe failed: {exc}",
                last_checked_at=utc_now(),
            )

    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        """Register the 7 curated Google Calendar tools on the FastMCP server."""
        client = self.client

        @server.tool(
            name="list_calendars",
            description=(
                "List Google Calendars accessible to the connected user, including primary "
                "and secondary calendars with access roles and timezone configurations."
            ),
        )
        async def list_calendars_tool(
            max_results: int = 50,
            page_token: str | None = None,
            show_hidden: bool = False,
            min_access_role: str | None = None,
        ) -> ListCalendarsOutput:
            """List accessible Google Calendars."""
            return await list_calendars(
                client=client,
                max_results=max_results,
                page_token=page_token,
                show_hidden=show_hidden,
                min_access_role=min_access_role,
            )

        @server.tool(
            name="list_calendar_events",
            description=(
                "Retrieve calendar events within a specified date/time range. Supports pagination, "
                "timezone-aware formatting, single event expansion, and text queries."
            ),
        )
        async def list_calendar_events_tool(
            calendar_id: str = "primary",
            time_min: str | None = None,
            time_max: str | None = None,
            timezone: str | None = None,
            query: str | None = None,
            max_results: int = 25,
            page_token: str | None = None,
            single_events: bool = True,
            order_by: str | None = "startTime",
        ) -> ListCalendarEventsOutput:
            """List events from a Google Calendar within a time window."""
            return await list_calendar_events(
                client=client,
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                timezone=timezone,
                query=query,
                max_results=max_results,
                page_token=page_token,
                single_events=single_events,
                order_by=order_by,
            )

        @server.tool(
            name="get_calendar_event",
            description=(
                "Retrieve complete details for a specific calendar event using "
                "calendar ID and event ID. Includes attendees, descriptions, "
                "locations, status, reminders, and start/end times."
            ),
        )
        async def get_calendar_event_tool(
            event_id: str,
            calendar_id: str = "primary",
            timezone: str | None = None,
        ) -> CalendarEventDetails:
            """Retrieve full metadata for a single calendar event."""
            return await get_calendar_event(
                client=client,
                event_id=event_id,
                calendar_id=calendar_id,
                timezone=timezone,
            )

        @server.tool(
            name="create_calendar_event",
            description=(
                "Create a new timed or all-day Google Calendar event with title, "
                "description, location, timezone, start/end times, and optional attendees."
            ),
        )
        async def create_calendar_event_tool(
            summary: str,
            start_time: str,
            end_time: str | None = None,
            duration_minutes: int | None = None,
            description: str | None = None,
            location: str | None = None,
            timezone: str = "UTC",
            calendar_id: str = "primary",
            attendees: list[str] | None = None,
            reminders_minutes: list[int] | None = None,
            is_all_day: bool = False,
        ) -> CreateCalendarEventOutput:
            """Create a new Google Calendar event."""
            return await create_calendar_event(
                client=client,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration_minutes,
                description=description,
                location=location,
                timezone=timezone,
                calendar_id=calendar_id,
                attendees=attendees,
                reminders_minutes=reminders_minutes,
                is_all_day=is_all_day,
            )

        @server.tool(
            name="update_calendar_event",
            description=(
                "Update selected fields of an existing calendar event without "
                "unintentionally overwriting other fields (patch semantics). "
                "Supports title, description, location, start/end times, "
                "timezone, and attendees."
            ),
        )
        async def update_calendar_event_tool(
            event_id: str,
            calendar_id: str = "primary",
            summary: str | None = None,
            description: str | None = None,
            location: str | None = None,
            start_time: str | None = None,
            end_time: str | None = None,
            timezone: str | None = None,
            attendees: list[str] | None = None,
        ) -> UpdateCalendarEventOutput:
            """Update specific fields of an existing calendar event."""
            return await update_calendar_event(
                client=client,
                event_id=event_id,
                calendar_id=calendar_id,
                summary=summary,
                description=description,
                location=location,
                start_time=start_time,
                end_time=end_time,
                timezone=timezone,
                attendees=attendees,
            )

        @server.tool(
            name="delete_calendar_event",
            description=(
                "Delete a calendar event from Google Calendar. For safety, this destructive "
                "operation requires explicit confirmation (confirm=True). If confirm is False, "
                "the event will not be deleted."
            ),
        )
        async def delete_calendar_event_tool(
            event_id: str,
            calendar_id: str = "primary",
            confirm: bool = False,
        ) -> DeleteCalendarEventOutput:
            """Delete a calendar event with mandatory confirmation."""
            return await delete_calendar_event(
                client=client,
                event_id=event_id,
                calendar_id=calendar_id,
                confirm=confirm,
            )

        @server.tool(
            name="find_free_time",
            description=(
                "Identify available free time slots within a specified date/time range. "
                "Accounts for existing events, filters by requested minimum meeting duration, "
                "and supports multiple calendar IDs."
            ),
        )
        async def find_free_time_tool(
            time_min: str,
            time_max: str,
            calendar_ids: list[str] | None = None,
            duration_minutes: int = 30,
            timezone: str = "UTC",
        ) -> FindFreeTimeOutput:
            """Find available free time intervals within a date range."""
            return await find_free_time(
                client=client,
                time_min=time_min,
                time_max=time_max,
                calendar_ids=calendar_ids,
                duration_minutes=duration_minutes,
                timezone=timezone,
            )

        return [
            "list_calendars",
            "list_calendar_events",
            "get_calendar_event",
            "create_calendar_event",
            "update_calendar_event",
            "delete_calendar_event",
            "find_free_time",
        ]
