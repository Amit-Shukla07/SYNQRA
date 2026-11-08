"""Unit and integration tests for SYNQRA Google Calendar integration.

Verifies GoogleCalendarClient, schema models, tools, and integration health checks using mocked
Composio sessions without requiring live external network credentials.
"""

from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
)
from synqra.integrations.google_calendar.client import (
    GOOGLECALENDAR_CREATE_EVENT_ACTION,
    GOOGLECALENDAR_EVENTS_GET_ACTION,
    GOOGLECALENDAR_EVENTS_LIST_ACTION,
    GOOGLECALENDAR_FIND_FREE_SLOTS_ACTION,
    GOOGLECALENDAR_LIST_CALENDARS_ACTION,
    GOOGLECALENDAR_PATCH_EVENT_ACTION,
    GoogleCalendarClient,
)
from synqra.integrations.google_calendar.integration import GoogleCalendarIntegration
from synqra.integrations.google_calendar.tools import (
    create_calendar_event,
    delete_calendar_event,
    find_free_time,
    get_calendar_event,
    list_calendar_events,
    list_calendars,
    update_calendar_event,
    validate_iana_timezone,
)
from synqra.integrations.models import IntegrationState

# ==============================================================================
# Client & Configuration Tests
# ==============================================================================


def test_google_calendar_client_unconfigured_without_api_key() -> None:
    """Verify GoogleCalendarClient reports unconfigured when no Composio key is provided."""
    empty_settings = Settings()
    client = GoogleCalendarClient(settings=empty_settings)

    assert client.is_configured is False
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client._get_composio_client()
    assert "Composio API key is not configured" in str(exc_info.value)


def test_google_calendar_client_configured_with_api_key() -> None:
    """Verify GoogleCalendarClient reports configured when Composio key is present."""
    client = GoogleCalendarClient(api_key="ak_mock_cal_key_12345")
    assert client.is_configured is True
    assert client.user_id == "synqra_default_user"


def test_google_calendar_client_check_connection_active() -> None:
    """Verify check_connection returns True when toolkit connection is active."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "googlecalendar"
    mock_item.connection = MagicMock(is_active=True)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is True


def test_google_calendar_client_check_connection_inactive() -> None:
    """Verify check_connection returns False when toolkit connection is inactive."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "googlecalendar"
    mock_item.connection = MagicMock(is_active=False)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_google_calendar_client_check_connection_error_returns_false() -> None:
    """Verify check_connection returns False when an unexpected exception occurs."""
    mock_session = MagicMock()
    mock_session.toolkits.side_effect = RuntimeError("Network timeout")

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_google_calendar_client_authorize_generates_redirect_url() -> None:
    """Verify authorize returns redirect URL for OAuth connection."""
    mock_session = MagicMock()
    mock_req = MagicMock(redirect_url="https://connect.composio.dev/auth/googlecalendar-test")
    mock_session.authorize.return_value = mock_req

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    auth_url = client.authorize()

    assert auth_url == "https://connect.composio.dev/auth/googlecalendar-test"
    assert client.authorization_url == auth_url
    mock_session.authorize.assert_called_once_with("googlecalendar", callback_url=None)


def test_google_calendar_client_execute_tool_success() -> None:
    """Verify execute_tool returns execution data on successful execution."""
    mock_session = MagicMock()
    mock_response = MagicMock(result_type="completed", error=None, data={"status": "confirmed"})
    mock_session.execute.return_value = mock_response

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    result = client.execute_tool(GOOGLECALENDAR_CREATE_EVENT_ACTION, {"summary": "Meeting"})

    assert result == {"status": "confirmed"}


@pytest.mark.parametrize(
    ("error_msg", "expected_phrase"),
    [
        ("429 Rate limit exceeded", "rate limit exceeded"),
        ("404 Resource not found", "resource not found"),
        ("403 Insufficient permissions", "Insufficient permissions"),
        ("Invalid date format", "Invalid date/time specification"),
        (
            "no active connection found for toolkit 'googlecalendar'",
            "Google Calendar authorization required",
        ),
    ],
)
def test_google_calendar_client_execute_tool_error_mapping(
    error_msg: str, expected_phrase: str
) -> None:
    """Verify specific error messages are converted to appropriate exceptions."""
    mock_session = MagicMock()
    mock_session.authorize.return_value = MagicMock(redirect_url="https://auth.example.com")
    mock_response = MagicMock(result_type="failed", error=error_msg, data=None)
    mock_session.execute.return_value = mock_response

    client = GoogleCalendarClient(api_key="ak_mock_key", session=mock_session)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        client.execute_tool(GOOGLECALENDAR_EVENTS_LIST_ACTION, {})
    assert expected_phrase.lower() in str(exc_info.value).lower()


# ==============================================================================
# Timezone & Date Validation Tests
# ==============================================================================


def test_validate_iana_timezone_valid() -> None:
    """Verify standard IANA timezones validate successfully."""
    assert validate_iana_timezone("UTC") == "UTC"
    assert validate_iana_timezone("America/New_York") == "America/New_York"
    assert validate_iana_timezone("Europe/London") == "Europe/London"
    assert validate_iana_timezone("Asia/Kolkata") == "Asia/Kolkata"
    assert validate_iana_timezone(None) == "UTC"


def test_validate_iana_timezone_invalid() -> None:
    """Verify invalid timezone strings raise IntegrationExecutionError."""
    with pytest.raises(IntegrationExecutionError) as exc_info:
        validate_iana_timezone("INVALID/NONEXISTENT_TZ")
    assert "Invalid IANA timezone" in str(exc_info.value)


# ==============================================================================
# Tool 1: list_calendars Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_list_calendars_success() -> None:
    """Verify list_calendars normalizes primary and secondary calendars."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "items": [
            {
                "id": "primary",
                "summary": "Personal Calendar",
                "description": "My main calendar",
                "timeZone": "America/New_York",
                "primary": True,
                "accessRole": "owner",
                "backgroundColor": "#ffffff",
                "foregroundColor": "#000000",
            },
            {
                "id": "team@group.calendar.google.com",
                "summary": "Engineering Sync",
                "timeZone": "UTC",
                "primary": False,
                "accessRole": "writer",
            },
        ],
        "nextPageToken": "token_page_2",
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLECALENDAR_LIST_CALENDARS_ACTION:
            return mock_data
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await list_calendars(client, max_results=20, page_token="token_page_1")

    assert result.total_count == 2
    assert result.next_page_token == "token_page_2"
    assert result.calendars[0].id == "primary"
    assert result.calendars[0].primary is True
    assert result.calendars[0].summary == "Personal Calendar"
    assert result.calendars[1].id == "team@group.calendar.google.com"
    assert result.calendars[1].primary is False


# ==============================================================================
# Tool 2: list_calendar_events Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_list_calendar_events_success() -> None:
    """Verify list_calendar_events returns normalized events."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "items": [
            {
                "id": "event_101",
                "summary": "Sprint Planning",
                "description": "Discuss sprint goals",
                "location": "Room 4A",
                "status": "confirmed",
                "htmlLink": "https://calendar.google.com/event?eid=101",
                "start": {"dateTime": "2026-11-09T10:00:00Z"},
                "end": {"dateTime": "2026-11-09T11:00:00Z"},
            },
            {
                "id": "event_102",
                "summary": "Company Holiday",
                "start": {"date": "2026-11-10"},
                "end": {"date": "2026-11-11"},
            },
        ],
        "nextPageToken": "page_2_events",
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLECALENDAR_EVENTS_LIST_ACTION:
            return mock_data
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await list_calendar_events(
        client,
        calendar_id="primary",
        time_min="2026-11-09T00:00:00Z",
        time_max="2026-11-12T00:00:00Z",
        timezone="UTC",
    )

    assert result.total_count == 2
    assert result.events[0].id == "event_101"
    assert result.events[0].summary == "Sprint Planning"
    assert result.events[0].is_all_day is False
    assert result.events[1].id == "event_102"
    assert result.events[1].is_all_day is True
    assert result.next_page_token == "page_2_events"


@pytest.mark.asyncio
async def test_list_calendar_events_invalid_range_raises() -> None:
    """Verify list_calendar_events raises error when time_min is later than time_max."""
    client = MagicMock(spec=GoogleCalendarClient)

    with pytest.raises(IntegrationExecutionError) as exc_info:
        await list_calendar_events(
            client,
            time_min="2026-11-15T00:00:00Z",
            time_max="2026-11-10T00:00:00Z",
        )
    assert "earlier than time_max" in str(exc_info.value)


# ==============================================================================
# Tool 3: get_calendar_event Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_calendar_event_success() -> None:
    """Verify get_calendar_event returns complete details including attendees and reminders."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "id": "evt_999",
        "summary": "1:1 Architecture Review",
        "description": "Quarterly architecture review",
        "location": "https://meet.google.com/abc-def-ghi",
        "status": "confirmed",
        "start": {"dateTime": "2026-11-09T14:00:00Z", "timeZone": "UTC"},
        "end": {"dateTime": "2026-11-09T15:00:00Z", "timeZone": "UTC"},
        "creator": {"email": "lead@example.com"},
        "organizer": {"email": "lead@example.com"},
        "attendees": [
            {
                "email": "dev@example.com",
                "displayName": "Staff Dev",
                "responseStatus": "accepted",
                "optional": False,
            }
        ],
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 15}]},
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO"],
        "created": "2026-11-01T08:00:00Z",
        "updated": "2026-11-05T09:00:00Z",
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLECALENDAR_EVENTS_GET_ACTION:
            return mock_data
        return {}

    client.execute_tool_async.side_effect = mock_execute

    event = await get_calendar_event(client, event_id="evt_999", calendar_id="primary")

    assert event.id == "evt_999"
    assert event.summary == "1:1 Architecture Review"
    assert len(event.attendees) == 1
    assert event.attendees[0].email == "dev@example.com"
    assert event.attendees[0].response_status == "accepted"
    assert event.reminders == ["popup:15m"]
    assert "RRULE:FREQ=WEEKLY;BYDAY=MO" in event.recurrence


@pytest.mark.asyncio
async def test_get_calendar_event_empty_id_raises() -> None:
    """Verify get_calendar_event raises error when event_id is blank."""
    client = MagicMock(spec=GoogleCalendarClient)

    with pytest.raises(IntegrationExecutionError) as exc_info:
        await get_calendar_event(client, event_id="")
    assert "event_id must not be empty" in str(exc_info.value)


# ==============================================================================
# Tool 4: create_calendar_event Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_create_calendar_event_timed_with_duration() -> None:
    """Verify create_calendar_event calculates end_time from duration_minutes."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        nonlocal captured_args
        captured_args = arguments
        if tool_slug == GOOGLECALENDAR_CREATE_EVENT_ACTION:
            return {
                "id": "new_evt_1",
                "summary": arguments.get("summary"),
                "status": "confirmed",
                "start": {"dateTime": arguments.get("start_datetime")},
                "end": {"dateTime": arguments.get("end_datetime")},
            }
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await create_calendar_event(
        client,
        summary="SYNQRA Standup",
        start_time="2026-11-08T09:00:00Z",
        duration_minutes=45,
        timezone="UTC",
        attendees=["alice@example.com", "bob@example.com"],
    )

    assert result.id == "new_evt_1"
    assert result.summary == "SYNQRA Standup"
    assert captured_args["start_datetime"] == "2026-11-08T09:00:00+00:00"
    assert captured_args["end_datetime"] == "2026-11-08T09:45:00+00:00"
    assert captured_args["attendees"] == ["alice@example.com", "bob@example.com"]


@pytest.mark.asyncio
async def test_create_calendar_event_all_day() -> None:
    """Verify create_calendar_event handles all-day events using date formatting."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        nonlocal captured_args
        captured_args = arguments
        return {
            "id": "all_day_evt",
            "summary": arguments.get("summary"),
            "status": "confirmed",
            "start": {"date": arguments.get("start_date")},
            "end": {"date": arguments.get("end_date")},
        }

    client.execute_tool_async.side_effect = mock_execute

    result = await create_calendar_event(
        client,
        summary="Team Offsite",
        start_time="2026-11-12",
        end_time="2026-11-14",
        is_all_day=True,
    )

    assert result.id == "all_day_evt"
    assert captured_args["start_date"] == "2026-11-12"
    assert captured_args["end_date"] == "2026-11-14"


@pytest.mark.asyncio
async def test_create_calendar_event_end_before_start_raises() -> None:
    """Verify create_calendar_event raises error when end_time is earlier than start_time."""
    client = MagicMock(spec=GoogleCalendarClient)

    with pytest.raises(IntegrationExecutionError) as exc_info:
        await create_calendar_event(
            client,
            summary="Invalid Meeting",
            start_time="2026-11-08T12:00:00Z",
            end_time="2026-11-08T10:00:00Z",
        )
    assert "must be after start_time" in str(exc_info.value)


# ==============================================================================
# Tool 5: update_calendar_event Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_update_calendar_event_patch_semantics() -> None:
    """Verify update_calendar_event only sends specified non-null fields."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        nonlocal captured_args
        captured_args = arguments
        if tool_slug == GOOGLECALENDAR_PATCH_EVENT_ACTION:
            return {
                "id": arguments.get("event_id"),
                "summary": arguments.get("summary", "Original Title"),
                "status": "confirmed",
            }
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await update_calendar_event(
        client,
        event_id="evt_to_patch",
        calendar_id="primary",
        summary="Updated Standup Title",
    )

    assert result.id == "evt_to_patch"
    assert result.summary == "Updated Standup Title"
    assert "summary" in captured_args
    assert "description" not in captured_args
    assert "location" not in captured_args
    assert "start_time" not in captured_args


# ==============================================================================
# Tool 6: delete_calendar_event Safety Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_delete_calendar_event_aborted_without_confirm() -> None:
    """Safety check: Verify delete_calendar_event does NOT delete if confirm=False."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    result = await delete_calendar_event(
        client,
        event_id="evt_safe_delete",
        calendar_id="primary",
        confirm=False,
    )

    assert result.confirmed is False
    assert result.status == "aborted"
    assert "NOT deleted" in result.message
    # Composio API must not be called
    client.execute_tool_async.assert_not_called()


@pytest.mark.asyncio
async def test_delete_calendar_event_executes_with_confirm() -> None:
    """Verify delete_calendar_event invokes deletion API when confirm=True."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        nonlocal captured_args
        captured_args = arguments
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await delete_calendar_event(
        client,
        event_id="evt_to_remove",
        calendar_id="primary",
        confirm=True,
    )

    assert result.confirmed is True
    assert result.status == "deleted"
    assert captured_args["event_id"] == "evt_to_remove"
    assert captured_args["calendar_id"] == "primary"


# ==============================================================================
# Tool 7: find_free_time Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_find_free_time_filters_by_duration() -> None:
    """Verify find_free_time filters slots to only those meeting minimum duration."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "calendars": {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-08T10:00:00Z",
                        "end": "2026-11-08T11:00:00Z",
                    }
                ],
                "free": [
                    # 60 minute slot (satisfies 30 min threshold)
                    {
                        "start": "2026-11-08T09:00:00Z",
                        "end": "2026-11-08T10:00:00Z",
                    },
                    # 15 minute slot (should be excluded by 30 min threshold)
                    {
                        "start": "2026-11-08T11:00:00Z",
                        "end": "2026-11-08T11:15:00Z",
                    },
                    # 45 minute slot (satisfies 30 min threshold)
                    {
                        "start": "2026-11-08T11:15:00Z",
                        "end": "2026-11-08T12:00:00Z",
                    },
                ],
            }
        }
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLECALENDAR_FIND_FREE_SLOTS_ACTION:
            return mock_data
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await find_free_time(
        client,
        time_min="2026-11-08T09:00:00Z",
        time_max="2026-11-08T12:00:00Z",
        duration_minutes=30,
        timezone="UTC",
    )

    assert result.total_free_slots == 2
    assert result.free_slots[0].duration_minutes == 60.0
    assert result.free_slots[1].duration_minutes == 45.0
    assert len(result.busy_slots) == 1
    assert result.busy_slots[0].duration_minutes == 60.0


# ==============================================================================
# Integration & FastMCP Tool Registration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_google_calendar_integration_tool_registration() -> None:
    """Verify GoogleCalendarIntegration registers all seven tools on FastMCP server."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.is_configured = True
    integration = GoogleCalendarIntegration(client=client)

    server = FastMCP("TestCalendarServer")
    registered = integration.register_tools(server)

    expected_tools = [
        "list_calendars",
        "list_calendar_events",
        "get_calendar_event",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        "find_free_time",
    ]

    assert len(registered) == 7
    for tool_name in expected_tools:
        assert tool_name in registered

    # Verify tools on the server
    server_tools = await server.list_tools()
    server_tool_names = [t.name for t in server_tools]
    for tool_name in expected_tools:
        assert tool_name in server_tool_names


@pytest.mark.asyncio
async def test_google_calendar_health_check_operational() -> None:
    """Verify health_check returns CONNECTED when configured and connected."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.is_configured = True
    client.check_connection_async = MagicMock()

    async def mock_conn():
        return True

    client.check_connection_async.side_effect = mock_conn

    integration = GoogleCalendarIntegration(client=client)
    status = await integration.health_check()

    assert status.state == IntegrationState.CONNECTED
    assert status.is_connected is True
    assert status.is_configured is True


@pytest.mark.asyncio
async def test_google_calendar_health_check_unauthenticated_provides_auth_url() -> None:
    """Verify health_check provides OAuth authorization URL when not connected."""
    client = MagicMock(spec=GoogleCalendarClient)
    client.is_configured = True
    client.check_connection_async = MagicMock()

    async def mock_conn():
        return False

    client.check_connection_async.side_effect = mock_conn

    async def mock_auth():
        return "https://connect.composio.dev/auth/cal_123"

    client.authorize_async = MagicMock(side_effect=mock_auth)

    integration = GoogleCalendarIntegration(client=client)
    status = await integration.health_check()

    assert status.state == IntegrationState.ERROR
    assert status.is_connected is False
    assert "https://connect.composio.dev/auth/cal_123" in str(status.error_message)
