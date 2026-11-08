"""Google Calendar tool implementations for SYNQRA.

Provides core business logic for calendar discovery, event scheduling, event updates,
event deletions, and free/busy analysis using Composio's verified Google Calendar action slugs.
"""

import logging
import zoneinfo
from datetime import date, datetime, timedelta
from typing import Any

from synqra.integrations.exceptions import IntegrationExecutionError
from synqra.integrations.google_calendar.client import (
    GOOGLECALENDAR_CREATE_EVENT_ACTION,
    GOOGLECALENDAR_DELETE_EVENT_ACTION,
    GOOGLECALENDAR_EVENTS_GET_ACTION,
    GOOGLECALENDAR_EVENTS_LIST_ACTION,
    GOOGLECALENDAR_FIND_FREE_SLOTS_ACTION,
    GOOGLECALENDAR_LIST_CALENDARS_ACTION,
    GOOGLECALENDAR_PATCH_EVENT_ACTION,
    GoogleCalendarClient,
)
from synqra.integrations.google_calendar.schemas import (
    CalendarEventDetails,
    CalendarEventSummary,
    CalendarSummary,
    CreateCalendarEventOutput,
    DeleteCalendarEventOutput,
    EventAttendee,
    FindFreeTimeOutput,
    ListCalendarEventsOutput,
    ListCalendarsOutput,
    TimeSlot,
    UpdateCalendarEventOutput,
)

logger = logging.getLogger("synqra.integrations.google_calendar.tools")


def validate_iana_timezone(tz_name: str | None) -> str:
    """Validate that the given string is a valid IANA timezone identifier.

    Args:
        tz_name: Timezone string (e.g. 'America/New_York', 'UTC').

    Returns:
        The valid timezone name.

    Raises:
        IntegrationExecutionError: If tz_name is invalid or unrecognized.
    """
    if not tz_name:
        return "UTC"

    try:
        zoneinfo.ZoneInfo(tz_name)
        return tz_name
    except Exception as exc:
        raise IntegrationExecutionError(
            f"Invalid IANA timezone '{tz_name}'. Specify a recognized IANA timezone identifier "
            "(e.g. 'UTC', 'America/New_York', 'Europe/London', 'Asia/Kolkata').",
            integration_name="google_calendar",
        ) from exc


def parse_iso_datetime(dt_str: str, default_tz: str = "UTC") -> datetime:
    """Parse an ISO 8601 datetime string, attaching a default timezone if naive.

    Args:
        dt_str: ISO formatted timestamp (e.g. '2026-11-08T10:00:00Z').
        default_tz: Default timezone name if the input string is naive.

    Returns:
        Timezone-aware datetime object.
    """
    cleaned = dt_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise IntegrationExecutionError(
            f"Invalid ISO 8601 datetime format '{dt_str}'. "
            "Expected format like '2026-11-08T10:00:00Z'.",
            integration_name="google_calendar",
        ) from exc

    if dt.tzinfo is None:
        tz = zoneinfo.ZoneInfo(validate_iana_timezone(default_tz))
        dt = dt.replace(tzinfo=tz)

    return dt


async def list_calendars(
    client: GoogleCalendarClient,
    max_results: int = 50,
    page_token: str | None = None,
    show_hidden: bool = False,
    min_access_role: str | None = None,
) -> ListCalendarsOutput:
    """List calendars accessible to the connected user.

    Args:
        client: Active GoogleCalendarClient instance.
        max_results: Maximum calendars to return per page.
        page_token: Optional token for paginating through results.
        show_hidden: Include hidden calendars.
        min_access_role: Filter by minimum role ('reader', 'writer', 'owner').

    Returns:
        ListCalendarsOutput containing normalized calendar entries.
    """
    args: dict[str, Any] = {
        "max_results": max_results,
        "show_hidden": show_hidden,
    }
    if page_token:
        args["page_token"] = page_token
    if min_access_role:
        args["min_access_role"] = min_access_role

    data = await client.execute_tool_async(GOOGLECALENDAR_LIST_CALENDARS_ACTION, args)

    raw_items = data.get("items") or data.get("calendars") or []
    next_page_token = data.get("nextPageToken") or data.get("next_page_token")

    calendars: list[CalendarSummary] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        calendars.append(
            CalendarSummary(
                id=item.get("id") or "",
                summary=item.get("summary") or item.get("name") or "Untitled Calendar",
                description=item.get("description"),
                time_zone=item.get("timeZone") or item.get("time_zone"),
                primary=bool(item.get("primary", False)),
                access_role=item.get("accessRole") or item.get("access_role"),
                background_color=item.get("backgroundColor") or item.get("background_color"),
                foreground_color=item.get("foregroundColor") or item.get("foreground_color"),
            )
        )

    return ListCalendarsOutput(
        total_count=len(calendars),
        calendars=calendars,
        next_page_token=next_page_token,
    )


async def list_calendar_events(
    client: GoogleCalendarClient,
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
    """Retrieve calendar events within a specified date/time range.

    Args:
        client: Active GoogleCalendarClient instance.
        calendar_id: Calendar identifier to query (default: 'primary').
        time_min: Lower bound for event end times in ISO format.
        time_max: Upper bound for event start times in ISO format.
        timezone: Optional IANA timezone for formatting result times.
        query: Free text search filter.
        max_results: Maximum events to return (1-250).
        page_token: Pagination token.
        single_events: Whether to expand recurring events into instances.
        order_by: Result ordering ('startTime' or 'updated').

    Returns:
        ListCalendarEventsOutput with normalized event summaries.
    """
    args: dict[str, Any] = {
        "calendarId": calendar_id,
        "maxResults": max_results,
        "singleEvents": single_events,
    }

    if timezone:
        args["timeZone"] = validate_iana_timezone(timezone)
    if query:
        args["query"] = query
    if page_token:
        args["pageToken"] = page_token

    if time_min:
        dt_min = parse_iso_datetime(time_min, default_tz=timezone or "UTC")
        args["timeMin"] = dt_min.isoformat()
    if time_max:
        dt_max = parse_iso_datetime(time_max, default_tz=timezone or "UTC")
        args["timeMax"] = dt_max.isoformat()

    if time_min and time_max and dt_min >= dt_max:
        raise IntegrationExecutionError(
            f"time_min ({time_min}) must be earlier than time_max ({time_max}).",
            integration_name="google_calendar",
        )

    if order_by and single_events:
        args["orderBy"] = order_by

    data = await client.execute_tool_async(GOOGLECALENDAR_EVENTS_LIST_ACTION, args)

    raw_items = data.get("items") or data.get("events") or []
    next_page_token = data.get("nextPageToken") or data.get("next_page_token")

    events: list[CalendarEventSummary] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        start_obj = item.get("start") or {}
        end_obj = item.get("end") or {}

        is_all_day = bool(start_obj.get("date") and not start_obj.get("dateTime"))
        start_val = start_obj.get("dateTime") or start_obj.get("date")
        end_val = end_obj.get("dateTime") or end_obj.get("date")
        event_tz = start_obj.get("timeZone") or timezone

        events.append(
            CalendarEventSummary(
                id=item.get("id") or "",
                calendar_id=calendar_id,
                summary=item.get("summary") or "Untitled Event",
                description=item.get("description"),
                location=item.get("location"),
                status=item.get("status"),
                html_link=item.get("htmlLink") or item.get("html_link"),
                start_time=start_val,
                end_time=end_val,
                time_zone=event_tz,
                is_all_day=is_all_day,
            )
        )

    return ListCalendarEventsOutput(
        calendar_id=calendar_id,
        total_count=len(events),
        events=events,
        next_page_token=next_page_token,
    )


async def get_calendar_event(
    client: GoogleCalendarClient,
    event_id: str,
    calendar_id: str = "primary",
    timezone: str | None = None,
) -> CalendarEventDetails:
    """Retrieve full details of an individual calendar event.

    Args:
        client: Active GoogleCalendarClient instance.
        event_id: Unique event ID.
        calendar_id: Calendar containing the event.
        timezone: Optional timezone for formatted timestamps.

    Returns:
        CalendarEventDetails with comprehensive event metadata.
    """
    if not event_id or not event_id.strip():
        raise IntegrationExecutionError(
            "event_id must not be empty.",
            integration_name="google_calendar",
        )

    args: dict[str, Any] = {
        "event_id": event_id.strip(),
        "calendar_id": calendar_id,
    }
    if timezone:
        args["time_zone"] = validate_iana_timezone(timezone)

    data = await client.execute_tool_async(GOOGLECALENDAR_EVENTS_GET_ACTION, args)
    item = data.get("event") if isinstance(data.get("event"), dict) else data

    start_obj = item.get("start") or {}
    end_obj = item.get("end") or {}

    is_all_day = bool(start_obj.get("date") and not start_obj.get("dateTime"))
    start_val = start_obj.get("dateTime") or start_obj.get("date")
    end_val = end_obj.get("dateTime") or end_obj.get("date")
    event_tz = start_obj.get("timeZone") or timezone

    creator = item.get("creator") or {}
    organizer = item.get("organizer") or {}

    attendees_raw = item.get("attendees") or []
    attendees: list[EventAttendee] = []
    for att in attendees_raw:
        if isinstance(att, dict):
            attendees.append(
                EventAttendee(
                    email=att.get("email") or "",
                    display_name=att.get("displayName") or att.get("display_name"),
                    response_status=att.get("responseStatus") or att.get("response_status"),
                    optional=bool(att.get("optional", False)),
                )
            )

    reminders_raw = item.get("reminders") or {}
    reminders_list: list[str] = []
    if reminders_raw.get("useDefault"):
        reminders_list.append("default")
    for override in reminders_raw.get("overrides") or []:
        if isinstance(override, dict):
            method = override.get("method", "popup")
            minutes = override.get("minutes", 10)
            reminders_list.append(f"{method}:{minutes}m")

    recurrence = item.get("recurrence") or []

    return CalendarEventDetails(
        id=item.get("id") or event_id,
        calendar_id=calendar_id,
        summary=item.get("summary") or "Untitled Event",
        description=item.get("description"),
        location=item.get("location"),
        status=item.get("status"),
        html_link=item.get("htmlLink") or item.get("html_link"),
        start_time=start_val,
        end_time=end_val,
        time_zone=event_tz,
        is_all_day=is_all_day,
        creator_email=creator.get("email"),
        organizer_email=organizer.get("email"),
        attendees=attendees,
        reminders=reminders_list,
        recurrence=recurrence if isinstance(recurrence, list) else [str(recurrence)],
        created_time=item.get("created"),
        updated_time=item.get("updated"),
    )


async def create_calendar_event(
    client: GoogleCalendarClient,
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
    """Create a new calendar event with conflict-safe parameter normalization.

    Args:
        client: Active GoogleCalendarClient instance.
        summary: Event title.
        start_time: Start datetime or start date (YYYY-MM-DD).
        end_time: Optional end datetime or end date.
        duration_minutes: Optional duration in minutes if end_time is not specified.
        description: Optional event description or agenda.
        location: Optional location string or meeting URL.
        timezone: IANA timezone for the event.
        calendar_id: Target calendar (default: 'primary').
        attendees: Optional list of attendee email addresses.
        reminders_minutes: Reminder lead times in minutes.
        is_all_day: True for all-day events.

    Returns:
        CreateCalendarEventOutput containing the created event metadata.
    """
    valid_tz = validate_iana_timezone(timezone)

    args: dict[str, Any] = {
        "summary": summary.strip(),
        "calendar_id": calendar_id,
        "timezone": valid_tz,
    }

    if description:
        args["description"] = description
    if location:
        args["location"] = location

    _ = reminders_minutes

    if is_all_day:
        # Date format YYYY-MM-DD
        start_date_str = start_time.strip().split("T")[0]
        try:
            start_d = date.fromisoformat(start_date_str)
        except ValueError as exc:
            raise IntegrationExecutionError(
                f"Invalid start date '{start_time}' for all-day event. Expected YYYY-MM-DD.",
                integration_name="google_calendar",
            ) from exc

        if end_time:
            end_date_str = end_time.strip().split("T")[0]
            try:
                end_d = date.fromisoformat(end_date_str)
            except ValueError as exc:
                raise IntegrationExecutionError(
                    f"Invalid end date '{end_time}' for all-day event. Expected YYYY-MM-DD.",
                    integration_name="google_calendar",
                ) from exc
        else:
            end_d = start_d + timedelta(days=1)
            end_date_str = end_d.isoformat()

        if end_d <= start_d:
            raise IntegrationExecutionError(
                f"All-day end_date ({end_date_str}) must be after start_date ({start_date_str}).",
                integration_name="google_calendar",
            )

        args["start_date"] = start_date_str
        args["end_date"] = end_date_str
    else:
        # Timed event
        start_dt = parse_iso_datetime(start_time, default_tz=valid_tz)

        if end_time:
            end_dt = parse_iso_datetime(end_time, default_tz=valid_tz)
        elif duration_minutes:
            end_dt = start_dt + timedelta(minutes=duration_minutes)
        else:
            # Default to 30 minutes
            end_dt = start_dt + timedelta(minutes=30)

        if end_dt <= start_dt:
            raise IntegrationExecutionError(
                f"Event end_time ({end_dt.isoformat()}) must be after "
                f"start_time ({start_dt.isoformat()}).",
                integration_name="google_calendar",
            )

        args["start_datetime"] = start_dt.isoformat()
        args["end_datetime"] = end_dt.isoformat()

    if attendees:
        args["attendees"] = [a.strip() for a in attendees if a.strip()]

    data = await client.execute_tool_async(GOOGLECALENDAR_CREATE_EVENT_ACTION, args)
    item = data.get("response_data") or data.get("event") or data

    start_info = item.get("start") or {}
    end_info = item.get("end") or {}
    resolved_start = (
        start_info.get("dateTime")
        or start_info.get("date")
        or args.get("start_datetime")
        or args.get("start_date")
    )
    resolved_end = (
        end_info.get("dateTime")
        or end_info.get("date")
        or args.get("end_datetime")
        or args.get("end_date")
    )

    return CreateCalendarEventOutput(
        id=item.get("id") or "created_event",
        calendar_id=calendar_id,
        summary=item.get("summary") or summary,
        status=item.get("status") or "confirmed",
        html_link=item.get("htmlLink") or item.get("html_link"),
        start_time=resolved_start,
        end_time=resolved_end,
        timezone=valid_tz,
        attendees=attendees or [],
        message=f"Event '{summary}' successfully created on calendar '{calendar_id}'.",
    )


async def update_calendar_event(
    client: GoogleCalendarClient,
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
    """Update selected event fields without unintentionally overwriting others (PATCH semantics).

    Args:
        client: Active GoogleCalendarClient instance.
        event_id: Target event identifier.
        calendar_id: Calendar identifier.
        summary: Optional updated title.
        description: Optional updated description.
        location: Optional updated location.
        start_time: Optional updated start time (RFC3339 / ISO).
        end_time: Optional updated end time (RFC3339 / ISO).
        timezone: Optional updated timezone.
        attendees: Optional updated attendee email list.

    Returns:
        UpdateCalendarEventOutput with updated event metadata.
    """
    if not event_id or not event_id.strip():
        raise IntegrationExecutionError(
            "event_id must not be empty.",
            integration_name="google_calendar",
        )

    args: dict[str, Any] = {
        "event_id": event_id.strip(),
        "calendar_id": calendar_id,
    }

    if summary is not None:
        args["summary"] = summary
    if description is not None:
        args["description"] = description
    if location is not None:
        args["location"] = location

    resolved_tz = validate_iana_timezone(timezone) if timezone else None
    if resolved_tz:
        args["timezone"] = resolved_tz

    if start_time:
        start_dt = parse_iso_datetime(start_time, default_tz=resolved_tz or "UTC")
        args["start_time"] = start_dt.isoformat()
    if end_time:
        end_dt = parse_iso_datetime(end_time, default_tz=resolved_tz or "UTC")
        args["end_time"] = end_dt.isoformat()

    if start_time and end_time and start_dt >= end_dt:
        raise IntegrationExecutionError(
            f"start_time ({start_time}) must be earlier than end_time ({end_time}).",
            integration_name="google_calendar",
        )

    if attendees is not None:
        args["attendees"] = [a.strip() for a in attendees if a.strip()]

    data = await client.execute_tool_async(GOOGLECALENDAR_PATCH_EVENT_ACTION, args)
    item = data.get("event") if isinstance(data.get("event"), dict) else data

    start_info = item.get("start") or {}
    end_info = item.get("end") or {}

    return UpdateCalendarEventOutput(
        id=item.get("id") or event_id,
        calendar_id=calendar_id,
        summary=item.get("summary") or (summary or "Updated Event"),
        status=item.get("status") or "confirmed",
        html_link=item.get("htmlLink") or item.get("html_link"),
        start_time=start_info.get("dateTime") or start_info.get("date") or args.get("start_time"),
        end_time=end_info.get("dateTime") or end_info.get("date") or args.get("end_time"),
        timezone=resolved_tz or item.get("timeZone"),
        message=f"Event '{event_id}' successfully updated on calendar '{calendar_id}'.",
    )


async def delete_calendar_event(
    client: GoogleCalendarClient,
    event_id: str,
    calendar_id: str = "primary",
    confirm: bool = False,
) -> DeleteCalendarEventOutput:
    """Delete a calendar event with mandatory explicit confirmation safety gate.

    Args:
        client: Active GoogleCalendarClient instance.
        event_id: Target event ID.
        calendar_id: Calendar containing the event.
        confirm: Safety confirmation flag. Must be True to execute deletion.

    Returns:
        DeleteCalendarEventOutput with status and confirmation details.
    """
    if not event_id or not event_id.strip():
        raise IntegrationExecutionError(
            "event_id must not be empty.",
            integration_name="google_calendar",
        )

    clean_event_id = event_id.strip()

    if not confirm:
        logger.warning(
            "Deletion request for event '%s' aborted: explicit confirmation not provided.",
            clean_event_id,
        )
        return DeleteCalendarEventOutput(
            event_id=clean_event_id,
            calendar_id=calendar_id,
            status="aborted",
            confirmed=False,
            message=(
                f"Event '{clean_event_id}' was NOT deleted. Deletion requires confirmation. "
                "To confirm deletion, call delete_calendar_event with confirm=True."
            ),
        )

    args = {
        "event_id": clean_event_id,
        "calendar_id": calendar_id,
    }

    await client.execute_tool_async(GOOGLECALENDAR_DELETE_EVENT_ACTION, args)

    return DeleteCalendarEventOutput(
        event_id=clean_event_id,
        calendar_id=calendar_id,
        status="deleted",
        confirmed=True,
        message=f"Event '{clean_event_id}' was successfully deleted from calendar '{calendar_id}'.",
    )


async def find_free_time(
    client: GoogleCalendarClient,
    time_min: str,
    time_max: str,
    calendar_ids: list[str] | None = None,
    duration_minutes: int = 30,
    timezone: str = "UTC",
) -> FindFreeTimeOutput:
    """Identify available free time slots within a date/time range satisfying duration requirements.

    Args:
        client: Active GoogleCalendarClient instance.
        time_min: Start timestamp of search window (ISO format).
        time_max: End timestamp of search window (ISO format).
        calendar_ids: Optional list of calendar IDs to query (defaults to ['primary']).
        duration_minutes: Minimum duration in minutes for an acceptable free slot.
        timezone: IANA timezone identifier for the search window.

    Returns:
        FindFreeTimeOutput with available slots and busy periods.
    """
    valid_tz = validate_iana_timezone(timezone)
    dt_min = parse_iso_datetime(time_min, default_tz=valid_tz)
    dt_max = parse_iso_datetime(time_max, default_tz=valid_tz)

    if dt_min >= dt_max:
        raise IntegrationExecutionError(
            f"time_min ({time_min}) must be earlier than time_max ({time_max}).",
            integration_name="google_calendar",
        )

    cals = calendar_ids or ["primary"]

    args: dict[str, Any] = {
        "items": cals,
        "time_min": dt_min.isoformat(),
        "time_max": dt_max.isoformat(),
        "timezone": valid_tz,
    }

    data = await client.execute_tool_async(GOOGLECALENDAR_FIND_FREE_SLOTS_ACTION, args)

    # Parse calendars map
    calendars_map = data.get("calendars") or {}
    raw_free_slots: list[dict[str, Any]] = []
    raw_busy_slots: list[dict[str, Any]] = []

    for _cal_id, window in calendars_map.items():
        if isinstance(window, dict):
            raw_free_slots.extend(window.get("free") or [])
            raw_busy_slots.extend(window.get("busy") or [])

    # If the response returned top-level free/busy lists
    if not raw_free_slots and "free" in data:
        raw_free_slots = data.get("free") or []
    if not raw_busy_slots and "busy" in data:
        raw_busy_slots = data.get("busy") or []

    # Filter and normalize free slots
    qualifying_free_slots: list[TimeSlot] = []
    for slot in raw_free_slots:
        if not isinstance(slot, dict):
            continue
        start_raw = slot.get("start")
        end_raw = slot.get("end")
        if not start_raw or not end_raw:
            continue

        try:
            slot_start = parse_iso_datetime(start_raw, default_tz=valid_tz)
            slot_end = parse_iso_datetime(end_raw, default_tz=valid_tz)
            dur = (slot_end - slot_start).total_seconds() / 60.0

            if dur >= duration_minutes:
                qualifying_free_slots.append(
                    TimeSlot(
                        start=slot_start.isoformat(),
                        end=slot_end.isoformat(),
                        duration_minutes=round(dur, 1),
                    )
                )
        except Exception as exc:
            logger.debug("Skipping unparseable free slot: %s (%s)", slot, exc)

    # Normalize busy slots
    busy_slots: list[TimeSlot] = []
    for slot in raw_busy_slots:
        if not isinstance(slot, dict):
            continue
        start_raw = slot.get("start")
        end_raw = slot.get("end")
        if not start_raw or not end_raw:
            continue

        try:
            slot_start = parse_iso_datetime(start_raw, default_tz=valid_tz)
            slot_end = parse_iso_datetime(end_raw, default_tz=valid_tz)
            dur = (slot_end - slot_start).total_seconds() / 60.0
            busy_slots.append(
                TimeSlot(
                    start=slot_start.isoformat(),
                    end=slot_end.isoformat(),
                    duration_minutes=round(dur, 1),
                )
            )
        except Exception as exc:
            logger.debug("Skipping unparseable busy slot: %s (%s)", slot, exc)

    # Sort qualifying slots by start time
    qualifying_free_slots.sort(key=lambda s: s.start)
    busy_slots.sort(key=lambda s: s.start)

    msg = (
        f"Found {len(qualifying_free_slots)} available time slot(s) satisfying "
        f"minimum duration of {duration_minutes} minutes between {time_min} and {time_max}."
    )

    return FindFreeTimeOutput(
        time_min=dt_min.isoformat(),
        time_max=dt_max.isoformat(),
        timezone=valid_tz,
        total_free_slots=len(qualifying_free_slots),
        free_slots=qualifying_free_slots,
        busy_slots=busy_slots,
        message=msg,
    )
