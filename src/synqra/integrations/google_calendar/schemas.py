"""Pydantic schemas for Google Calendar integration in SYNQRA.

Defines validated input parameters and structured response models for calendar listing,
event retrieval, event creation, event updates, event deletion, and free/busy search.
"""

from pydantic import BaseModel, Field


class CalendarSummary(BaseModel):
    """Normalized summary of a calendar accessible to the user."""

    id: str = Field(description="Calendar identifier (e.g. 'primary' or email address).")
    summary: str = Field(description="Title or name of the calendar.")
    description: str | None = Field(default=None, description="Description of the calendar.")
    time_zone: str | None = Field(default=None, description="Default timezone for the calendar.")
    primary: bool = Field(default=False, description="Whether this is the user's primary calendar.")
    access_role: str | None = Field(
        default=None,
        description="Effective access role for the user (e.g. 'owner', 'writer', 'reader').",
    )
    background_color: str | None = Field(
        default=None, description="Hex color code for calendar background."
    )
    foreground_color: str | None = Field(
        default=None, description="Hex color code for calendar text."
    )


class ListCalendarsInput(BaseModel):
    """Input parameters for listing calendars accessible to the user."""

    max_results: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum calendars to return in a single page (1-100).",
    )
    page_token: str | None = Field(
        default=None,
        description="Token for fetching a specific results page from a previous call.",
    )
    show_hidden: bool = Field(
        default=False,
        description="Whether to include hidden calendars that are not shown in the UI.",
    )
    min_access_role: str | None = Field(
        default=None,
        description=(
            "Filter by minimum access role ('freeBusyReader', 'reader', 'writer', 'owner')."
        ),
    )


class ListCalendarsOutput(BaseModel):
    """Structured response containing accessible Google Calendars."""

    total_count: int = Field(description="Number of calendars returned in this page.")
    calendars: list[CalendarSummary] = Field(
        default_factory=list, description="List of accessible calendars."
    )
    next_page_token: str | None = Field(
        default=None, description="Token for retrieving the next page of calendars."
    )


class EventAttendee(BaseModel):
    """Details of an event attendee."""

    email: str = Field(description="Email address of the attendee.")
    display_name: str | None = Field(default=None, description="Display name of the attendee.")
    response_status: str | None = Field(
        default=None,
        description=(
            "Status of the attendee response ('needsAction', 'declined', 'tentative', 'accepted')."
        ),
    )
    optional: bool = Field(
        default=False, description="Whether this attendee is optional for the event."
    )


class CalendarEventSummary(BaseModel):
    """Normalized summary of a calendar event."""

    id: str = Field(description="Unique identifier of the calendar event.")
    calendar_id: str = Field(default="primary", description="Calendar containing the event.")
    summary: str = Field(description="Title or summary of the event.")
    description: str | None = Field(default=None, description="Description or notes for the event.")
    location: str | None = Field(default=None, description="Physical or virtual meeting location.")
    status: str | None = Field(
        default=None, description="Status of the event ('confirmed', 'tentative', 'cancelled')."
    )
    html_link: str | None = Field(
        default=None, description="URL link to view the event in Google Calendar."
    )
    start_time: str | None = Field(
        default=None, description="ISO 8601 start datetime or YYYY-MM-DD date."
    )
    end_time: str | None = Field(
        default=None, description="ISO 8601 end datetime or YYYY-MM-DD date."
    )
    time_zone: str | None = Field(
        default=None, description="Timezone associated with the start/end times."
    )
    is_all_day: bool = Field(default=False, description="Whether the event is an all-day event.")


class ListCalendarEventsInput(BaseModel):
    """Input parameters for listing calendar events."""

    calendar_id: str = Field(
        default="primary",
        description="Calendar identifier (use 'primary' for the user's primary calendar).",
    )
    time_min: str | None = Field(
        default=None,
        description=(
            "Lower bound (inclusive) for event end times in ISO 8601 format "
            "(e.g. '2026-11-08T00:00:00Z')."
        ),
    )
    time_max: str | None = Field(
        default=None,
        description=(
            "Upper bound (exclusive) for event start times in ISO 8601 format "
            "(e.g. '2026-11-15T23:59:59Z')."
        ),
    )
    timezone: str | None = Field(
        default=None,
        description=(
            "IANA timezone to format returned event times (e.g. 'America/New_York', 'UTC')."
        ),
    )
    query: str | None = Field(
        default=None,
        description="Free-text search terms to filter events by summary, description, or attendee.",
    )
    max_results: int = Field(
        default=25,
        ge=1,
        le=250,
        description="Maximum events to return in a single page (1-250).",
    )
    page_token: str | None = Field(
        default=None,
        description="Token for retrieving the next page of events.",
    )
    single_events: bool = Field(
        default=True,
        description="Whether to expand recurring events into individual instances.",
    )
    order_by: str | None = Field(
        default="startTime",
        description=(
            "Order of events ('startTime' or 'updated'). Requires single_events=True if startTime."
        ),
    )


class ListCalendarEventsOutput(BaseModel):
    """Structured response containing calendar events."""

    calendar_id: str = Field(description="Calendar ID queried.")
    total_count: int = Field(description="Number of events returned in this page.")
    events: list[CalendarEventSummary] = Field(
        default_factory=list, description="List of matching events."
    )
    next_page_token: str | None = Field(
        default=None, description="Token for retrieving the next page of events."
    )


class GetCalendarEventInput(BaseModel):
    """Input parameters for retrieving an individual calendar event."""

    event_id: str = Field(description="Unique event ID to retrieve.", min_length=1)
    calendar_id: str = Field(
        default="primary",
        description="Calendar identifier containing the event (default: 'primary').",
    )
    timezone: str | None = Field(
        default=None,
        description="Timezone used to format event start and end times in response.",
    )


class CalendarEventDetails(BaseModel):
    """Detailed metadata for a calendar event."""

    id: str = Field(description="Event ID.")
    calendar_id: str = Field(description="Calendar identifier.")
    summary: str = Field(description="Event title or summary.")
    description: str | None = Field(default=None, description="Event description or notes.")
    location: str | None = Field(default=None, description="Physical or virtual meeting location.")
    status: str | None = Field(default=None, description="Event status.")
    html_link: str | None = Field(
        default=None, description="Web link to open the event in Google Calendar."
    )
    start_time: str | None = Field(
        default=None, description="ISO start datetime or YYYY-MM-DD date."
    )
    end_time: str | None = Field(default=None, description="ISO end datetime or YYYY-MM-DD date.")
    time_zone: str | None = Field(default=None, description="Timezone of the event.")
    is_all_day: bool = Field(default=False, description="Whether this is an all-day event.")
    creator_email: str | None = Field(default=None, description="Email of event creator.")
    organizer_email: str | None = Field(default=None, description="Email of event organizer.")
    attendees: list[EventAttendee] = Field(
        default_factory=list, description="List of event attendees."
    )
    reminders: list[str] = Field(
        default_factory=list, description="Summary of configured reminders."
    )
    recurrence: list[str] = Field(
        default_factory=list, description="Recurrence rules (RRULE) if recurring."
    )
    created_time: str | None = Field(default=None, description="Timestamp of event creation.")
    updated_time: str | None = Field(default=None, description="Timestamp of last update.")


class CreateCalendarEventInput(BaseModel):
    """Input parameters for creating a new calendar event."""

    summary: str = Field(description="Title or summary of the event.", min_length=1)
    start_time: str = Field(
        description=(
            "Event start time in ISO 8601 format (e.g. '2026-11-08T10:00:00Z') "
            "or YYYY-MM-DD for all-day."
        )
    )
    end_time: str | None = Field(
        default=None,
        description=(
            "Event end time in ISO 8601 format or YYYY-MM-DD. If omitted, duration_minutes is used."
        ),
    )
    duration_minutes: int | None = Field(
        default=None,
        ge=1,
        description="Event duration in minutes if end_time is not explicitly provided.",
    )
    description: str | None = Field(
        default=None, description="Optional description or agenda for the event."
    )
    location: str | None = Field(default=None, description="Optional physical or virtual location.")
    timezone: str = Field(
        default="UTC",
        description="IANA timezone (e.g. 'America/New_York', 'UTC'). Required for naive datetimes.",
    )
    calendar_id: str = Field(
        default="primary",
        description="Calendar identifier where event will be created (default: 'primary').",
    )
    attendees: list[str] | None = Field(
        default=None,
        description="List of attendee email addresses to invite.",
    )
    reminders_minutes: list[int] | None = Field(
        default=None,
        description="List of reminder trigger times in minutes prior to event (e.g. [10, 30]).",
    )
    is_all_day: bool = Field(
        default=False,
        description="Set to True for all-day events using YYYY-MM-DD date formats.",
    )


class CreateCalendarEventOutput(BaseModel):
    """Structured response for event creation."""

    id: str = Field(description="ID of the newly created calendar event.")
    calendar_id: str = Field(description="Calendar ID where event was created.")
    summary: str = Field(description="Title of the created event.")
    status: str = Field(default="confirmed", description="Event status.")
    html_link: str | None = Field(default=None, description="Web link to view the created event.")
    start_time: str | None = Field(description="Scheduled start time.")
    end_time: str | None = Field(description="Scheduled end time.")
    timezone: str | None = Field(description="Timezone of the event.")
    attendees: list[str] = Field(default_factory=list, description="List of invited attendees.")
    message: str = Field(description="Human-readable success message.")


class UpdateCalendarEventInput(BaseModel):
    """Input parameters for updating selected fields of an existing event (patch semantics)."""

    event_id: str = Field(description="Unique event ID to update.", min_length=1)
    calendar_id: str = Field(
        default="primary",
        description="Calendar identifier containing the event (default: 'primary').",
    )
    summary: str | None = Field(default=None, description="New title for the event.")
    description: str | None = Field(default=None, description="New description or agenda.")
    location: str | None = Field(default=None, description="New location or meeting link.")
    start_time: str | None = Field(
        default=None,
        description="New start time in ISO 8601 format (e.g. '2026-11-08T11:00:00Z').",
    )
    end_time: str | None = Field(
        default=None,
        description="New end time in ISO 8601 format (e.g. '2026-11-08T12:00:00Z').",
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone for updated start/end times.",
    )
    attendees: list[str] | None = Field(
        default=None,
        description="New list of attendee emails (replaces existing attendee list if provided).",
    )


class UpdateCalendarEventOutput(BaseModel):
    """Structured response for event updates."""

    id: str = Field(description="ID of the updated calendar event.")
    calendar_id: str = Field(description="Calendar ID containing the event.")
    summary: str = Field(description="Title of the event.")
    status: str = Field(default="confirmed", description="Event status.")
    html_link: str | None = Field(default=None, description="Web link to view the updated event.")
    start_time: str | None = Field(default=None, description="Updated start time.")
    end_time: str | None = Field(default=None, description="Updated end time.")
    timezone: str | None = Field(default=None, description="Timezone of the event.")
    message: str = Field(description="Human-readable confirmation message.")


class DeleteCalendarEventInput(BaseModel):
    """Input parameters for deleting a calendar event with mandatory confirmation."""

    event_id: str = Field(description="Unique ID of the event to delete.", min_length=1)
    calendar_id: str = Field(
        default="primary",
        description="Calendar identifier containing the event (default: 'primary').",
    )
    confirm: bool = Field(
        default=False,
        description="Explicit safety confirmation flag. Must be set to True to execute deletion.",
    )


class DeleteCalendarEventOutput(BaseModel):
    """Structured response for event deletion."""

    event_id: str = Field(description="ID of the target event.")
    calendar_id: str = Field(description="Calendar ID containing the event.")
    status: str = Field(description="Status of deletion ('deleted', 'aborted', 'failed').")
    confirmed: bool = Field(description="Whether explicit confirmation was provided.")
    message: str = Field(description="Human-readable status description.")


class TimeSlot(BaseModel):
    """A discrete block of time within a free/busy query."""

    start: str = Field(description="Start timestamp in ISO 8601 format.")
    end: str = Field(description="End timestamp in ISO 8601 format.")
    duration_minutes: float = Field(description="Duration of the time slot in minutes.")


class FindFreeTimeInput(BaseModel):
    """Input parameters for finding available free time slots."""

    time_min: str = Field(
        description=(
            "Start datetime for availability search window in ISO 8601 format "
            "(e.g. '2026-11-08T09:00:00Z')."
        )
    )
    time_max: str = Field(
        description=(
            "End datetime for availability search window in ISO 8601 format "
            "(e.g. '2026-11-08T18:00:00Z')."
        )
    )
    calendar_ids: list[str] = Field(
        default_factory=lambda: ["primary"],
        description="List of calendar identifiers or attendee emails to check for conflicts.",
    )
    duration_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minimum duration in minutes required for an available slot (default: 30).",
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone identifier for the search window (e.g. 'America/New_York', 'UTC')."
        ),
    )


class FindFreeTimeOutput(BaseModel):
    """Structured response containing available free time slots."""

    time_min: str = Field(description="Search window start timestamp.")
    time_max: str = Field(description="Search window end timestamp.")
    timezone: str = Field(description="Timezone of the query.")
    total_free_slots: int = Field(description="Number of qualifying free time slots found.")
    free_slots: list[TimeSlot] = Field(
        default_factory=list,
        description="Available free time slots that satisfy the requested minimum duration.",
    )
    busy_slots: list[TimeSlot] = Field(
        default_factory=list, description="Recorded busy periods within the search window."
    )
    message: str = Field(description="Human-readable summary of availability.")
