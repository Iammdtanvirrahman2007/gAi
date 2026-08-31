from datetime import datetime, timezone
from zoneinfo import ZoneInfo

CAPABILITY = {
    "name": "time",
    "description": "Provides current date, time, weekday, and timezone-aware time information.",
}


def get_current_time(timezone_name="Asia/Dhaka"):
    """Return the current date and time for a timezone."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc

    now = datetime.now(tz)
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%I:%M:%S %p"),
        "weekday": now.strftime("%A"),
        "timezone": timezone_name,
    }


def get_time(timezone_name="Asia/Dhaka"):
    return get_current_time(timezone_name)["time"]


def get_date(timezone_name="Asia/Dhaka"):
    return get_current_time(timezone_name)["date"]


def get_weekday(timezone_name="Asia/Dhaka"):
    return get_current_time(timezone_name)["weekday"]


def answer(query, timezone_name="Asia/Dhaka"):
    """Handle basic time/date questions and return a structured result."""
    q = query.lower().strip()
    info = get_current_time(timezone_name)

    if any(word in q for word in ("time", "clock", "what time")):
        return {
            "success": True,
            "type": "time",
            "answer": f"The current time is {info['time']}.",
            "timezone": info["timezone"],
        }

    if any(word in q for word in ("date", "today's date", "what date")):
        return {
            "success": True,
            "type": "date",
            "answer": f"Today's date is {info['date']}.",
            "timezone": info["timezone"],
        }

    if any(word in q for word in ("day", "weekday", "what day")):
        return {
            "success": True,
            "type": "weekday",
            "answer": f"Today is {info['weekday']}.",
            "timezone": info["timezone"],
        }

    return {
        "success": False,
        "type": "unknown",
        "answer": None,
        "reason": "The time capability could not determine what information was requested.",
    }
