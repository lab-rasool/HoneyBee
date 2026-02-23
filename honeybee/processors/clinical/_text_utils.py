"""String-based text scanning utilities — no regex."""

from __future__ import annotations

from typing import List, Tuple

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_SET = frozenset(_MONTHS)


def find_dates(text: str) -> List[Tuple[int, int, str]]:
    """Find date-like strings in *text* without regex.

    Finds: MM/DD/YYYY, MM-DD-YYYY, Month DD YYYY, DD Month YYYY.
    Returns list of (start, end, matched_text).
    """
    results: list[tuple[int, int, str]] = []
    text_lower = text.lower()
    n = len(text)

    # Strategy 1: numeric dates — scan for / and - separators
    for i in range(n):
        ch = text[i]
        if ch in ('/', '-'):
            # Look for D{1,2} before separator
            left_end = i
            left_start = i - 1
            while left_start >= 0 and text[left_start].isdigit():
                left_start -= 1
            left_start += 1
            left_digits = i - left_start
            if left_digits < 1 or left_digits > 2:
                continue
            # Look for D{1,2} after separator
            mid_start = i + 1
            mid_end = mid_start
            while mid_end < n and text[mid_end].isdigit():
                mid_end += 1
            mid_digits = mid_end - mid_start
            if mid_digits < 1 or mid_digits > 2:
                continue
            # Look for second separator (same type)
            if mid_end >= n or text[mid_end] != ch:
                continue
            # Look for year digits
            yr_start = mid_end + 1
            yr_end = yr_start
            while yr_end < n and text[yr_end].isdigit():
                yr_end += 1
            yr_digits = yr_end - yr_start
            if yr_digits not in (2, 4):
                continue
            # Word boundary checks
            if left_start > 0 and text[left_start - 1].isalnum():
                continue
            if yr_end < n and text[yr_end].isalnum():
                continue
            matched = text[left_start:yr_end]
            results.append((left_start, yr_end, matched))

    # Strategy 2: month-name dates
    for month in _MONTHS:
        start = 0
        while True:
            idx = text_lower.find(month, start)
            if idx < 0:
                break
            m_end = idx + len(month)
            # Word boundary for month
            if idx > 0 and text_lower[idx - 1].isalpha():
                start = idx + 1
                continue
            if m_end < n and text_lower[m_end].isalpha():
                start = idx + 1
                continue

            # Try "Month DD, YYYY" or "Month DD YYYY"
            pos = m_end
            # skip whitespace
            while pos < n and text[pos] == ' ':
                pos += 1
            day_start = pos
            while pos < n and text[pos].isdigit():
                pos += 1
            day_digits = pos - day_start
            if 1 <= day_digits <= 2:
                # optional comma
                comma_pos = pos
                if pos < n and text[pos] == ',':
                    pos += 1
                # skip whitespace
                while pos < n and text[pos] == ' ':
                    pos += 1
                yr_start = pos
                while pos < n and text[pos].isdigit():
                    pos += 1
                yr_digits = pos - yr_start
                if yr_digits == 4:
                    # word boundary after year
                    if pos >= n or not text[pos].isalnum():
                        matched = text[idx:pos]
                        results.append((idx, pos, matched))
                        start = pos
                        continue

            # Try "DD Month YYYY" — look backwards from month
            back = idx - 1
            while back >= 0 and text[back] == ' ':
                back -= 1
            day_end = back + 1
            while back >= 0 and text[back].isdigit():
                back -= 1
            day_start2 = back + 1
            day_digits2 = day_end - day_start2
            if 1 <= day_digits2 <= 2:
                # word boundary before day
                if day_start2 == 0 or not text[day_start2 - 1].isalnum():
                    # look for year after month
                    pos2 = m_end
                    while pos2 < n and text[pos2] == ' ':
                        pos2 += 1
                    yr_s = pos2
                    while pos2 < n and text[pos2].isdigit():
                        pos2 += 1
                    if pos2 - yr_s == 4:
                        if pos2 >= n or not text[pos2].isalnum():
                            matched = text[day_start2:pos2]
                            results.append((day_start2, pos2, matched))
                            start = pos2
                            continue

            start = idx + 1

    # Deduplicate overlapping matches (keep longer)
    results.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    deduped: list[tuple[int, int, str]] = []
    for item in results:
        if deduped and item[0] < deduped[-1][1]:
            continue
        deduped.append(item)
    return deduped
