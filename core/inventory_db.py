"""Manages data/inventory.xlsx (Timestamp, Item_Description, Quantity, Unit,
Manager_Name) with fuzzy-matched lookups and stock updates.
"""

import os
import re
import threading
from datetime import datetime

from openpyxl import Workbook, load_workbook
from thefuzz import fuzz

from . import PROJECT_ROOT

COLUMNS = ["Timestamp", "Item_Description", "Quantity", "Unit", "Manager_Name"]
HISTORY_SHEET = "History"
HISTORY_COLUMNS = ["Timestamp", "Item_Description", "Manager_Name", "Resulting_Quantity"]
TIMESTAMP_FORMAT = "%d-%m-%Y / %H:%M:%S"
DEFAULT_PATH = str(PROJECT_ROOT / "data" / "inventory.xlsx")


class InventoryDB:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            wb = Workbook()
            ws = wb.active
            ws.title = "Inventory"
            ws.append(COLUMNS)
            wb.create_sheet(HISTORY_SHEET).append(HISTORY_COLUMNS)
            wb.save(self.path)
            return

        try:
            wb = load_workbook(self.path)
            ws = wb.active
            header = [cell.value for cell in ws[1]] if ws.max_row >= 1 else []

            if header and header[0] == "Item_Code":
                self._migrate_from_item_code_schema(wb, ws)
            elif header and "Manager_Name" not in header:
                self._migrate_add_manager_name(wb, ws)

            if HISTORY_SHEET not in wb.sheetnames:
                wb.create_sheet(HISTORY_SHEET).append(HISTORY_COLUMNS)
                wb.save(self.path)
        except PermissionError:
            # inventory.xlsx is open elsewhere (e.g. Excel) - don't crash
            # app startup over it, just retry the migration next launch.
            pass

    def _migrate_from_item_code_schema(self, wb, ws):
        """One-time migration from the old 5-column schema (Item_Code,
        Item_Description, Quantity, Unit, Last_Updated) straight to the
        current one (Timestamp, Item_Description, Quantity, Unit,
        Manager_Name), preserving existing rows instead of silently
        misreading or discarding them. Manager_Name is unknown for rows
        recorded before this column existed, so it's left blank."""
        old_rows = list(ws.iter_rows(min_row=2, values_only=True))
        ws.delete_rows(1, ws.max_row)
        ws.append(COLUMNS)
        for row in old_rows:
            if len(row) < 2 or row[1] is None:
                continue
            desc, qty, unit = row[1], row[2], row[3]
            old_timestamp = row[4] if len(row) > 4 else None
            ws.append([self._reformat_timestamp(old_timestamp), desc, qty, unit, ""])
        wb.save(self.path)

    def _migrate_add_manager_name(self, wb, ws):
        """Adds the Manager_Name column to a file saved before it existed.
        Past rows have no recorded manager, so it's left blank for them."""
        old_rows = list(ws.iter_rows(min_row=2, values_only=True))
        ws.delete_rows(1, ws.max_row)
        ws.append(COLUMNS)
        for row in old_rows:
            if len(row) < 2 or row[1] is None:
                continue
            ws.append([row[0], row[1], row[2], row[3], ""])
        wb.save(self.path)

    @staticmethod
    def _reformat_timestamp(value):
        if value is None:
            return datetime.now().strftime(TIMESTAMP_FORMAT)
        if isinstance(value, datetime):
            return value.strftime(TIMESTAMP_FORMAT)
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", TIMESTAMP_FORMAT):
            try:
                return datetime.strptime(text, fmt).strftime(TIMESTAMP_FORMAT)
            except ValueError:
                continue
        return text  # unrecognized format - keep as-is rather than lose data

    @staticmethod
    def _format_qty(qty):
        qty = float(qty)
        return str(int(qty)) if qty.is_integer() else f"{qty:g}"

    _NAME_RE = re.compile(r"^[A-Za-z]+(?:\s+[A-Za-z]+)*")

    @classmethod
    def _initials(cls, name):
        # Regex out the plain name only (first/middle/last) - stops at the
        # first non-letter character, so a trailing designation like
        # ", Jt.GM" or ",Jt.GM" is never included.
        match = cls._NAME_RE.match(name.strip())
        parts = match.group(0).split() if match else []
        if not parts:
            return "?"
        return "".join(p[0].upper() for p in parts)

    @classmethod
    def _format_manager_entry(cls, manager_name, qty, sign):
        manager_name = (manager_name or "Unknown").strip() or "Unknown"
        initials = cls._initials(manager_name)
        return f"{initials} ({sign}{cls._format_qty(qty)})"

    @staticmethod
    def _append_history(wb, timestamp, desc, manager_entry, resulting_qty):
        if HISTORY_SHEET not in wb.sheetnames:
            wb.create_sheet(HISTORY_SHEET).append(HISTORY_COLUMNS)
        wb[HISTORY_SHEET].append([timestamp, desc, manager_entry, resulting_qty])

    def get_history(self, desc, limit=100):
        """Every keep/withdraw transaction recorded for this item, newest
        first - unlike the Inventory sheet's one row per item (which only
        ever holds the latest state), nothing here is overwritten."""
        desc = self._clean_desc(desc)
        if not desc:
            return []

        with self._lock:
            wb = load_workbook(self.path)
            if HISTORY_SHEET not in wb.sheetnames:
                return []
            entries = []
            for row in wb[HISTORY_SHEET].iter_rows(min_row=2, values_only=True):
                if row[1] is None:
                    continue
                if self._clean_desc(row[1]).lower() != desc.lower():
                    continue
                entries.append({
                    "Timestamp": row[0],
                    "Item_Description": row[1],
                    "Manager_Name": row[2],
                    "Resulting_Quantity": row[3],
                })

        entries.reverse()
        return entries[:limit]

    def _load_rows(self):
        wb = load_workbook(self.path)
        ws = wb.active
        rows = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1] is None:
                continue
            rows.append({
                "Timestamp": row[0],
                "Item_Description": row[1],
                "Quantity": row[2] if row[2] is not None else 0,
                "Unit": row[3],
                "Manager_Name": row[4] if len(row) > 4 else "",
            })
        return wb, ws, rows

    @staticmethod
    def _clean_desc(text):
        """Collapses embedded newlines/extra whitespace (e.g. from
        multi-line OCR output) into a single space, so a description is
        always one clean line - Treeview renders a literal '\\n' as a real
        line break, which looks like a wrapping bug otherwise."""
        return " ".join((text or "").split())

    @staticmethod
    def _similarity(a, b):
        a, b = a.strip(), b.strip()
        if not a or not b:
            return 0
        if a.lower() == b.lower():
            return 100
        return fuzz.token_set_ratio(a, b)

    def search_item(self, query_text, limit=10, score_cutoff=55):
        """Fuzzy-matches query_text against every Item_Description, but a
        plain substring/prefix match (e.g. 'chill' inside 'Chillarwar Farms
        and Resort') always scores 100 and is returned, even when the two
        strings share no common whole words and fuzzy token matching alone
        would otherwise miss it."""
        query_text = self._clean_desc(query_text)
        if not query_text:
            return []
        query_lower = query_text.lower()

        with self._lock:
            _, _, rows = self._load_rows()

        scored = []
        for row in rows:
            # Clean the stored description too, not just the query - a row
            # saved before descriptions were sanitized at write time (e.g.
            # one with an embedded newline from multi-line OCR output) would
            # otherwise never match even a plain substring search, since the
            # newline breaks up what should be one contiguous string.
            desc = self._clean_desc(row["Item_Description"])
            if not desc:
                continue
            desc_lower = desc.lower()
            if query_lower in desc_lower or desc_lower in query_lower:
                score = 100
            else:
                score = self._similarity(query_text, desc)
            if score >= score_cutoff:
                scored.append({**row, "Score": score})

        scored.sort(key=lambda r: r["Score"], reverse=True)
        return scored[:limit]

    def _find_best_match(self, rows, desc, match_cutoff):
        best_idx, best_score = None, 0
        for idx, row in enumerate(rows):
            existing = self._clean_desc(row["Item_Description"])
            score = self._similarity(desc, existing)
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is not None and best_score >= match_cutoff:
            return best_idx
        return None

    def keep_item(self, desc, qty, unit, manager_name=None, match_cutoff=85):
        """Adds stock: increments the matched item's quantity, or creates a
        new row if no close match exists. Manager_Name records who made the
        change and the signed delta, e.g. 'Tusar Pandey [+2]'."""
        desc = self._clean_desc(desc)
        if not desc:
            raise ValueError("Item description is required")
        qty = float(qty)

        with self._lock:
            wb, ws, rows = self._load_rows()
            best_idx = self._find_best_match(rows, desc, match_cutoff)
            now = datetime.now().strftime(TIMESTAMP_FORMAT)
            manager_entry = self._format_manager_entry(manager_name, qty, "+")

            if best_idx is not None:
                excel_row = best_idx + 2  # +1 for header, +1 for 1-based indexing
                new_qty = (rows[best_idx]["Quantity"] or 0) + qty
                ws.cell(row=excel_row, column=1, value=now)
                ws.cell(row=excel_row, column=3, value=new_qty)
                ws.cell(row=excel_row, column=5, value=manager_entry)
                action = "updated"
            else:
                ws.append([now, desc, qty, unit, manager_entry])
                new_qty = qty
                action = "created"

            self._append_history(wb, now, desc, manager_entry, new_qty)
            wb.save(self.path)

        return {"action": action, "description": desc, "quantity": new_qty,
                "timestamp": now, "manager_entry": manager_entry}

    def withdraw_item(self, desc, qty, manager_name=None, match_cutoff=85):
        """Withdraws stock from an existing item. Raises ValueError if no
        matching item exists or if there isn't enough quantity on hand."""
        desc = self._clean_desc(desc)
        if not desc:
            raise ValueError("Item description is required")
        qty = float(qty)

        with self._lock:
            wb, ws, rows = self._load_rows()
            best_idx = self._find_best_match(rows, desc, match_cutoff)

            if best_idx is None:
                raise ValueError(f"No matching item found for '{desc}'")

            current_qty = rows[best_idx]["Quantity"] or 0
            if qty > current_qty:
                raise ValueError(f"Insufficient stock: only {current_qty} available")

            new_qty = current_qty - qty
            excel_row = best_idx + 2
            now = datetime.now().strftime(TIMESTAMP_FORMAT)
            manager_entry = self._format_manager_entry(manager_name, qty, "-")
            ws.cell(row=excel_row, column=1, value=now)
            ws.cell(row=excel_row, column=3, value=new_qty)
            ws.cell(row=excel_row, column=5, value=manager_entry)
            matched_desc = rows[best_idx]["Item_Description"]

            self._append_history(wb, now, matched_desc, manager_entry, new_qty)
            wb.save(self.path)

        return {"action": "withdrawn", "description": matched_desc, "quantity": new_qty,
                "timestamp": now, "manager_entry": manager_entry}
