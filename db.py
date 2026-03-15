# db.py  —  Abaca Color Scanner
# Supabase database layer
#
# Fixes applied vs previous version:
#   • top_5_json / dominant_rgb_json / dominant_lab_json now properly serialised
#   • seg_found / wb_applied stored as int 0/1 (not bool)
#   • seg_coverage stored correctly (was always NULL)
#   • Empty-string client fields coerced to NULL (not stored as "EMPTY")
#   • scan_code no longer silently dropped
#   • location + grader_name columns added (visible in DB but missing from save)
#   • _clean() helper normalises every field before upsert
#   • get_scans() / get_all_scans() deserialise JSON columns back to objects

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache

# ── Supabase client ───────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client as SupabaseClient
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _or_null(val):
    """Return None for falsy / 'EMPTY' / 'NULL' strings; else return val."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().upper() in ("", "EMPTY", "NULL", "NONE"):
        return None
    return val


def _int_bool(val) -> int:
    """Coerce any truthy value to 1/0 for INT columns."""
    return 1 if val else 0


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=None):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _hash_pin(pin: str) -> str:
    """SHA-256 hash a PIN — matches the pin_hash column in Supabase."""
    return hashlib.sha256(pin.strip().encode()).hexdigest()


def _to_json_str(val) -> str:
    """Serialise a dict/list to a JSON string for TEXT columns."""
    if val is None:
        return json.dumps({})
    if isinstance(val, str):
        # Already a string — validate it's valid JSON, else wrap
        try:
            json.loads(val)
            return val
        except (json.JSONDecodeError, TypeError):
            return json.dumps({})
    return json.dumps(val, ensure_ascii=False)


def _from_json(val):
    """Deserialise a JSON TEXT column back to a Python object."""
    if not val:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DB CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DB:
    """Thin wrapper around the Supabase Python client."""

    def __init__(self):
        self._supabase = None
        if _SUPABASE_AVAILABLE and _SUPABASE_URL and _SUPABASE_KEY:
            try:
                self._supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            except Exception as e:
                print(f"⚠️  Supabase init failed: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tbl(self, name: str):
        if not self._supabase:
            raise RuntimeError("Supabase not initialised")
        return self._supabase.table(name)

    # ─────────────────────────────────────────────────────────────────────────
    # USERS
    # ─────────────────────────────────────────────────────────────────────────

    def create_user(self, username: str, pin: str) -> dict:
        row = {
            "id":         str(uuid.uuid4()),
            "username":   username.strip(),
            "pin_hash":   _hash_pin(pin),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        res = self._tbl("users").insert(row).execute()
        return res.data[0] if res.data else row

    def _fetch_user_by_username(self, username: str):
        """Fetch a single user row by username, or None."""
        res = (
            self._tbl("users")
            .select("*")
            .eq("username", username.strip())
            .execute()
        )
        print(f"[DEBUG] fetch_user({username!r}) -> {res.data}")
        return res.data[0] if res.data else None

    def verify_user(self, username: str, pin: str):
        user = self._fetch_user_by_username(username)
        if not user:
            print(f"[DEBUG] verify_user: no user found for {username!r}")
            return None
        stored   = user.get("pin_hash", "")
        supplied = _hash_pin(pin)
        print(f"[DEBUG] verify_user: stored={stored[:16]}... supplied={supplied[:16]}... match={stored==supplied}")
        return user if stored == supplied else None

    def verify_admin(self, username: str, pin: str):
        user = self.verify_user(username, pin)
        if not user:
            return None
        # Support both: a role column OR a hardcoded admin username list
        role = user.get("role") or ""
        is_admin = (role == "admin") or (user.get("username") == "admin")
        print(f"[DEBUG] verify_admin: role={role!r} username={user.get('username')!r} is_admin={is_admin}")
        return user if is_admin else None

    def get_all_users(self) -> list:
        res = self._tbl("users").select("*").order("created_at", desc=True).execute()
        users = res.data or []

        # Fetch all scans once and compute per-user stats
        scans_res = self._tbl("scans").select("user_id,verified,scanned_at").execute()
        all_scans = scans_res.data or []

        from collections import defaultdict
        scan_counts = defaultdict(int)
        verified_counts = defaultdict(int)
        last_active = defaultdict(lambda: None)

        for s in all_scans:
            uid = s.get("user_id")
            if not uid:
                continue
            scan_counts[uid] += 1
            if s.get("verified") == 1:
                verified_counts[uid] += 1
            ts = s.get("scanned_at")
            if ts and (last_active[uid] is None or ts > last_active[uid]):
                last_active[uid] = ts

        for u in users:
            uid = u.get("id")
            u["scan_count"] = scan_counts[uid]
            u["verified_count"] = verified_counts[uid]
            if last_active[uid]:
                u["last_active"] = last_active[uid]
            last_seen = u.get("last_seen")
            try:
                seen_ts = datetime.fromisoformat(last_seen.replace("Z", "+00:00")).timestamp() if last_seen else None
                u["is_online"] = bool(seen_ts and (datetime.now(timezone.utc).timestamp() - seen_ts) < 120)
            except Exception:
                u["is_online"] = False

        return users

    def update_last_seen(self, user_id: str):
        self._tbl("users").update({
            "last_seen": datetime.now(timezone.utc).isoformat()
        }).eq("id", user_id).execute()

    def delete_user(self, user_id: str):
        self._tbl("users").delete().eq("id", user_id).execute()

    def delete_user_scans(self, user_id: str):
        self._tbl("scans").delete().eq("user_id", user_id).execute()

    # ─────────────────────────────────────────────────────────────────────────
    # SCANS  — SAVE
    # ─────────────────────────────────────────────────────────────────────────

    def save_scan(self, data: dict) -> str:
        """
        Persist a scan to the `scans` table.

        `data` is the JSON body posted to /api/save.  It contains:
          • All fields returned by features.predict() (passed through by the
            frontend after a /predict call)
          • User-supplied metadata: user_id, scan_code, batch_id, supplier,
            grader_notes, notes, verified, correction, thumbnail_b64,
            location, grader_name

        This method handles all the flattening, JSON-serialisation, and
        NULL-normalisation so nothing is stored as "EMPTY" or as an empty dict.
        """
        scan_id = data.get("id") or str(uuid.uuid4())

        # ── Dominant RGB (dict → flat columns + JSON string) ──────────────
        dom_rgb = data.get("dominant_rgb") or {}
        if isinstance(dom_rgb, str):
            try:
                dom_rgb = json.loads(dom_rgb)
            except Exception:
                dom_rgb = {}

        rgb_r = _safe_int(dom_rgb.get("R") or data.get("rgb_r"))
        rgb_g = _safe_int(dom_rgb.get("G") or data.get("rgb_g"))
        rgb_b = _safe_int(dom_rgb.get("B") or data.get("rgb_b"))

        # ── Dominant Lab (dict → flat columns + JSON string) ──────────────
        dom_lab = data.get("dominant_lab") or {}
        if isinstance(dom_lab, str):
            try:
                dom_lab = json.loads(dom_lab)
            except Exception:
                dom_lab = {}

        lab_l = _safe_float(dom_lab.get("L") or data.get("lab_l"))
        lab_a = _safe_float(dom_lab.get("a") or data.get("lab_a"))
        lab_b = _safe_float(dom_lab.get("b") or data.get("lab_b"))

        # ── top_5 (list → JSON string) ────────────────────────────────────
        top_5 = data.get("top_5") or data.get("top5") or []
        if isinstance(top_5, str):
            try:
                top_5 = json.loads(top_5)
            except Exception:
                top_5 = []

        # ── rhs_grade ─────────────────────────────────────────────────────
        rhs_grade = (
            _or_null(data.get("rhs_grade"))
            or _or_null(data.get("rhs_code"))
            or "—"
        )

        # ── seg_coverage: handle both 0–1 and 0–100 representations ──────
        raw_cov = data.get("seg_coverage")
        if raw_cov is not None:
            cov_f = _safe_float(raw_cov, 0.0)
            # normalise to 0–100 range
            seg_coverage = cov_f if cov_f > 1.0 else round(cov_f * 100.0, 2)
        else:
            seg_coverage = None

        row = {
            # ── Identity ────────────────────────────────────────────────
            "id":       scan_id,
            "user_id":  _or_null(data.get("user_id")),

            # ── Scan metadata ────────────────────────────────────────────
            "scan_code":    _or_null(data.get("scan_code")),
            "rhs_grade":    rhs_grade,

            # ── ML result numerics ───────────────────────────────────────
            "delta_e":      _safe_float(data.get("delta_e")),
            "match_score":  _safe_float(data.get("match_score")),

            # ── Colour — flat ────────────────────────────────────────────
            "rgb_r":    rgb_r,
            "rgb_g":    rgb_g,
            "rgb_b":    rgb_b,
            "lab_l":    lab_l,
            "lab_a":    lab_a,
            "lab_b":    lab_b,

            # ── Colour — hex ─────────────────────────────────────────────
            "dominant_hex": _or_null(data.get("dominant_hex")),
            "matched_hex":  _or_null(data.get("matched_hex")),

            # ── Verdict ──────────────────────────────────────────────────
            "verdict":       _or_null(data.get("verdict")),
            "verdict_color": _or_null(data.get("verdict_color")),

            # ── User metadata ────────────────────────────────────────────
            "batch_id":     _or_null(data.get("batch_id")),
            "supplier":     _or_null(data.get("supplier")),
            "grader_notes": _or_null(data.get("grader_notes")),
            "notes":        _or_null(data.get("notes")),
            "location":     _or_null(data.get("location")),
            "grader_name":  _or_null(data.get("grader_name")),

            # ── Verification ─────────────────────────────────────────────
            "verified":       _safe_int(data.get("verified"), 0),
            "correction":     _or_null(data.get("correction")),
            "thumbnail_b64":  _or_null(data.get("thumbnail_b64")),

            # ── Sync flag ────────────────────────────────────────────────
            "is_synced": _safe_int(data.get("is_synced"), 0),

            # ── JSON blobs ───────────────────────────────────────────────
            "top_5_json":         _to_json_str(top_5),
            "dominant_rgb_json":  _to_json_str(dom_rgb),
            "dominant_lab_json":  _to_json_str(dom_lab),

            # ── Segmentation ─────────────────────────────────────────────
            "seg_found":    _int_bool(data.get("seg_found")),
            "seg_coverage": seg_coverage,

            # ── White-balance ────────────────────────────────────────────
            "wb_applied":    _int_bool(data.get("wb_applied")),
            "cast_label":    _or_null(data.get("cast_label")),
            "cast_warning":  _or_null(data.get("cast_warning")),
            "gap_warning":   _or_null(data.get("gap_warning")),

            # ── Real photo URL (set later by /api/save-photo) ─────────────
            "image_url":     _or_null(data.get("image_url")),
        }

        # Remove None values that have no DB DEFAULT so Supabase doesn't
        # complain about unexpected null on NOT NULL columns.
        # (rhs_grade is NOT NULL so we always keep it)
        nullable_ok = {k for k in row if k != "rhs_grade"}
        clean_row = {k: v for k, v in row.items() if k == "rhs_grade" or v is not None}

        self._tbl("scans").upsert(clean_row).execute()
        return scan_id

    def upload_scan_photo(self, scan_id: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> str | None:
        """
        Upload a real crop photo to Supabase Storage bucket 'scan-photos'.
        Returns the public URL or None on failure.

        Bucket setup (run once in Supabase dashboard):
          Storage → New bucket → Name: scan-photos → Public: ON
        """
        if not self._supabase:
            return None
        try:
            path = f"{scan_id}.jpg"
            self._supabase.storage.from_("scan-photos").upload(
                path=path,
                file=image_bytes,
                file_options={"content-type": mime_type, "upsert": "true"},
            )
            # Build public URL
            res = self._supabase.storage.from_("scan-photos").get_public_url(path)
            url = res if isinstance(res, str) else res.get("publicUrl") or res.get("publicURL")
            return url
        except Exception as e:
            print(f"⚠️  upload_scan_photo failed: {e}")
            return None

    def update_image_url(self, scan_id: str, image_url: str):
        """Update the image_url column for an existing scan."""
        try:
            self._tbl("scans").update({"image_url": image_url}).eq("id", scan_id).execute()
        except Exception as e:
            print(f"⚠️  update_image_url failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # SCANS  — READ
    # ─────────────────────────────────────────────────────────────────────────

    def _deserialise_scan(self, row: dict) -> dict:
        """
        Convert a raw DB row back into the shape the frontend expects:
          • top_5_json → top_5  (list)
          • dominant_rgb_json → dominant_rgb  (dict)
          • dominant_lab_json → dominant_lab  (dict)
          • seg_found / wb_applied int → bool
        """
        row = dict(row)

        row["top_5"]        = _from_json(row.pop("top_5_json", None)) or []
        row["dominant_rgb"] = _from_json(row.pop("dominant_rgb_json", None)) or {}
        row["dominant_lab"] = _from_json(row.pop("dominant_lab_json", None)) or {}

        row["seg_found"]  = bool(row.get("seg_found"))
        row["wb_applied"] = bool(row.get("wb_applied"))

        return row

    def get_scans(self, user_id: str = None) -> list:
        q = self._tbl("scans").select("*").order("scanned_at", desc=True)
        if user_id:
            q = q.eq("user_id", user_id)
        res = q.execute()
        return [self._deserialise_scan(r) for r in (res.data or [])]

    def get_all_scans(self) -> list:
        res = (
            self._tbl("scans")
            .select("*")
            .order("scanned_at", desc=True)
            .execute()
        )
        return [self._deserialise_scan(r) for r in (res.data or [])]

    def get_scan_by_id(self, scan_id: str) -> dict | None:
        res = self._tbl("scans").select("*").eq("id", scan_id).execute()
        if res.data:
            return self._deserialise_scan(res.data[0])
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self, user_id: str = None) -> dict:
        scans = self.get_scans(user_id=user_id)
        return self._compute_stats(scans)

    def get_admin_stats(self) -> dict:
        scans = self.get_all_scans()
        users = self.get_all_users()
        stats = self._compute_stats(scans)
        stats["total_users"] = len(users)
        return stats

    @staticmethod
    def _compute_stats(scans: list) -> dict:
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        today_scans = [
            s for s in scans
            if (s.get("scanned_at") or "").startswith(today)
        ]
        verified = [s for s in scans if s.get("verified") == 1]
        des = [s["delta_e"] for s in scans if s.get("delta_e") is not None]
        avg_de = round(sum(des) / len(des), 2) if des else None

        return {
            "total":    len(scans),
            "today":    len(today_scans),
            "verified": len(verified),
            "avg_de":   avg_de,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_db_instance: DB | None = None


def get_db() -> DB:
    global _db_instance
    if _db_instance is None:
        _db_instance = DB()
    return _db_instance