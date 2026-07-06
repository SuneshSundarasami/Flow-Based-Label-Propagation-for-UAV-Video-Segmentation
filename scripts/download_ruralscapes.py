#!/usr/bin/env python3
"""Download the Ruralscapes dataset from its anonymous SharePoint share.

The dataset is published as an "anyone with the link" SharePoint folder. Such
folders can't be fetched with a single ``curl`` (only individual files can), so
this script drives SharePoint's REST API instead:

  1. GET the share link once -> SharePoint issues an anonymous ``FedAuth`` cookie
     and redirects to the real folder (``Documents/Ruralscapes``).
  2. Recursively enumerate ``.../Files`` and ``.../Folders`` to build a manifest.
  3. Download each file via ``GetFileByServerRelativeUrl('...')/$value``.

Downloads are resumable: a file whose on-disk size already matches the server is
skipped, so re-running after an interruption continues where it left off. Every
download is verified by exact byte size and retried (refreshing the anonymous
cookie) on mismatch.

Result layout matches what the pipeline expects (see README "Dataset"):

    data/Ruralscapes/
      ruralscapes_training_videos.txt
      ruralscapes_testing_videos.txt
      videos/*.MP4
      labels/manual_labels/<video>/segfull_*.png

Usage:
    python scripts/download_ruralscapes.py                 # -> ./data/Ruralscapes
    python scripts/download_ruralscapes.py --dest /some/dir
    python scripts/download_ruralscapes.py --url "<other share link>"
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Default public share (Ruralscapes, dragos_costea_upb_ro OneDrive).
DEFAULT_SHARE_URL = (
    "https://ctipub-my.sharepoint.com/:f:/g/personal/dragos_costea_upb_ro/"
    "EhvTFqxqv1FCk6wLdn3YdXoBOmRUuO8Js9eFoURj1SkXOg?e=fy0ixt"
)
UA = "Mozilla/5.0 (X11; Linux x86_64) download_ruralscapes.py"


class Share:
    """A live, cookie-backed session against one anonymous SharePoint share."""

    def __init__(self, share_url: str):
        self.share_url = share_url
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        self.site = ""   # e.g. https://host/personal/<user>
        self.root = ""   # server-relative path of the shared folder
        self.refresh()

    def refresh(self) -> None:
        """Follow the share link to (re)issue the anonymous cookie and learn
        the real site + folder path."""
        req = urllib.request.Request(self.share_url, headers={"User-Agent": UA})
        with self.opener.open(req, timeout=60) as resp:
            final = resp.geturl()
        parsed = urllib.parse.urlparse(final)
        qs = urllib.parse.parse_qs(parsed.query)
        if "id" not in qs:
            raise RuntimeError(f"could not locate folder id in redirect: {final}")
        self.root = qs["id"][0]                       # /personal/<user>/Documents/<folder>
        user_path = "/".join(self.root.split("/")[:3])  # /personal/<user>
        self.site = f"{parsed.scheme}://{parsed.netloc}{user_path}"

    def _api(self, folder: str, kind: str) -> list[dict]:
        url = (
            f"{self.site}/_api/web/GetFolderByServerRelativeUrl("
            f"'{urllib.parse.quote(folder)}')/{kind}"
            f"?$select=Name,Length&$top=5000"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json;odata=nometadata"}
        )
        with self.opener.open(req, timeout=120) as resp:
            return json.load(resp).get("value", [])

    def walk(self, folder: str | None = None, rel: str = "") -> list[tuple[str, int, str]]:
        """Return (server_relative_path, size, local_relative_path) for every file."""
        folder = folder or self.root
        out: list[tuple[str, int, str]] = []
        for f in self._api(folder, "Files"):
            out.append((f"{folder}/{f['Name']}", int(f.get("Length", 0)), rel + f["Name"]))
        for d in self._api(folder, "Folders"):
            out += self.walk(f"{folder}/{d['Name']}", f"{rel}{d['Name']}/")
        return out

    def fetch(self, server_path: str, dest: Path) -> int:
        url = (
            f"{self.site}/_api/web/GetFileByServerRelativeUrl("
            f"'{urllib.parse.quote(server_path)}')/$value"
        )
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self.opener.open(req, timeout=300) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        size = tmp.stat().st_size
        tmp.replace(dest)
        return size


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_SHARE_URL, help="SharePoint share link")
    ap.add_argument("--dest", default="data/Ruralscapes", type=Path,
                    help="output directory (default: data/Ruralscapes)")
    ap.add_argument("--retries", type=int, default=4, help="download attempts per file")
    args = ap.parse_args()

    print(f"Connecting to share ...", flush=True)
    share = Share(args.url)
    print(f"  folder : {share.root}")

    print("Enumerating files (this walks every subfolder) ...", flush=True)
    manifest = share.walk()
    total_bytes = sum(sz for _, sz, _ in manifest)
    print(f"  {len(manifest)} files, {human(total_bytes)} total\n", flush=True)

    args.dest.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    for i, (server_path, size, rel) in enumerate(manifest, 1):
        out = args.dest / rel
        if out.exists() and out.stat().st_size == size:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            try:
                got = share.fetch(server_path, out)
                if got == size:
                    done += 1
                    print(f"[{i}/{len(manifest)}] {rel}  ({human(size)})", flush=True)
                    break
                print(f"  size mismatch {rel}: got {got} want {size} (attempt {attempt})")
            except (urllib.error.URLError, OSError) as e:
                print(f"  error {rel}: {e} (attempt {attempt})")
            share.refresh()  # cookie may have expired mid-run
        else:
            failed += 1
            print(f"  FAILED {rel}")

    print(f"\nDone. downloaded={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
