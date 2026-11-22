"""Read members of a remote zip archive through HTTP range requests (no full download)."""
import io
import urllib.request
import zipfile


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()   # follow redirects once

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, b):
        if b not in self.cache:
            lo, hi = b * self.block, min(self.size, (b + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        self.cache[b] = r.read()
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 4:
                        raise
            if len(self.cache) > 256:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b, off = divmod(self.pos, self.block)
            chunk = self._get(b)[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=1 << 20))
