"""
Deobfuscator: recursively strips Base64, hex (\x41 and 0x41 style), and URL encoding.
"""
import base64
import re
import urllib.parse
from dataclasses import dataclass

class Deobfuscator:
    MAX_ITERATIONS = 10

    def decode(self, text: str) -> str:
        """
        Recursively decode text through Base64, hex, and URL encoding layers.
        Iterates until stable or MAX_ITERATIONS reached.
        """
        for _ in range(self.MAX_ITERATIONS):
            decoded = self._decode_once(text)
            if decoded == text:
                break
            text = decoded
        return text

    def _decode_once(self, text: str) -> str:
        text = self._decode_url(text)
        text = self._decode_hex_escape(text)    # \x41\x42
        text = self._decode_hex_spaced(text)    # 0x41 0x42
        text = self._decode_base64(text)
        return text

    def _decode_url(self, text: str) -> str:
        try:
            decoded = urllib.parse.unquote(text)
            return decoded
        except Exception:
            return text

    def _decode_hex_escape(self, text: str) -> str:
        # Match sequences like \x41\x42 — replace whole sequence at once
        def replacer(m):
            try:
                return bytes.fromhex(m.group(0).replace('\\x', '')).decode('utf-8', errors='replace')
            except Exception:
                return m.group(0)
        return re.sub(r'(?:\\x[0-9a-fA-F]{2})+', replacer, text)

    def _decode_hex_spaced(self, text: str) -> str:
        # Match sequences like 0x41 0x42 0x43
        def replacer(m):
            try:
                hex_values = re.findall(r'0x([0-9a-fA-F]{2})', m.group(0))
                return bytes.fromhex(''.join(hex_values)).decode('utf-8', errors='replace')
            except Exception:
                return m.group(0)
        return re.sub(r'(?:0x[0-9a-fA-F]{2}\s*)+', replacer, text)

    def _decode_base64(self, text: str) -> str:
        # Find base64-looking tokens (at least 8 chars, valid base64 alphabet)
        pattern = r'[A-Za-z0-9+/]{8,}={0,2}'
        def replacer(m):
            token = m.group(0)
            # Pad if needed
            padded = token + '=' * (-len(token) % 4)
            try:
                decoded_bytes = base64.b64decode(padded)
                decoded_str = decoded_bytes.decode('utf-8')
                # Only return decoded if it's printable ASCII (not binary noise)
                if decoded_str.isprintable():
                    return decoded_str
            except Exception:
                pass
            return token
        return re.sub(pattern, replacer, text)
