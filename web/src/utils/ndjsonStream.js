/**
 * Read an NDJSON (newline-delimited JSON) ReadableStream response.
 *
 * Backend uses `application/x-ndjson` for streaming chat tokens.
 */
export async function readNdjsonStream(response, onJson, { onParseError } = {}) {
  if (!response?.body) {
    throw new Error('ReadableStream not supported.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');

    // Keep last (possibly incomplete) line in the buffer.
    buffer = lines.pop() || '';

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        // Allow async callbacks.
        await onJson(obj);
      } catch (e) {
        if (onParseError) onParseError(e, line);
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    try {
      await onJson(JSON.parse(tail));
    } catch (e) {
      if (onParseError) onParseError(e, tail);
    }
  }
}
