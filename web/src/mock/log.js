export function mockLogResponse({ lines = 200, level, search } = {}) {
  const now = new Date();
  const fmt = (d) => d.toISOString().slice(0, 16).replace('T', ' ');

  const mk = (offsetMinutes, mod, line, lv, msg) => {
    const d = new Date(now.getTime() - offsetMinutes * 60 * 1000);
    const lvPad = String(lv).padEnd(7, ' ');
    return `[${fmt(d)}] [${mod}] [line:${line}] ${lvPad}: ${msg}`;
  };

  const sample = [
    mk(0, 'mock|startup', 1, 'INFO', 'Mock backend enabled (frontend demo mode)'),
    mk(1, 'mock|config', 2, 'INFO', 'Using localStorage for demo data'),
    mk(2, 'mock|chat', 3, 'WARNING', 'Chat responses are placeholders in offline mode'),
    mk(3, 'mock|tools', 18, 'DEBUG', 'Resolved /tools/ from mock handler'),
    mk(4, 'mock|kb', 44, 'DEBUG', 'Loaded demo knowledge base state'),
    mk(5, 'mock|graph', 67, 'INFO', 'Served demo graph nodes/edges'),
    mk(6, 'mock|tokens', 12, 'INFO', 'Created demo token for agent=demo'),
    mk(7, 'mock|mcp', 21, 'WARNING', 'MCP coords returned demo location'),
    mk(8, 'mock|http', 9, 'ERROR', 'Backend unreachable; falling back to mock (auto mode)'),
  ];

  const wantLevels = String(level || '')
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  const wantSearch = String(search || '').trim().toLowerCase();

  let list = sample;
  if (wantLevels.length > 0) list = list.filter((l) => wantLevels.some((lv) => l.includes(` ${lv}`)));
  if (wantSearch) list = list.filter((l) => l.toLowerCase().includes(wantSearch));

  const n = Math.max(1, Math.min(list.length, Number(lines) || 200));
  return { log: list.slice(0, n).join('\n') };
}
