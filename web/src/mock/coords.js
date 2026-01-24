export function mockCoordsResponse(place = '') {
  const q = String(place || '').trim();
  // A few deterministic demo locations.
  const samples = [
    { location: '东京（Demo）', lat: 35.6762, lng: 139.6503 },
    { location: '上海（Demo）', lat: 31.2304, lng: 121.4737 },
    { location: '旧金山（Demo）', lat: 37.7749, lng: -122.4194 },
  ];
  if (!q) return { coords: samples.slice(0, 1) };
  // Simple selection by hash-like behavior.
  const idx = Math.abs([...q].reduce((acc, ch) => acc + ch.charCodeAt(0), 0)) % samples.length;
  return { coords: [samples[idx]] };
}

