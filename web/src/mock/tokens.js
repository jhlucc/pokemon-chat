import { randomId } from '@/utils/id';
import { getMockState, setMockState } from './state';

export function mockListTokens(agentId) {
  const state = getMockState();
  const list = state.tokensByAgent?.[agentId] || [];
  return Array.isArray(list) ? list : [];
}

export function mockCreateToken({ agent_id, name } = {}) {
  const agentId = String(agent_id || '').trim();
  const tokenName = String(name || '').trim();
  if (!agentId) return { status: 'failed', message: 'agent_id required' };
  if (!tokenName) return { status: 'failed', message: 'name required' };

  const state = getMockState();
  state.tokensByAgent = state.tokensByAgent || {};
  const list = Array.isArray(state.tokensByAgent[agentId]) ? state.tokensByAgent[agentId] : [];

  const item = {
    id: randomId(10),
    name: tokenName,
    token: `mock_${randomId(24)}`,
    created_at: new Date().toISOString(),
  };
  list.unshift(item);
  state.tokensByAgent[agentId] = list;
  setMockState(state);
  return item;
}

export function mockDeleteToken({ tokenId } = {}) {
  const id = String(tokenId || '').trim();
  if (!id) return { status: 'failed', message: 'tokenId required' };

  const state = getMockState();
  state.tokensByAgent = state.tokensByAgent || {};
  Object.keys(state.tokensByAgent).forEach((agentId) => {
    state.tokensByAgent[agentId] = (state.tokensByAgent[agentId] || []).filter((t) => String(t.id) !== id);
  });
  setMockState(state);
  return { status: 'ok' };
}

export function mockVerifyToken({ agent_id, token } = {}) {
  const agentId = String(agent_id || '').trim();
  const tk = String(token || '').trim();
  if (!agentId || !tk) return { ok: false, detail: 'missing agent_id/token' };

  const list = mockListTokens(agentId);
  const ok = list.some((t) => String(t.token) === tk);
  if (ok) return { ok: true };
  return { ok: false, detail: 'invalid token' };
}

