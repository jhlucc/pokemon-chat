import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { ChatSession, Message, Agent, ChatConfig } from '../types/chat';

interface ChatState {
    sessions: ChatSession[];
    currentSessionId: string | null;
    agents: Agent[];
    config: ChatConfig;

    // Actions
    addSession: (session: ChatSession) => void;
    setCurrentSession: (sessionId: string) => void;
    addMessage: (sessionId: string, message: Message) => void;
    updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
    setAgents: (agents: Agent[]) => void;
    setConfig: (config: ChatConfig) => void;
}

export const useChatStore = create<ChatState>()(
    persist(
        (set) => ({
            sessions: [],
            currentSessionId: null,
            agents: [],
            config: { provider: 'siliconflow', model: '' },

            addSession: (session) => set((state) => ({ sessions: [session, ...state.sessions], currentSessionId: session.id })),
            setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),
            addMessage: (sessionId, message) => set((state) => ({
                sessions: state.sessions.map((s) =>
                    s.id === sessionId ? { ...s, messages: [...s.messages, message] } : s
                ),
            })),
            updateMessage: (sessionId, messageId, updates) => set((state) => ({
                sessions: state.sessions.map((s) =>
                    s.id === sessionId
                        ? {
                            ...s,
                            messages: s.messages.map((m) =>
                                m.id === messageId ? { ...m, ...updates } : m
                            ),
                        }
                        : s
                ),
            })),
            setAgents: (agents) => set({ agents }),
            setConfig: (config) => set({ config }),
        }),
        {
            name: 'pokemon-chat-storage',
            storage: createJSONStorage(() => localStorage),
        }
    )
);
