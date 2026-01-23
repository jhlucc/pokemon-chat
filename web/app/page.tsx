"use client";

import { ChatArea } from "@/components/chat/chat-area";
import { useChatStore } from "@/store/useChatStore";
import { useEffect } from "react";

export default function Home() {
  const { currentSessionId, addSession, sessions } = useChatStore();

  useEffect(() => {
    // Create initial session if none exists
    if (!currentSessionId && sessions.length === 0) {
      addSession({
        id: crypto.randomUUID(),
        title: "New Chat",
        messages: [],
        createdAt: Date.now(),
      });
    }
  }, [currentSessionId, sessions.length, addSession]);

  if (!currentSessionId && sessions.length === 0) {
    return null;
  }

  return (
    <main className="flex h-full flex-col w-full bg-background">
      <ChatArea />
    </main>
  );
}
