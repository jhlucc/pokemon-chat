"use client";

import { useEffect, useRef } from "react";
import { useChatStore } from "@/store/useChatStore";
import { ChatMessage } from "@/components/chat/chat-message";
import { ChatInput } from "@/components/chat/chat-input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "@/types/chat";

export function ChatArea() {
    const { currentSessionId, sessions, addMessage, updateMessage } = useChatStore();
    const session = sessions.find((s) => s.id === currentSessionId);
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [session?.messages.length, session?.messages[session?.messages.length - 1]?.content]);

    const handleSend = async (content: string) => {
        if (!session) return;

        // 1. Add user message
        const userMsg: Message = {
            id: crypto.randomUUID(),
            role: "user",
            content,
            timestamp: Date.now(),
        };
        addMessage(session.id, userMsg);

        // 2. Placeholder AI message
        const aiMsgId = crypto.randomUUID();
        const aiMsg: Message = {
            id: aiMsgId,
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            status: "loading",
        };
        addMessage(session.id, aiMsg);

        try {
            // 3. API Call
            const response = await fetch("/chat/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: content,
                    meta: { stream: true },
                    thread_id: session.id,
                    history: session.messages
                }),
            });

            if (!response.ok) throw new Error("Network response was not ok");
            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedContent = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split("\n").filter((line) => line.trim() !== "");

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        // Handle reasoning
                        if (data.reasoning_content) {
                            // ChatRouter sends full accumulated reasoning, so we update directly.
                            updateMessage(session.id, aiMsgId, { reasoning_content: data.reasoning_content });
                        }

                        // Handle content
                        if (data.content || data.response) { // Backend uses 'response' or 'content' in different places? 
                            // ChatRouter make_chunk uses 'response' kwarg -> `{"response": content}`.
                            const delta = data.response || data.content;
                            // Wait, ChatRouter line 69: `"response": content`.
                            // But line 179: `make_chunk(meta, content=delta.content, status="loading")`.
                            // make_chunk assigns `content` arg to `response` key in payload.
                            // So we look for `response`.
                            if (delta) {
                                accumulatedContent += delta; // Wait, if it's delta, we append.
                                // The backend router yields `delta.content`. So it IS a delta?
                                // ChatRouter line 177: `content += delta.content`.
                                // Line 179: `yield make_chunk(..., content=delta.content...)`.
                                // So the backend sends DELTAS in streaming?
                                // BUT, line 131 (cache) returns `cached_response` (full).
                                // Line 179 seems to send delta.
                                // IMPORTANT: Next.js frontend logic needs to append if it's delta.
                                // If I am not sure, I should check `chat_router`.
                                // `chat_router`:
                                // `content += delta.content` (server side accumulator)
                                // `yield make_chunk(..., content=delta.content...)` (sending delta)
                                // So yes, it sends deltas.

                                // However, for the initial implementation, I will assume deltas.
                                // Wait, `accumulatedContent` logic in my code:
                                // I should perform `accumulatedContent += delta`.
                                // And update the message with `accumulatedContent`.

                                // BUT, if I update state every chunk, it's fine.
                            }
                        }

                        updateMessage(session.id, aiMsgId, {
                            content: accumulatedContent,
                            status: data.meta?.status === "finished" ? "finished" : "generating"
                        });

                    } catch (e) {
                        console.error("Error parsing JSON chunk", e);
                    }
                }
            }

            updateMessage(session.id, aiMsgId, { status: "finished" });

        } catch (error) {
            console.error("Chat error", error);
            updateMessage(session.id, aiMsgId, {
                content: "Error: Failed to send message.",
                status: "error",
            });
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-background relative">
            <ScrollArea className="flex-1">
                <div className="flex flex-col pb-32 pt-4">
                    {session?.messages.map((msg) => (
                        <ChatMessage key={msg.id} message={msg} />
                    ))}
                    <div ref={bottomRef} className="h-1" />
                </div>
            </ScrollArea>
            <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-background via-background to-transparent pt-10 pb-4 px-4">
                <div className="max-w-3xl mx-auto">
                    <ChatInput onSend={handleSend} disabled={session ? false : true} />
                    <p className="text-xs text-center text-muted-foreground mt-2">
                        AI can make mistakes. Please verify important information.
                    </p>
                </div>
            </div>
        </div>
    );
}
