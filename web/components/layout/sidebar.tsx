"use client";

import { useChatStore } from "@/store/useChatStore";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Plus, MessageSquare, Settings, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { SettingsDialog } from "@/components/settings/settings-dialog";

export function Sidebar({ className }: { className?: string }) {
    const { sessions, currentSessionId, setCurrentSession, addSession } = useChatStore();
    const [settingsOpen, setSettingsOpen] = useState(false);

    const handleNewChat = () => {
        const newSession = {
            id: crypto.randomUUID(),
            title: "New Chat",
            messages: [],
            createdAt: Date.now(),
        };
        addSession(newSession);
    };

    return (
        <div className={cn("flex h-full flex-col bg-muted/20 border-r", className)}>
            <div className="p-4">
                <Button onClick={handleNewChat} className="w-full justify-start gap-2" variant="default">
                    <Plus className="h-4 w-4" />
                    New Chat
                </Button>
            </div>
            <Separator />
            <ScrollArea className="flex-1 px-2">
                <div className="flex flex-col gap-2 p-2">
                    {sessions.map((session) => (
                        <Button
                            key={session.id}
                            variant={currentSessionId === session.id ? "secondary" : "ghost"}
                            className="justify-start gap-2 h-auto py-3 px-4 text-left font-normal"
                            onClick={() => setCurrentSession(session.id)}
                        >
                            <MessageSquare className="h-4 w-4 shrink-0" />
                            <div className="overflow-hidden">
                                <p className="truncate text-sm">{session.title}</p>
                                <p className="truncate text-xs text-muted-foreground">
                                    {new Date(session.createdAt).toLocaleDateString()}
                                </p>
                            </div>
                        </Button>
                    ))}
                    {sessions.length === 0 && (
                        <div className="text-center text-sm text-muted-foreground p-4">
                            No chats yet.
                        </div>
                    )}
                </div>
            </ScrollArea>
            <Separator />
            <div className="p-4 flex flex-col gap-2">
                <Button variant="ghost" className="justify-start gap-2">
                    <Bot className="h-4 w-4" />
                    Agents
                </Button>
                <Button variant="ghost" className="justify-start gap-2" onClick={() => setSettingsOpen(true)}>
                    <Settings className="h-4 w-4" />
                    Settings
                </Button>
            </div>
            <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
        </div>
    );
}
