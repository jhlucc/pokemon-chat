"use client";

import { Message } from "@/types/chat";
import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Markdown } from "@/components/chat/markdown";

interface ChatMessageProps {
    message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === "user";

    return (
        <div
            className={cn(
                "group w-full text-gray-800 dark:text-gray-100 border-b border-black/10 dark:border-gray-900/50",
                isUser ? "bg-white dark:bg-gray-800" : "bg-gray-50 dark:bg-[#444654]"
            )}
        >
            <div className="text-base gap-4 md:gap-6 md:max-w-2xl lg:max-w-xl xl:max-w-3xl flex p-4 m-auto">
                <div className="flex-shrink-0 flex flex-col relative items-end">
                    <Avatar className={cn("h-8 w-8", isUser ? "rounded-full" : "rounded-sm")}>
                        <AvatarFallback className={isUser ? "bg-blue-500 text-white" : "bg-green-500 text-white"}>
                            {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
                        </AvatarFallback>
                    </Avatar>
                </div>
                <div className="relative flex-1 overflow-hidden">
                    <div className="font-semibold text-sm mb-1 opacity-90">
                        {isUser ? "You" : "Pokemon Assistant"}
                    </div>
                    {message.reasoning_content && (
                        <div className="mb-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 text-xs text-muted-foreground rounded border border-yellow-200 dark:border-yellow-800">
                            <div className="font-semibold mb-1">Thinking Process:</div>
                            <Markdown content={message.reasoning_content} />
                        </div>
                    )}
                    <Markdown content={message.content} />
                    {message.status === "loading" && <span className="inline-block w-2 h-4 ml-1 align-bottom bg-gray-400 animate-pulse" />}
                </div>
            </div>
        </div>
    );
}
