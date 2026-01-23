"use client";

import * as React from "react";
import { Send, Paperclip } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
    onSend: (content: string) => void;
    disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
    const [content, setContent] = React.useState("");
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSend = () => {
        if (!content.trim() || disabled) return;
        onSend(content);
        setContent("");
    };

    return (
        <div className="relative flex items-end gap-2 p-4 border-t bg-background">
            <Button variant="outline" size="icon" className="shrink-0" disabled={disabled}>
                <Paperclip className="h-4 w-4" />
            </Button>
            <Textarea
                ref={textareaRef}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message..."
                className="min-h-[44px] max-h-[200px] resize-none pr-10"
                rows={1}
                disabled={disabled}
            />
            <Button onClick={handleSend} size="icon" disabled={!content.trim() || disabled}>
                <Send className="h-4 w-4" />
            </Button>
        </div>
    );
}
