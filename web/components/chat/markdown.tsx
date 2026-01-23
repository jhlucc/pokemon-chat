"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

interface MarkdownProps {
    content: string;
    className?: string;
}

export function Markdown({ content, className }: MarkdownProps) {
    return (
        <div
            className={cn(
                "prose dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 break-words custom-markdown",
                className
            )}
        >
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    p({ children }) {
                        return <p className="mb-2 last:mb-0">{children}</p>;
                    },
                    code({ node, className, children, ...props }) {
                        // Basic code block styling
                        const match = /language-(\w+)/.exec(className || "");
                        const isInline = !match && !String(children).includes("\n");

                        if (isInline) {
                            return <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm" {...props}>{children}</code>;
                        }

                        return (
                            <pre className="bg-muted p-4 rounded-lg overflow-x-auto my-4 font-mono text-sm">
                                <code className={className} {...props}>
                                    {children}
                                </code>
                            </pre>
                        );
                    },
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
}
