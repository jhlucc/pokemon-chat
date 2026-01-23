export interface Message {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: number;
    // For streaming
    status?: "loading" | "thinking" | "generating" | "finished" | "error";
    reasoning_content?: string;
}

export interface ChatSession {
    id: string;
    title: string;
    messages: Message[];
    createdAt: number;
    agentName?: string;
}

export interface Agent {
    name: string;
    description?: string;
    supports_streaming?: boolean;
}

export interface ChatConfig {
    provider: string;
    model: string;
}

