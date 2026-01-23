"use client";

import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useChatStore } from "@/store/useChatStore";
import { useEffect, useState } from "react";
import Image from "next/image";
import { useTheme } from "next-themes";

interface SettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const PROVIDERS = [
    { id: "siliconflow", name: "SiliconCloud", icon: "siliconcloud-color.svg" },
    { id: "openai", name: "OpenAI", icon: "openai.svg" },
    { id: "dashscope", name: "DashScope", icon: "bailian-color.svg" },
    { id: "zhipu", name: "Zhipu AI", icon: "zhipu-color.svg" },
    { id: "deepseek", name: "DeepSeek", icon: "deepseek-color.svg" },
    { id: "doubao", name: "Doubao", icon: "doubao-color.svg" },
];

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
    const { config, setConfig } = useChatStore();
    const [loading, setLoading] = useState(false);
    const [models, setModels] = useState<string[]>([]);

    // Init local state from store
    const [provider, setProvider] = useState(config?.provider || "siliconflow");
    const [model, setModel] = useState(config?.model || "");
    const { theme, setTheme } = useTheme();

    useEffect(() => {
        if (open) {
            setProvider(config?.provider || "siliconflow");
            setModel(config?.model || "");
        }
    }, [open, config]);

    // Fetch models when provider changes
    useEffect(() => {
        async function fetchModels() {
            if (!provider) return;
            setLoading(true);
            try {
                const res = await fetch(`/chat/models?model_provider=${provider}`);
                if (res.ok) {
                    const data = await res.json();
                    setModels(data.models || []);
                    // If current model is not in list, select first
                    if (data.models && data.models.length > 0 && !data.models.includes(model)) {
                        // Don't auto-switch immediately to avoid annoying jumps, but maybe useful
                    }
                }
            } catch (e) {
                console.error("Failed to fetch models", e);
            } finally {
                setLoading(false);
            }
        }
        fetchModels();
    }, [provider]);

    const handleSave = () => {
        setConfig({ ...config, provider, model });
        onOpenChange(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Settings</DialogTitle>
                    <DialogDescription>
                        Configure chat settings and model selection.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="provider" className="text-right">
                            Provider
                        </Label>
                        <Select value={provider} onValueChange={setProvider}>
                            <SelectTrigger className="col-span-3">
                                <SelectValue placeholder="Select provider" />
                            </SelectTrigger>
                            <SelectContent>
                                {PROVIDERS.map((p) => (
                                    <SelectItem key={p.id} value={p.id}>
                                        <div className="flex items-center gap-2">
                                            <Image
                                                src={`/providers/${p.icon}`}
                                                alt={p.name}
                                                width={20}
                                                height={20}
                                                className="rounded-sm"
                                            />
                                            {p.name}
                                        </div>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="model" className="text-right">
                            Model
                        </Label>
                        <Select value={model} onValueChange={setModel} disabled={loading}>
                            <SelectTrigger className="col-span-3">
                                <SelectValue placeholder={loading ? "Loading..." : "Select model"} />
                            </SelectTrigger>
                            <SelectContent>
                                {models.map((m) => (
                                    <SelectItem key={m} value={m}>{m}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="theme" className="text-right">
                            Theme
                        </Label>
                        <Select value={theme} onValueChange={setTheme}>
                            <SelectTrigger className="col-span-3">
                                <SelectValue placeholder="Select theme" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="light">Light</SelectItem>
                                <SelectItem value="dark">Dark</SelectItem>
                                <SelectItem value="system">System</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                </div>
                <DialogFooter>
                    <Button onClick={handleSave}>Save changes</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog >
    );
}
