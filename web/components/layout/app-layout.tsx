"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { useState, useEffect } from "react";

export function AppLayout({ children }: { children: React.ReactNode }) {
    const [isMobile, setIsMobile] = useState(false);

    // Simple mobile check
    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth < 768);
        checkMobile();
        window.addEventListener("resize", checkMobile);
        return () => window.removeEventListener("resize", checkMobile);
    }, []);

    if (isMobile) {
        // Return simple mobile layout (Sidebar in Sheet - TODO)
        // For now just render children
        return <div className="flex flex-col h-screen">{children}</div>;
    }

    return (
        <ResizablePanelGroup direction="horizontal" className="h-screen w-full">
            <ResizablePanel defaultSize={20} minSize={15} maxSize={30} className="hidden md:flex">
                <Sidebar className="h-full" />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={80}>
                <div className="flex h-full flex-col">
                    {children}
                </div>
            </ResizablePanel>
        </ResizablePanelGroup>
    );
}
