import * as react from 'react';
import * as react_jsx_runtime from 'react/jsx-runtime';

type Annotation = {
    id: string;
    x: number;
    y: number;
    comment: string;
    element: string;
    elementPath: string;
    timestamp: number;
    selectedText?: string;
    boundingBox?: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    nearbyText?: string;
    cssClasses?: string;
    nearbyElements?: string;
    computedStyles?: string;
    fullPath?: string;
    accessibility?: string;
    isMultiSelect?: boolean;
    isFixed?: boolean;
};

type DemoAnnotation = {
    selector: string;
    comment: string;
    selectedText?: string;
};
type PageFeedbackToolbarCSSProps = {
    demoAnnotations?: DemoAnnotation[];
    demoDelay?: number;
    enableDemoMode?: boolean;
    /** Callback fired when an annotation is added. */
    onAnnotationAdd?: (annotation: Annotation) => void;
    /** Callback fired when an annotation is deleted. */
    onAnnotationDelete?: (annotation: Annotation) => void;
    /** Callback fired when an annotation comment is edited. */
    onAnnotationUpdate?: (annotation: Annotation) => void;
    /** Callback fired when all annotations are cleared. Receives the annotations that were cleared. */
    onAnnotationsClear?: (annotations: Annotation[]) => void;
    /** Callback fired when the copy button is clicked. Receives the markdown output. */
    onCopy?: (markdown: string) => void;
    /** Whether to copy to clipboard when the copy button is clicked. Defaults to true. */
    copyToClipboard?: boolean;
};
/** Alias for PageFeedbackToolbarCSSProps */
type AgentationProps = PageFeedbackToolbarCSSProps;
declare function PageFeedbackToolbarCSS({ demoAnnotations, demoDelay, enableDemoMode, onAnnotationAdd, onAnnotationDelete, onAnnotationUpdate, onAnnotationsClear, onCopy, copyToClipboard, }?: PageFeedbackToolbarCSSProps): react.ReactPortal | null;

interface AnnotationPopupCSSProps {
    /** Element name to display in header */
    element: string;
    /** Optional timestamp display (e.g., "@ 1.23s" for animation feedback) */
    timestamp?: string;
    /** Optional selected/highlighted text */
    selectedText?: string;
    /** Placeholder text for the textarea */
    placeholder?: string;
    /** Initial value for textarea (for edit mode) */
    initialValue?: string;
    /** Label for submit button (default: "Add") */
    submitLabel?: string;
    /** Called when annotation is submitted with text */
    onSubmit: (text: string) => void;
    /** Called when popup is cancelled/dismissed */
    onCancel: () => void;
    /** Position styles (left, top) */
    style?: React.CSSProperties;
    /** Custom color for submit button and textarea focus (hex) */
    accentColor?: string;
    /** External exit state (parent controls exit animation) */
    isExiting?: boolean;
    /** Light mode styling */
    lightMode?: boolean;
    /** Computed styles for the selected element */
    computedStyles?: Record<string, string>;
}
interface AnnotationPopupCSSHandle {
    /** Shake the popup (e.g., when user clicks outside) */
    shake: () => void;
}
declare const AnnotationPopupCSS: react.ForwardRefExoticComponent<AnnotationPopupCSSProps & react.RefAttributes<AnnotationPopupCSSHandle>>;

declare const IconClose: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconPlus: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheck: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheckSmall: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconListSparkle: ({ size, style, }: {
    size?: number;
    style?: React.CSSProperties;
}) => react_jsx_runtime.JSX.Element;
declare const IconHelp: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheckSmallAnimated: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCopyAlt: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCopyAnimated: ({ size, copied }: {
    size?: number;
    copied?: boolean;
}) => react_jsx_runtime.JSX.Element;
declare const IconEye: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconEyeAlt: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconEyeClosed: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconEyeAnimated: ({ size, isOpen }: {
    size?: number;
    isOpen?: boolean;
}) => react_jsx_runtime.JSX.Element;
declare const IconPausePlayAnimated: ({ size, isPaused }: {
    size?: number;
    isPaused?: boolean;
}) => react_jsx_runtime.JSX.Element;
declare const IconEyeMinus: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconGear: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconPauseAlt: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconPause: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconPlayAlt: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconTrashAlt: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconChatEllipsis: ({ size, style, }: {
    size?: number;
    style?: React.CSSProperties;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheckmark: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheckmarkLarge: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconCheckmarkCircle: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconXmark: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconXmarkLarge: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconSun: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const IconMoon: ({ size }: {
    size?: number;
}) => react_jsx_runtime.JSX.Element;
declare const AnimatedBunny: ({ size, color, }: {
    size?: number;
    color?: string;
}) => react_jsx_runtime.JSX.Element;

/**
 * Gets a readable path for an element (e.g., "article > section > p")
 */
declare function getElementPath(target: HTMLElement, maxDepth?: number): string;
/**
 * Identifies an element and returns a human-readable name + path
 */
declare function identifyElement(target: HTMLElement): {
    name: string;
    path: string;
};
/**
 * Gets text content from element and siblings for context
 */
declare function getNearbyText(element: HTMLElement): string;
/**
 * Simplified element identifier for animation feedback (less verbose)
 */
declare function identifyAnimationElement(target: HTMLElement): string;
/**
 * Gets CSS class names from an element (cleaned of module hashes)
 */
declare function getElementClasses(target: HTMLElement): string;

declare function getStorageKey(pathname: string): string;
declare function loadAnnotations<T = Annotation>(pathname: string): T[];
declare function saveAnnotations<T = Annotation>(pathname: string, annotations: T[]): void;

export { PageFeedbackToolbarCSS as Agentation, type AgentationProps, AnimatedBunny, type Annotation, AnnotationPopupCSS, type AnnotationPopupCSSHandle, type AnnotationPopupCSSProps, type DemoAnnotation, IconChatEllipsis, IconCheck, IconCheckSmall, IconCheckSmallAnimated, IconCheckmark, IconCheckmarkCircle, IconCheckmarkLarge, IconClose, IconCopyAlt, IconCopyAnimated, IconEye, IconEyeAlt, IconEyeAnimated, IconEyeClosed, IconEyeMinus, IconGear, IconHelp, IconListSparkle, IconMoon, IconPause, IconPauseAlt, IconPausePlayAnimated, IconPlayAlt, IconPlus, IconSun, IconTrashAlt, IconXmark, IconXmarkLarge, PageFeedbackToolbarCSS, getElementClasses, getElementPath, getNearbyText, getStorageKey, identifyAnimationElement, identifyElement, loadAnnotations, saveAnnotations };
