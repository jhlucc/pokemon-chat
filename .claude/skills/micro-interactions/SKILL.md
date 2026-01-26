---
name: micro-interactions
description: "Micro-interactions animation expert. Create subtle, performance-optimized animations that enhance UX. Actions: add hover effects, loading states, form feedback, transition animations. Elements: button, card, form field, icon, modal, toast. Triggers: hover, click, focus, load, success, error. Priorities: duration under 300ms, natural easing curves, prefers-reduced-motion support, GPU-accelerated transforms."
---

# Micro-Interactions Animation Expert

Create subtle and delightful micro-interactions that enhance user experience without being overwhelming. Includes hover effects, loading states, form feedback, and transition animations.

## When to Apply

Reference these guidelines when:
- Adding hover effects to buttons and cards
- Creating loading states and feedback animations
- Designing smooth transitions between UI states
- Implementing form validation feedback
- Building toast notifications and alerts

## Core Principles

### 1. Duration Guidelines

| Interaction Type | Duration | Use Case |
|-----------------|----------|----------|
| Micro (hover, focus) | 100-150ms | Button hover, focus rings |
| Small (feedback) | 150-250ms | Form validation, toggles |
| Medium (state change) | 250-350ms | Modal open/close, panels |
| Large (page transition) | 350-500ms | Route changes, major reveals |

### 2. Easing Functions

```css
/* Natural motion curves */
--ease-default: cubic-bezier(0.4, 0, 0.2, 1);     /* Standard */
--ease-in: cubic-bezier(0.4, 0, 1, 1);            /* Accelerate */
--ease-out: cubic-bezier(0, 0, 0.2, 1);           /* Decelerate */
--ease-bounce: cubic-bezier(0.34, 1.56, 0.64, 1); /* Playful */
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Springy */
```

### 3. Performance Rules

- **DO**: Use `transform` and `opacity` only (GPU-accelerated)
- **DON'T**: Animate `width`, `height`, `margin`, `padding`, `top`, `left`
- **DO**: Use `will-change` sparingly for known animations
- **DON'T**: Apply `will-change` to many elements

### 4. Accessibility

```css
/* Always respect user preferences */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Animation Patterns

### Button Interactions

```css
/* Hover scale with shadow */
.btn-interactive {
  transition: transform 150ms var(--ease-out),
              box-shadow 150ms var(--ease-out);
}
.btn-interactive:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.btn-interactive:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Loading state */
.btn-loading {
  position: relative;
  pointer-events: none;
}
.btn-loading::after {
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 600ms linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Card Hover Effects

```css
/* Lift effect */
.card-lift {
  transition: transform 200ms var(--ease-out),
              box-shadow 200ms var(--ease-out);
}
.card-lift:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
}

/* Glow effect (for dark mode) */
.card-glow:hover {
  box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.3);
}

/* Border reveal */
.card-border {
  position: relative;
}
.card-border::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 2px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask-composite: exclude;
  opacity: 0;
  transition: opacity 200ms;
}
.card-border:hover::before {
  opacity: 1;
}
```

### Form Feedback

```css
/* Input focus */
.input-animated {
  transition: border-color 150ms, box-shadow 150ms;
}
.input-animated:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(var(--primary-rgb), 0.15);
}

/* Validation states */
.input-success {
  border-color: var(--success);
  animation: success-pulse 300ms var(--ease-out);
}
.input-error {
  border-color: var(--error);
  animation: shake 300ms var(--ease-out);
}

@keyframes success-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(var(--success-rgb), 0); }
  50% { box-shadow: 0 0 0 4px rgba(var(--success-rgb), 0.2); }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  20%, 60% { transform: translateX(-4px); }
  40%, 80% { transform: translateX(4px); }
}
```

### Loading States

```css
/* Skeleton loading */
.skeleton {
  background: linear-gradient(
    90deg,
    var(--gray-200) 25%,
    var(--gray-100) 50%,
    var(--gray-200) 75%
  );
  background-size: 200% 100%;
  animation: skeleton-wave 1.5s infinite;
}
@keyframes skeleton-wave {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Pulse loading */
.pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Spinner */
.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--gray-200);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 800ms linear infinite;
}
```

### Toast/Notification Animations

```css
/* Slide in from top */
.toast-enter {
  animation: toast-in 300ms var(--ease-out) forwards;
}
.toast-exit {
  animation: toast-out 200ms var(--ease-in) forwards;
}

@keyframes toast-in {
  from {
    transform: translateY(-100%);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

@keyframes toast-out {
  from {
    transform: translateY(0);
    opacity: 1;
  }
  to {
    transform: translateY(-20px);
    opacity: 0;
  }
}
```

### Modal/Overlay Animations

```css
/* Backdrop fade */
.backdrop {
  opacity: 0;
  transition: opacity 200ms;
}
.backdrop.active {
  opacity: 1;
}

/* Modal scale in */
.modal {
  opacity: 0;
  transform: scale(0.95);
  transition: opacity 200ms, transform 200ms var(--ease-out);
}
.modal.active {
  opacity: 1;
  transform: scale(1);
}
```

## Vue 3 Integration

```vue
<template>
  <Transition name="fade-scale">
    <div v-if="visible" class="modal">
      <!-- content -->
    </div>
  </Transition>
</template>

<style>
.fade-scale-enter-active,
.fade-scale-leave-active {
  transition: opacity 200ms, transform 200ms var(--ease-out);
}
.fade-scale-enter-from,
.fade-scale-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>
```

## Checklist

Before delivering animated components:

- [ ] All durations under 350ms for micro-interactions
- [ ] Only `transform` and `opacity` animated
- [ ] `prefers-reduced-motion` media query included
- [ ] Easing functions use natural curves (not linear)
- [ ] No layout shift during animations
- [ ] Loading states provide visual feedback
- [ ] Focus states clearly visible
- [ ] Animations enhance, not distract from content
