---
name: mobile-first-layout
description: "Mobile-first responsive layout expert. Create device-optimized layouts prioritizing mobile UX. Actions: build responsive layouts, implement breakpoint strategies, optimize touch targets, create adaptive navigation. Patterns: landing page, dashboard, blog, e-commerce, portfolio. Techniques: CSS Grid, Flexbox, container queries, fluid typography. Breakpoints: 320px mobile, 768px tablet, 1024px desktop, 1400px large."
---

# Mobile-First Responsive Layout Expert

Create responsive layouts using mobile-first approach with modern CSS techniques like Grid and Flexbox for optimal user experience across all devices.

## When to Apply

Reference these guidelines when:
- Building new page layouts from scratch
- Refactoring existing layouts for better responsiveness
- Implementing navigation systems
- Optimizing touch interactions for mobile
- Creating adaptive component layouts

## Core Principles

### 1. Mobile-First Methodology

```css
/* Start with mobile styles (no media query) */
.container {
  padding: 16px;
  font-size: 16px;
}

/* Then enhance for larger screens */
@media (min-width: 768px) {
  .container {
    padding: 24px;
    font-size: 18px;
  }
}

@media (min-width: 1024px) {
  .container {
    padding: 32px;
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

### 2. Breakpoint Strategy

| Breakpoint | Target | Use Case |
|------------|--------|----------|
| Default | 320px+ | Mobile phones (portrait) |
| `sm` 640px | 640px+ | Large phones (landscape) |
| `md` 768px | 768px+ | Tablets (portrait) |
| `lg` 1024px | 1024px+ | Tablets (landscape), small laptops |
| `xl` 1280px | 1280px+ | Desktops |
| `2xl` 1536px | 1536px+ | Large desktops |

```css
:root {
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
  --breakpoint-2xl: 1536px;
}
```

### 3. Touch Target Sizing

```css
/* Minimum touch target: 44x44px (WCAG) / 48x48px (Material) */
.touch-target {
  min-width: 44px;
  min-height: 44px;
  padding: 12px;
}

/* Adequate spacing between targets */
.touch-list > * + * {
  margin-top: 8px; /* Minimum 8px between touch targets */
}
```

## Layout Patterns

### Responsive Container

```css
.container {
  width: 100%;
  max-width: var(--content-max-width, 1280px);
  margin-inline: auto;
  padding-inline: var(--page-padding-x, 16px);
}

@media (min-width: 768px) {
  .container {
    --page-padding-x: 24px;
  }
}

@media (min-width: 1024px) {
  .container {
    --page-padding-x: 32px;
  }
}
```

### Responsive Grid

```css
/* Auto-fit grid that adapts to available space */
.grid-auto {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: 24px;
}

/* Fixed column grid with breakpoints */
.grid-cols {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

@media (min-width: 768px) {
  .grid-cols {
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
  }
}

@media (min-width: 1024px) {
  .grid-cols {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1280px) {
  .grid-cols {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

### Sidebar Layout

```css
/* Mobile: stacked, Desktop: sidebar */
.layout-sidebar {
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
}

@media (min-width: 1024px) {
  .layout-sidebar {
    grid-template-columns: 280px 1fr;
  }
}

/* Sticky sidebar on desktop */
@media (min-width: 1024px) {
  .sidebar {
    position: sticky;
    top: 80px; /* Account for fixed header */
    height: fit-content;
    max-height: calc(100vh - 100px);
    overflow-y: auto;
  }
}
```

### Holy Grail Layout

```css
.layout-holy-grail {
  display: grid;
  grid-template-areas:
    "header"
    "main"
    "sidebar"
    "footer";
  grid-template-rows: auto 1fr auto auto;
  min-height: 100vh;
}

@media (min-width: 1024px) {
  .layout-holy-grail {
    grid-template-areas:
      "header header header"
      "nav main sidebar"
      "footer footer footer";
    grid-template-columns: 200px 1fr 250px;
    grid-template-rows: auto 1fr auto;
  }
}

.header { grid-area: header; }
.nav { grid-area: nav; }
.main { grid-area: main; }
.sidebar { grid-area: sidebar; }
.footer { grid-area: footer; }
```

## Navigation Patterns

### Mobile Hamburger Menu

```html
<header class="header">
  <a href="/" class="logo">Logo</a>

  <button
    class="menu-toggle"
    aria-expanded="false"
    aria-controls="nav-menu"
    aria-label="Toggle navigation"
  >
    <span class="hamburger"></span>
  </button>

  <nav id="nav-menu" class="nav" aria-hidden="true">
    <ul class="nav-list">
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
      <li><a href="/contact">Contact</a></li>
    </ul>
  </nav>
</header>
```

```css
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  position: sticky;
  top: 0;
  background: var(--surface-color);
  z-index: var(--z-sticky);
}

/* Mobile: Hidden by default */
.nav {
  position: fixed;
  inset: 0;
  top: 60px; /* Header height */
  background: var(--surface-color);
  padding: 24px;
  transform: translateX(100%);
  transition: transform 300ms var(--ease-out);
}

.nav[aria-hidden="false"] {
  transform: translateX(0);
}

.menu-toggle {
  display: flex;
  padding: 12px;
}

/* Desktop: Inline navigation */
@media (min-width: 768px) {
  .nav {
    position: static;
    transform: none;
    padding: 0;
    background: transparent;
  }

  .nav-list {
    display: flex;
    gap: 24px;
  }

  .menu-toggle {
    display: none;
  }
}
```

### Bottom Navigation (Mobile)

```css
.bottom-nav {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-around;
  background: var(--surface-color);
  border-top: 1px solid var(--border-color);
  padding: 8px 0;
  padding-bottom: env(safe-area-inset-bottom, 8px);
  z-index: var(--z-fixed);
}

.bottom-nav-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 8px 16px;
  min-width: 64px;
  font-size: 12px;
}

/* Hide on desktop */
@media (min-width: 768px) {
  .bottom-nav {
    display: none;
  }
}
```

## Responsive Typography

### Fluid Type Scale

```css
:root {
  /* Fluid typography using clamp() */
  --font-size-sm: clamp(0.875rem, 0.8rem + 0.25vw, 0.9375rem);
  --font-size-base: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);
  --font-size-lg: clamp(1.125rem, 1rem + 0.5vw, 1.25rem);
  --font-size-xl: clamp(1.25rem, 1rem + 1vw, 1.5rem);
  --font-size-2xl: clamp(1.5rem, 1.25rem + 1.25vw, 2rem);
  --font-size-3xl: clamp(2rem, 1.5rem + 2vw, 3rem);
  --font-size-4xl: clamp(2.5rem, 2rem + 2.5vw, 4rem);
}

body {
  font-size: var(--font-size-base);
  line-height: 1.6;
}

h1 { font-size: var(--font-size-4xl); line-height: 1.1; }
h2 { font-size: var(--font-size-3xl); line-height: 1.2; }
h3 { font-size: var(--font-size-2xl); line-height: 1.3; }
h4 { font-size: var(--font-size-xl); line-height: 1.4; }
```

### Readable Line Length

```css
/* Optimal reading width: 45-75 characters */
.prose {
  max-width: 65ch;
}

/* Full width on mobile, constrained on desktop */
.content {
  max-width: 100%;
}

@media (min-width: 768px) {
  .content {
    max-width: 75ch;
  }
}
```

## Responsive Images

```html
<!-- Responsive image with art direction -->
<picture>
  <source
    media="(min-width: 1024px)"
    srcset="hero-desktop.webp"
  />
  <source
    media="(min-width: 768px)"
    srcset="hero-tablet.webp"
  />
  <img
    src="hero-mobile.webp"
    alt="Hero image"
    loading="lazy"
    decoding="async"
    width="800"
    height="600"
  />
</picture>
```

```css
/* Responsive image container */
.img-responsive {
  width: 100%;
  height: auto;
  aspect-ratio: 16 / 9;
  object-fit: cover;
}

/* Background image responsive */
.hero {
  background-image: url('hero-mobile.jpg');
  background-size: cover;
  background-position: center;
}

@media (min-width: 768px) {
  .hero {
    background-image: url('hero-desktop.jpg');
  }
}
```

## Spacing System

```css
:root {
  /* Responsive spacing scale */
  --space-1: clamp(4px, 0.25rem + 0.1vw, 6px);
  --space-2: clamp(8px, 0.5rem + 0.2vw, 12px);
  --space-3: clamp(12px, 0.75rem + 0.3vw, 16px);
  --space-4: clamp(16px, 1rem + 0.5vw, 24px);
  --space-6: clamp(24px, 1.5rem + 1vw, 36px);
  --space-8: clamp(32px, 2rem + 1.5vw, 48px);
  --space-12: clamp(48px, 3rem + 2vw, 72px);
  --space-16: clamp(64px, 4rem + 3vw, 96px);
}

/* Section spacing */
section {
  padding-block: var(--space-12);
}

@media (min-width: 1024px) {
  section {
    padding-block: var(--space-16);
  }
}
```

## Safe Areas (Notch/Home Indicator)

```css
/* Account for device safe areas */
.header {
  padding-top: env(safe-area-inset-top, 0);
  padding-left: env(safe-area-inset-left, 16px);
  padding-right: env(safe-area-inset-right, 16px);
}

.footer {
  padding-bottom: env(safe-area-inset-bottom, 16px);
}

/* Full viewport height accounting for mobile browser chrome */
.full-height {
  min-height: 100vh;
  min-height: 100dvh; /* Dynamic viewport height */
}
```

## Testing Checklist

### Mobile Testing (320px - 767px)

- [ ] Content readable without horizontal scroll
- [ ] Touch targets minimum 44x44px
- [ ] Adequate spacing between interactive elements
- [ ] Navigation accessible (hamburger menu works)
- [ ] Forms usable with virtual keyboard
- [ ] Images scale properly
- [ ] Text size minimum 16px (prevents zoom on iOS)

### Tablet Testing (768px - 1023px)

- [ ] Layout adapts appropriately (2-column, etc.)
- [ ] Navigation transition smooth
- [ ] Touch and mouse interactions both work
- [ ] Portrait and landscape orientations tested

### Desktop Testing (1024px+)

- [ ] Full navigation visible
- [ ] Content width constrained (max-width)
- [ ] Hover states present
- [ ] Large screen utilizes space well
- [ ] No excessive white space or cramped layouts

### Performance

- [ ] Images use srcset/sizes
- [ ] Lazy loading for below-fold content
- [ ] CSS animations use GPU-accelerated properties
- [ ] No layout shift during loading
- [ ] Fonts display properly (font-display: swap)
