---
name: aria-accessibility
description: "ARIA accessibility implementation specialist. Ensure screen reader compatibility and WCAG compliance. Actions: implement ARIA attributes, add keyboard navigation, create focus management, build accessible components. Components: dropdown, modal, tabs, carousel, form, menu, tooltip, dialog. Standards: WCAG 2.1 AA/AAA, screen readers (NVDA, JAWS, VoiceOver)."
---

# ARIA Accessibility Implementation Specialist

Generate proper ARIA (Accessible Rich Internet Applications) implementations for complex UI components to ensure screen reader compatibility and full accessibility compliance.

## When to Apply

Reference these guidelines when:
- Building custom interactive components (dropdowns, modals, tabs)
- Implementing keyboard navigation
- Creating dynamic content that updates
- Adding form validation and error handling
- Making charts and data visualizations accessible

## Core ARIA Principles

### 1. Semantic HTML First

Always prefer native HTML elements over ARIA:

| Instead of | Use |
|------------|-----|
| `<div role="button">` | `<button>` |
| `<span role="link">` | `<a href="">` |
| `<div role="checkbox">` | `<input type="checkbox">` |
| `<div role="textbox">` | `<input>` or `<textarea>` |

### 2. ARIA Roles Reference

```html
<!-- Landmark roles -->
<header role="banner">
<nav role="navigation">
<main role="main">
<aside role="complementary">
<footer role="contentinfo">

<!-- Widget roles -->
<div role="dialog">         <!-- Modal dialog -->
<div role="alertdialog">    <!-- Alert requiring response -->
<div role="menu">           <!-- Navigation menu -->
<div role="menuitem">       <!-- Item in menu -->
<div role="tablist">        <!-- Tab container -->
<div role="tab">            <!-- Tab button -->
<div role="tabpanel">       <!-- Tab content -->
<div role="tooltip">        <!-- Tooltip -->
<div role="alert">          <!-- Important message -->
<div role="status">         <!-- Status update -->
```

### 3. Essential ARIA Attributes

```html
<!-- Labeling -->
aria-label="Close dialog"           <!-- Accessible name -->
aria-labelledby="heading-id"        <!-- Reference to label element -->
aria-describedby="description-id"   <!-- Reference to description -->

<!-- States -->
aria-expanded="true|false"          <!-- Expandable elements -->
aria-selected="true|false"          <!-- Selected item -->
aria-checked="true|false|mixed"     <!-- Checkbox state -->
aria-pressed="true|false"           <!-- Toggle button -->
aria-disabled="true"                <!-- Disabled state -->
aria-hidden="true"                  <!-- Hidden from AT -->
aria-invalid="true"                 <!-- Invalid input -->
aria-busy="true"                    <!-- Loading state -->

<!-- Properties -->
aria-haspopup="true|menu|dialog"    <!-- Has popup -->
aria-controls="element-id"          <!-- Controls element -->
aria-owns="element-id"              <!-- Owns element -->
aria-live="polite|assertive"        <!-- Live region -->
aria-atomic="true"                  <!-- Announce all changes -->
```

## Accessible Component Patterns

### Modal Dialog

```html
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  aria-describedby="modal-description"
>
  <h2 id="modal-title">Dialog Title</h2>
  <p id="modal-description">Description of the dialog purpose.</p>

  <!-- Content -->

  <button aria-label="Close dialog">X</button>
</div>
```

```javascript
// Focus management
const modal = {
  open(dialogEl) {
    this.previousFocus = document.activeElement;
    dialogEl.setAttribute('aria-hidden', 'false');

    // Focus first focusable element
    const focusable = dialogEl.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    focusable[0]?.focus();

    // Trap focus
    dialogEl.addEventListener('keydown', this.trapFocus);
  },

  close(dialogEl) {
    dialogEl.setAttribute('aria-hidden', 'true');
    this.previousFocus?.focus();
  },

  trapFocus(e) {
    if (e.key !== 'Tab') return;

    const focusable = e.currentTarget.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
};
```

### Dropdown Menu

```html
<div class="dropdown">
  <button
    id="menu-button"
    aria-haspopup="true"
    aria-expanded="false"
    aria-controls="menu-list"
  >
    Options
  </button>

  <ul
    id="menu-list"
    role="menu"
    aria-labelledby="menu-button"
    hidden
  >
    <li role="menuitem" tabindex="-1">Edit</li>
    <li role="menuitem" tabindex="-1">Duplicate</li>
    <li role="separator"></li>
    <li role="menuitem" tabindex="-1">Delete</li>
  </ul>
</div>
```

```javascript
// Keyboard navigation
menuButton.addEventListener('keydown', (e) => {
  switch (e.key) {
    case 'Enter':
    case ' ':
    case 'ArrowDown':
      e.preventDefault();
      openMenu();
      menuItems[0].focus();
      break;
    case 'ArrowUp':
      e.preventDefault();
      openMenu();
      menuItems[menuItems.length - 1].focus();
      break;
  }
});

menuList.addEventListener('keydown', (e) => {
  const currentIndex = menuItems.indexOf(document.activeElement);

  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      menuItems[(currentIndex + 1) % menuItems.length].focus();
      break;
    case 'ArrowUp':
      e.preventDefault();
      menuItems[(currentIndex - 1 + menuItems.length) % menuItems.length].focus();
      break;
    case 'Escape':
      closeMenu();
      menuButton.focus();
      break;
    case 'Home':
      e.preventDefault();
      menuItems[0].focus();
      break;
    case 'End':
      e.preventDefault();
      menuItems[menuItems.length - 1].focus();
      break;
  }
});
```

### Tabs

```html
<div class="tabs">
  <div role="tablist" aria-label="Sample tabs">
    <button
      role="tab"
      id="tab-1"
      aria-selected="true"
      aria-controls="panel-1"
      tabindex="0"
    >
      Tab 1
    </button>
    <button
      role="tab"
      id="tab-2"
      aria-selected="false"
      aria-controls="panel-2"
      tabindex="-1"
    >
      Tab 2
    </button>
  </div>

  <div
    role="tabpanel"
    id="panel-1"
    aria-labelledby="tab-1"
    tabindex="0"
  >
    Panel 1 content
  </div>

  <div
    role="tabpanel"
    id="panel-2"
    aria-labelledby="tab-2"
    tabindex="0"
    hidden
  >
    Panel 2 content
  </div>
</div>
```

### Form Validation

```html
<form>
  <div class="form-group">
    <label for="email">Email *</label>
    <input
      type="email"
      id="email"
      name="email"
      aria-required="true"
      aria-invalid="false"
      aria-describedby="email-error email-hint"
    />
    <span id="email-hint" class="hint">We'll never share your email.</span>
    <span id="email-error" class="error" role="alert" hidden>
      Please enter a valid email address.
    </span>
  </div>
</form>
```

```javascript
// Show validation error
function showError(input, errorEl, message) {
  input.setAttribute('aria-invalid', 'true');
  errorEl.textContent = message;
  errorEl.hidden = false;
}

// Clear validation error
function clearError(input, errorEl) {
  input.setAttribute('aria-invalid', 'false');
  errorEl.hidden = true;
}
```

### Live Regions

```html
<!-- Polite announcements (wait for user to finish) -->
<div aria-live="polite" aria-atomic="true" class="sr-only">
  <!-- Dynamic content updates here -->
</div>

<!-- Assertive announcements (interrupt immediately) -->
<div aria-live="assertive" role="alert" class="sr-only">
  <!-- Critical alerts here -->
</div>

<!-- Status updates -->
<div role="status" aria-live="polite">
  3 items in cart
</div>
```

## Focus Management

### Skip Links

```html
<body>
  <a href="#main-content" class="skip-link">
    Skip to main content
  </a>

  <header><!-- ... --></header>

  <main id="main-content" tabindex="-1">
    <!-- Main content -->
  </main>
</body>
```

```css
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  padding: 8px 16px;
  background: var(--primary);
  color: white;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

### Focus Indicators

```css
/* Visible focus for all interactive elements */
:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

/* Remove default outline when using focus-visible */
:focus:not(:focus-visible) {
  outline: none;
}

/* High contrast focus for buttons */
button:focus-visible {
  outline: 3px solid var(--primary);
  outline-offset: 2px;
  box-shadow: 0 0 0 6px rgba(var(--primary-rgb), 0.2);
}
```

## Screen Reader Utilities

```css
/* Visually hidden but accessible to screen readers */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Show on focus (for skip links) */
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
```

## Testing Checklist

### Keyboard Testing

- [ ] All interactive elements reachable via Tab
- [ ] Tab order follows visual order
- [ ] Focus visible on all elements
- [ ] Escape closes modals/dropdowns
- [ ] Arrow keys work in menus/tabs
- [ ] Enter/Space activate buttons

### Screen Reader Testing

- [ ] All images have descriptive alt text
- [ ] Form inputs have associated labels
- [ ] Headings create logical outline
- [ ] Links have descriptive text (not "click here")
- [ ] Dynamic content announced via live regions
- [ ] Error messages announced immediately

### WCAG Compliance

- [ ] Color contrast minimum 4.5:1 (AA) or 7:1 (AAA)
- [ ] Text resizable to 200% without loss
- [ ] No content requires horizontal scrolling
- [ ] Focus not trapped (except modals)
- [ ] No automatic time limits (or user can extend)
- [ ] No content flashes more than 3 times/second

### Testing Tools

- **axe DevTools**: Browser extension for automated testing
- **WAVE**: Web accessibility evaluation tool
- **NVDA**: Free Windows screen reader
- **VoiceOver**: Built-in macOS/iOS screen reader
- **Lighthouse**: Chrome DevTools accessibility audit
