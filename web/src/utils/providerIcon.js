// Centralized provider icon resolver.
// Keep filenames in `src/assets/providers/*.png` and map provider keys -> icon filenames here.

// Vite glob: use absolute `/src/...` (alias `@` may not work in glob patterns).
const providerIcons = import.meta.glob('/src/assets/providers/*.png', {
  eager: true,
  import: 'default'
})

// Provider key -> icon filename (without extension)
const PROVIDER_ICON_ALIAS = {
  zhipu: 'zhipuai',
  togetherai: 'together.ai'
}

export function getProviderIcon(provider) {
  const p = PROVIDER_ICON_ALIAS[provider] || provider
  return (
    providerIcons[`/src/assets/providers/${p}.png`] || providerIcons['/src/assets/providers/default.png']
  )
}

