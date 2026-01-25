// Minimal telemetry hooks.
// Enterprise deployments usually forward these to Sentry/Datadog/etc.
// Keeping this as a tiny module lets us wire real telemetry later without touching call sites.

export type TelemetryContext = Record<string, unknown>;

export function trackEvent(name: string, ctx: TelemetryContext = {}): void {
  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console.debug("[telemetry:event]", name, ctx);
  }
}

export function trackError(err: unknown, ctx: TelemetryContext = {}): void {
  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console.error("[telemetry:error]", err, ctx);
  }
}

