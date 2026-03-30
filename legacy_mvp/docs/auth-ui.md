# Authentication UI

The authentication user interface is driven by a "Human Experience First" philosophy, served directly from the Identity API using Jinja2 templates and Tailwind CSS.

## The Flow

1. **Entry Point (`/auth/login`)**:
   - Clean, branded Maestro ID screen.
   - **Primary CTA**: "Continue with Passkey" (Fastest, zero-knowledge, highest assurance).
   - **Fallback**: "Email me a sign-in link".
   - *Anti-Pattern avoided*: We do not dump a grid of 10 social providers on the first screen. "More options" are hidden behind a disclosure widget to reduce cognitive load.

2. **Onboarding (`/onboarding`)**:
   - Appears only on a user's first successful login.
   - Pushes passkey enrollment immediately to upgrade the user's assurance level from AAL1 (Email) to AAL2/3 (Passkey).
   - Can be skipped, updating the `onboarding_state` in `ActorProfile`.

3. **Account Settings (`/account/settings`)**:
   - The hub for credential management.
   - Lists active bindings, allows unlinking (if safe), and provides entry points for adding passkeys or social providers.
   - Displays recent `AuthEvent` logs for security visibility.

## Extensibility

The UI templates are located in `services/identity_api/templates/`. They use a base layout (`base.html`) that pulls Tailwind from a CDN for development ease. In production, this should be replaced by a compiled CSS asset or integrated into a standard React frontend application.

