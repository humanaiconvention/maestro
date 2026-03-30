# Account Linking Policy

Project Maestro implements a conservative, explicit account linking strategy designed to prevent silent account takeovers.

## Core Rules

1. **One Actor, Many Bindings**: A single `actor_id` can be associated with multiple login methods (e.g., one passkey, one magic link, one Google account).
2. **Never Auto-Link on Name**: Display names are not unique identifiers. We never merge actors based on matching names.
3. **Email Match != Silent Merge**: If a user logs in via an upstream provider (e.g., Google) and the email matches an existing account, **we do not silently merge**. This prevents OAuth-hijacking if an upstream provider has weak email verification. 
4. **Intentional Linking Wins**: The safest way to link an account is explicitly from the authenticated `/account/settings` page. When a logged-in user requests a new link, the system creates a short-lived transient state binding the new credential to the current `actor_id`.

## Risk Cases & Mitigation

### The "Ambiguous Email Collision"
**Scenario**: User has a Maestro account via Magic Link (`alice@example.com`). They later click "Sign in with Google" and Google asserts the same email.
**Policy**: The `LinkingService` detects the collision. Instead of logging them in and merging the Google ID to the existing Actor, it throws a `LinkingError`. The UI instructs the user to log in via Magic Link first, then add Google from their settings.

### The "Accidental Orphan"
**Scenario**: User attempts to remove their only active `CredentialBinding` from the settings page.
**Policy**: The `unlink_provider` method checks active binding count. If `count <= 1`, the unlink is rejected.

## Future Expansion

To add a new provider (e.g., GitHub, Enterprise SAML):
1. Configure the upstream connector in Authentik.
2. Ensure Authentik passes the correct `external_subject` and `provider_name` down to the Identity API via OIDC claims.
3. Add the new type to the `ProviderType` Enum in `models.py`.
4. The `LinkingService` automatically supports new providers as long as the provider name and subject are stable.

