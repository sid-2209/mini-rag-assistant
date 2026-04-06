Title: Gmail OAuth Flow
Source: Google Integration Docs

Gmail integration uses OAuth 2.0 for authentication.

The OAuth flow consists of:

Redirecting the user to Google's authorization page
Exchanging the authorization code for access and refresh tokens
Refreshing the access token when it expires

Token refresh happens automatically when the token expiry time is within a safety window.

If a refresh token is not available, the system continues using the existing access token.
