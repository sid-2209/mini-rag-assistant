Title: Workspace Synchronization
Source: Platform Architecture Docs

Workspace synchronization is responsible for keeping Slack workspace metadata up to date.

During sync:

All public channels are fetched
The bot automatically joins channels it is not a member of
Workspace users are fetched, excluding bots and deactivated users

Channel sync does not store message history.
Message history ingestion is handled separately.
