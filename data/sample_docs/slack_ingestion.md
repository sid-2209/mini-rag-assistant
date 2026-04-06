Title: Slack Ingestion Overview
Source: Internal Design Notes

Slack messages are ingested using a two-step process.

First, messages are fetched from Slack channels using the Slack API.
Only messages containing non-empty text are processed.

Each message is normalized into a standard format with the following fields:

text
channel_id
user_id
timestamp
readable_time

After normalization, messages are converted into vector documents and stored in the vector database.

Slack messages are stored with the source set to "slack".