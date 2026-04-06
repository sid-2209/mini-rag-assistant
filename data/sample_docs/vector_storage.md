Title: Vector Storage Design
Source: AI Platform Docs

All documents are stored in a vector database to enable semantic search.

Each vector document contains:

text
team_id
channel_id
user_id
timestamp
source

Documents are grouped by team_id to isolate workspaces.

Search queries retrieve the top-k most relevant chunks based on embedding similarity.

If no relevant chunks are found above a confidence threshold, the system must return no answer.
