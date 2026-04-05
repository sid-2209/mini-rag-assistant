# Document Format

The loader accepts `.md` and `.txt` files.

For the intended evaluation flow, point the CLI at a folder containing 3-5 supported documents. The CLI will warn if the folder has fewer or more files than that expected range.

Each document should provide:

- a title
- a source name
- body content

Preferred metadata styles:

```md
---
title: Example Title
source: Example Source
---

Document body...
```

or

```txt
Title: Example Title
Source: Example Source

Document body...
```

Fallbacks:

- A top-level Markdown heading can act as the title.
- The filename becomes the title if no explicit title is present.
- The filename becomes the source if no explicit source is present.

Only the document body is chunked and indexed.

If metadata is omitted, the assistant can still run because of the fallbacks above, but explicit `title` and `source` fields produce clearer citations.
