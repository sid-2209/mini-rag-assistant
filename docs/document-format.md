# Document Format

The loader accepts `.md` and `.txt` files.

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

