---
name: sql-generation
description: Generate safe parameterized SQL plans for oceanographic retrieval tasks. Use when natural language requests must be translated into query filters, joins, and time/depth constraints.
---

# SQL Generation

1. Build parameterized queries only; avoid string interpolation.
2. Prefer explicit column projection and bounded `LIMIT`.
3. Constrain by latitude, longitude, depth, and time when available.
4. Add fallback query plans with broader depth/region tolerance.
5. Return SQL + parameter map + rationale for planner transparency.

