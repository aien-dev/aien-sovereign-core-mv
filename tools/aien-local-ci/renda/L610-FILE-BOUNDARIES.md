# File-aligned complete review

Two actual own-engine probes rejected artificial cuts inside arithmetic-replay.json. Both negatives are retained unchanged; no production review/PR state was written. Prompt-only clarification did not fix the input shape and is not claimed as success.

Measured the authentic625diff:374725characters,39files; the rejected JSON head blob is3313bytes and its whole diff is3655characters. Every changed JSON diff fits the12000-character packet budget. The old fixed-character splitter nevertheless cut it because preceding files consumed the packet.

Changed only segmentation: prefer complete file boundaries, then complete lines within an oversized file, falling back to exact character cuts only for oversized individual lines. No bytes omitted, normalized, summarized or classified away. Avoid a header-only packet for one oversized line. Bump authenticated cache schema to3; source policy and exact per-packet ranges/digests bind the new partition. HMAC, strict parsing, terminalfalse, <=2callbacks/pass and all publication/qualification boundaries remain unchanged. Captured negatives remain negative.

New file-boundary and complete-line regressions were observed RED before correction.100/100service testsPASS Mac+Spark. The authentic-source coverage proof independently verifies all bytes and whole fitting JSON files. No actual approval inferred; a one-request actual-engine check of the formerly split JSON packet and separate independent delta review remain required before deployment.
