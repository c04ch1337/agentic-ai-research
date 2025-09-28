"""CLI entry point for agentic_ai package"""

import sys
import asyncio
from .cli import cli

if __name__ == "__main__":
    # Handle async CLI commands
    if len(sys.argv) > 1 and sys.argv[1] in ["research", "upload"]:
        asyncio.run(cli())
    else:
        cli()