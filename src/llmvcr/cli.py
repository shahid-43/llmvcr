"""
llmvcr CLI

Usage:
    llmvcr record --output cassettes/demo.yaml python my_script.py
    llmvcr playback --cassette cassettes/demo.yaml python my_script.py
    llmvcr info cassettes/demo.yaml
"""

import argparse
import sys
import os


def cmd_info(args):
    """Show metadata and interactions stored in a cassette file."""
    import yaml

    path = args.cassette
    if not os.path.exists(path):
        print(f"Error: cassette file not found: '{path}'", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    interactions = data.get("interactions", [])
    print(f"Cassette:      {path}")
    print(f"Provider:      {data.get('provider', 'unknown')}")
    print(f"Recorded at:   {data.get('recorded_at', 'unknown')}")
    print(f"Version:       {data.get('llmvcr_version', 'unknown')}")
    print(f"Interactions:  {len(interactions)}")
    print()

    for i, interaction in enumerate(interactions, 1):
        req = interaction.get("request", {})
        resp = interaction.get("response", {})
        model = req.get("model", "?")
        messages = req.get("messages", [])
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "")
        if len(content) > 60:
            content = content[:57] + "..."

        choices = resp.get("choices", [])
        reply = ""
        if choices:
            reply = choices[0].get("message", {}).get("content", "")
            if len(reply) > 60:
                reply = reply[:57] + "..."

        usage = resp.get("usage", {})
        tokens = usage.get("total_tokens", "?")

        print(f"  [{i}] model={model}")
        print(f"      prompt:   {content!r}")
        print(f"      response: {reply!r}")
        print(f"      tokens:   {tokens}")
        print()


def cmd_record(args):
    """Set LLMVCR_RECORD env var and run the target script."""
    os.environ["LLMVCR_MODE"] = "record"
    os.environ["LLMVCR_CASSETTE"] = args.output
    _run_script(args.script)


def cmd_playback(args):
    """Set LLMVCR_PLAYBACK env var and run the target script."""
    os.environ["LLMVCR_MODE"] = "playback"
    os.environ["LLMVCR_CASSETTE"] = args.cassette
    _run_script(args.script)


def _run_script(script_args):
    """Execute a script in a subprocess with the current environment."""
    import subprocess
    result = subprocess.run(script_args, env=os.environ)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="llmvcr",
        description="Record and replay LLM API responses for testing.",
    )
    parser.add_argument("--version", action="version", version="llmvcr 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # llmvcr info <cassette>
    info_parser = subparsers.add_parser("info", help="Inspect a cassette file")
    info_parser.add_argument("cassette", help="Path to cassette YAML file")
    info_parser.set_defaults(func=cmd_info)

    # llmvcr record --output <path> <script...>
    record_parser = subparsers.add_parser("record", help="Record API calls to a cassette")
    record_parser.add_argument("--output", "-o", required=True, help="Path to save the cassette")
    record_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    record_parser.add_argument("script", nargs=argparse.REMAINDER, help="Script to run")
    record_parser.set_defaults(func=cmd_record)

    # llmvcr playback --cassette <path> <script...>
    playback_parser = subparsers.add_parser("playback", help="Replay API calls from a cassette")
    playback_parser.add_argument("--cassette", "-c", required=True, help="Path to cassette to replay")
    playback_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    playback_parser.add_argument("script", nargs=argparse.REMAINDER, help="Script to run")
    playback_parser.set_defaults(func=cmd_playback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()