"""
CLI entry point for Speaker Gate POC server.

Usage:
    python -m poc.server serve --port 8765 --debug
    python -m poc.server profiles
    python -m poc.server delete <profile_id>
"""
import logging
import typer

app = typer.Typer(
    name="speaker-gate",
    help="Speaker Gate WebSocket Server (POC/Testing)",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8765, help="Server port"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """Start the WebSocket server for testing."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    
    # Silence noisy third-party loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    
    from .contrib import create_server
    server = create_server(host=host, port=port, debug=debug)
    server.start()


@app.command()
def profiles():
    """List all enrolled profiles."""
    from .contrib.config import get_settings
    from .core import FileProfileStore
    
    settings = get_settings()
    store = FileProfileStore(settings.profiles_dir)
    
    profile_ids = store.list_ids()
    if not profile_ids:
        typer.echo("No profiles found.")
        return
    
    typer.echo(f"Found {len(profile_ids)} profile(s):")
    for pid in profile_ids:
        profile = store.load(pid)
        if profile:
            typer.echo(
                f"  {pid}: threshold={profile.threshold:.3f}, "
                f"consistency={profile.consistency_score:.3f}, "
                f"duration={profile.duration_sec:.1f}s"
            )


@app.command()
def delete(profile_id: str):
    """Delete a profile by ID."""
    from .contrib.config import get_settings
    from .core import FileProfileStore
    
    settings = get_settings()
    store = FileProfileStore(settings.profiles_dir)
    
    if store.delete(profile_id):
        typer.echo(f"Deleted: {profile_id}")
    else:
        typer.echo(f"Not found: {profile_id}")
        raise typer.Exit(1)


@app.command()
def info():
    """Show SDK information."""
    from . import __version__
    from .contrib.config import get_settings
    
    settings = get_settings()
    
    typer.echo(f"Speaker Gate SDK v{__version__}")
    typer.echo(f"")
    typer.echo(f"Configuration:")
    typer.echo(f"  Profiles dir: {settings.profiles_dir}")
    typer.echo(f"  Debug audio dir: {settings.debug_audio_dir}")
    typer.echo(f"  Sample rate: {settings.sample_rate} Hz")
    typer.echo(f"  Default threshold: {settings.default_threshold}")
    typer.echo(f"")
    typer.echo(f"Environment variables (prefix: SPEAKER_GATE_):")
    typer.echo(f"  SPEAKER_GATE_PORT, SPEAKER_GATE_DEBUG, etc.")


if __name__ == "__main__":
    app()
