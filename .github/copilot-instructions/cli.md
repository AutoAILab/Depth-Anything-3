# CLI Development with Typer

## Command Structure
- Use `typer.Typer` for main app and subcommands
- Define commands as functions decorated with `@app.command()`
- Use meaningful command names and descriptions

## Parameter Handling
- Use `typer.Option` for optional parameters
- Use `typer.Argument` for required positional arguments
- Provide helpful help text for all parameters
- Use appropriate types (str, int, float, bool, Path)

## Error Handling
- Use `typer.echo()` for user output
- Raise `typer.BadParameter` for invalid inputs
- Use `typer.Exit()` for controlled exits
- Handle exceptions gracefully with user-friendly messages

## Best Practices
- Keep commands focused and single-purpose
- Use `--help` for all commands
- Support `--verbose` or `--quiet` flags for output control
- Implement `--version` command
- Use `typer.Option(callback=...)` for validation

## File I/O
- Use `pathlib.Path` for file paths
- Validate file existence and permissions
- Handle relative and absolute paths appropriately
- Use temporary files when needed

## Progress and Feedback
- Use `rich` or `tqdm` for progress bars
- Provide clear status messages
- Use colors and formatting for better UX