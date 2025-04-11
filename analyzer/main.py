import uvicorn
import argparse
from pathlib import Path


def main():
    """Run the TokenLense API server"""
    parser = argparse.ArgumentParser(description="TokenLense API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        data_dir.mkdir(parents=True)

    # Set environment variable for data directory
    import os
    os.environ["TOKENLENSE_DATA_DIR"] = str(data_dir)

    print(f"Starting TokenLense API on {args.host}:{args.port}")
    print(f"Using data directory: {data_dir}")

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()